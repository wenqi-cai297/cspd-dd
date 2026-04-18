"""Generate a raw-class-label -> archetype mapping using class text plus sampled class images.

This script classifies each class into exactly one fixed archetype from the
manual taxonomy. Unlike the earlier text-only version, it now uses multimodal
class evidence:
- readable class name / raw label text
- a small sample of images from that class directory

The goal is to reduce label-only ambiguity for ImageNet-style classes whose
names are polysemous or visually counterintuitive.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from cspd_stage1.io_utils import append_jsonl, write_jsonl
from cspd_stage1.vlm.json_utils import parse_json_object

DEFAULT_TAXONOMY_PATH = "configs/stage1/archetype_taxonomy_manual.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".JPEG", ".JPG", ".PNG", ".WEBP"}
SYSTEM_PROMPT = (
    "You are classifying ImageNet-style classes into a fixed manual semantic archetype set. "
    "Use both the class text and the sampled class images as evidence. "
    "The class name can be ambiguous, so do not rely on text alone when the images clearly indicate a different sense. "
    "Return JSON only."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate class->archetype mapping using multimodal class evidence")
    parser.add_argument("--input", required=True, help="Path to classes.json")
    parser.add_argument("--dataset-root", required=True, help="ImageFolder dataset root used to sample class images")
    parser.add_argument("--output", required=True, help="Path to output mapping JSON")
    parser.add_argument("--detail-output", required=True, help="Path to output detail JSONL")
    parser.add_argument("--taxonomy", default=DEFAULT_TAXONOMY_PATH, help="Path to fixed taxonomy JSON")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct", help="Local Qwen model name")
    parser.add_argument("--torch-dtype", default="float16", help="Torch dtype for local model loading")
    parser.add_argument("--device-map", default="auto", help="Transformers device_map")
    parser.add_argument("--images-per-class", type=int, default=5, help="Number of class images to sample per class")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Generation cap for the local model")
    parser.add_argument("--disable-fast-processor", action="store_true", help="Use the slower processor implementation")
    parser.add_argument("--flush-every", type=int, default=20, help="Flush intermediate JSONL rows every N classes")
    return parser


def load_taxonomy(path: str) -> tuple[list[str], dict[str, dict[str, Any]]]:
    taxonomy_path = Path(path)
    payload = json.loads(taxonomy_path.read_text(encoding="utf-8-sig"))
    archetypes = payload.get("archetypes", [])
    if not isinstance(archetypes, list) or not archetypes:
        raise ValueError("Taxonomy file must contain a non-empty 'archetypes' list")
    names: list[str] = []
    details: dict[str, dict[str, Any]] = {}
    for item in archetypes:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ValueError("Each taxonomy archetype entry must be an object with a string 'name'")
        name = item["name"].strip()
        names.append(name)
        details[name] = item
    return names, details


def build_user_prompt(
    raw_label: str,
    readable_name: str,
    allowed_archetypes: list[str],
    taxonomy_details: dict[str, dict[str, Any]],
    sampled_image_paths: list[str],
) -> str:
    short_taxonomy = []
    for name in allowed_archetypes:
        item = taxonomy_details[name]
        short_taxonomy.append(
            {
                "name": name,
                "definition": item.get("definition", ""),
                "includes": item.get("includes", [])[:4],
                "excludes": item.get("excludes", [])[:4],
                "notes": item.get("notes", [])[:3],
            }
        )
    template = {
        "raw_label": raw_label,
        "readable_name": readable_name,
        "predicted_archetype": f"one of: {' | '.join(allowed_archetypes)}",
        "confidence": "high | medium | low",
        "reason": "short justification based on class text plus the majority visual evidence",
    }
    return (
        "Classify the following dataset class into exactly one semantic archetype from the fixed manual taxonomy.\n"
        "Use BOTH sources of evidence:\n"
        "1) the class label text\n"
        "2) the sampled images for this class\n"
        "If the class label is ambiguous, prefer the sense supported by the majority of sampled images.\n"
        "Do not invent new archetypes.\n"
        f"Allowed archetypes: {', '.join(allowed_archetypes)}\n"
        f"Sampled image count: {len(sampled_image_paths)}\n"
        f"Sampled image paths: {json.dumps(sampled_image_paths, ensure_ascii=False)}\n"
        "Fixed taxonomy reference:\n"
        f"{json.dumps(short_taxonomy, ensure_ascii=False, indent=2)}\n"
        "Return JSON only in this exact structure:\n"
        f"{json.dumps(template, ensure_ascii=False, indent=2)}\n"
        "Rules:\n"
        "- Output JSON only\n"
        "- Keep raw_label and readable_name unchanged\n"
        "- predicted_archetype must be exactly one allowed label\n"
        "- reason must be short and grounded in the majority visual evidence\n"
    )


def load_local_qwen(model_name: str, torch_dtype: str, device_map: str, use_fast_processor: bool):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    normalized = torch_dtype.strip().lower()
    if normalized not in dtype_map:
        raise ValueError(f"Unsupported torch dtype: {torch_dtype}")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype_map[normalized],
        device_map=device_map,
    )
    processor = AutoProcessor.from_pretrained(model_name, use_fast=use_fast_processor)
    return model, processor


def collect_class_image_paths(dataset_root: str, raw_label: str, images_per_class: int) -> list[Path]:
    class_dir = Path(dataset_root) / raw_label
    if not class_dir.exists():
        raise FileNotFoundError(f"Class directory not found for raw label '{raw_label}': {class_dir}")
    if not class_dir.is_dir():
        raise NotADirectoryError(f"Class path is not a directory for raw label '{raw_label}': {class_dir}")

    image_paths = sorted(
        [path for path in class_dir.rglob("*") if path.is_file() and path.suffix.lower() in {ext.lower() for ext in IMAGE_EXTENSIONS}]
    )
    if not image_paths:
        raise ValueError(f"No images found under class directory: {class_dir}")
    return image_paths[: max(images_per_class, 1)]


def classify_one(
    model,
    processor,
    dataset_root: str,
    raw_label: str,
    readable_name: str,
    allowed_archetypes: list[str],
    taxonomy_details: dict[str, dict[str, Any]],
    images_per_class: int,
    max_new_tokens: int,
) -> tuple[dict[str, Any], str, list[str]]:
    image_paths = collect_class_image_paths(dataset_root, raw_label, images_per_class)
    sampled_images = [Image.open(path).convert("RGB") for path in image_paths]
    sampled_image_paths = [str(path) for path in image_paths]

    user_prompt = build_user_prompt(
        raw_label=raw_label,
        readable_name=readable_name,
        allowed_archetypes=allowed_archetypes,
        taxonomy_details=taxonomy_details,
        sampled_image_paths=sampled_image_paths,
    )

    try:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": image} for image in sampled_images],
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=sampled_images, padding=True, return_tensors="pt")
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        payload = parse_json_object(output_text)
        return payload, output_text, sampled_image_paths
    finally:
        for image in sampled_images:
            image.close()



def main() -> None:
    args = build_parser().parse_args()
    classes = json.loads(Path(args.input).read_text(encoding="utf-8-sig"))
    if not isinstance(classes, dict):
        raise ValueError("Input classes file must be a JSON object")

    allowed_archetypes, taxonomy_details = load_taxonomy(args.taxonomy)
    model, processor = load_local_qwen(
        args.model_name,
        args.torch_dtype,
        args.device_map,
        use_fast_processor=not args.disable_fast_processor,
    )

    output_path = Path(args.output)
    detail_output_path = Path(args.detail_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    detail_output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(detail_output_path, [])

    fallback = "household_object" if "household_object" in allowed_archetypes else allowed_archetypes[-1]
    final_mapping: dict[str, str] = {}
    pending_rows: list[dict[str, Any]] = []

    items = list(classes.items())
    for index, (raw_label, readable_name) in enumerate(items, start=1):
        try:
            payload, raw_output, sampled_image_paths = classify_one(
                model=model,
                processor=processor,
                dataset_root=args.dataset_root,
                raw_label=str(raw_label),
                readable_name=str(readable_name),
                allowed_archetypes=allowed_archetypes,
                taxonomy_details=taxonomy_details,
                images_per_class=args.images_per_class,
                max_new_tokens=args.max_new_tokens,
            )
            predicted = str(payload.get("predicted_archetype", fallback)).strip()
            if predicted not in allowed_archetypes:
                predicted = fallback
            final_mapping[str(raw_label)] = predicted
            pending_rows.append(
                {
                    "raw_label": str(raw_label),
                    "readable_name": str(readable_name),
                    "predicted_archetype": predicted,
                    "confidence": payload.get("confidence"),
                    "reason": payload.get("reason"),
                    "sampled_image_paths": sampled_image_paths,
                    "status": "success",
                    "raw_response": raw_output,
                }
            )
        except Exception as exc:  # noqa: BLE001
            final_mapping[str(raw_label)] = fallback
            pending_rows.append(
                {
                    "raw_label": str(raw_label),
                    "readable_name": str(readable_name),
                    "predicted_archetype": fallback,
                    "sampled_image_paths": [],
                    "status": f"failed_fallback_{fallback}",
                    "error_message": str(exc),
                }
            )

        if index % max(args.flush_every, 1) == 0 or index == len(items):
            append_jsonl(detail_output_path, pending_rows)
            pending_rows.clear()
            print(f"[progress] {index}/{len(items)} classes processed")

    output_path.write_text(json.dumps(final_mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    counts = Counter(final_mapping.values())
    print(f"[OK] Wrote {len(final_mapping)} entries to {output_path}")
    print("[INFO] Archetype counts:")
    for archetype, count in sorted(counts.items()):
        print(f"  - {archetype}: {count}")


if __name__ == "__main__":
    main()
