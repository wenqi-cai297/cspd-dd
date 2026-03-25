"""Generate a raw-class-label -> archetype mapping from classes.json using a local Qwen model.

This script is meant to replace the crude keyword heuristic with an explicit
model-based class-level categorization pass.

Input:
    {
      "n01440764": "tench, Tinca tinca",
      ...
    }

Output artifacts:
- mapping JSON: raw class label -> archetype
- detail JSONL: one row per class with readable name, raw model output, etc.

Example:
    python scripts/data/generate_class_to_archetype_map_vlm.py \
      --input /path/to/classes.json \
      --output /path/to/class_to_archetype.json \
      --detail-output /path/to/class_to_archetype_details.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from cspd_stage1.io_utils import append_jsonl, write_jsonl
from cspd_stage1.vlm.json_utils import parse_json_object

ALLOWED_ARCHETYPES = [
    "animal",
    "vehicle",
    "food",
    "instrument",
    "generic_object",
]

SYSTEM_PROMPT = (
    "You are classifying ImageNet-style class names into a small fixed semantic archetype set. "
    "Return JSON only. Do not include explanations."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate class->archetype mapping using a local Qwen model")
    parser.add_argument("--input", required=True, help="Path to classes.json")
    parser.add_argument("--output", required=True, help="Path to output mapping JSON")
    parser.add_argument("--detail-output", required=True, help="Path to output detail JSONL")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct", help="Local Qwen model name")
    parser.add_argument("--torch-dtype", default="float16", help="Torch dtype for local model loading")
    parser.add_argument("--device-map", default="auto", help="Transformers device_map")
    parser.add_argument("--flush-every", type=int, default=20, help="Flush intermediate JSONL rows every N classes")
    return parser


def build_user_prompt(raw_label: str, readable_name: str) -> str:
    template = {
        "raw_label": raw_label,
        "readable_name": readable_name,
        "archetype": "one of: animal | vehicle | food | instrument | generic_object",
    }
    return (
        "Classify the following dataset class into exactly one semantic archetype.\n"
        f"Allowed archetypes: {', '.join(ALLOWED_ARCHETYPES)}\n"
        "Use the readable class name as the primary evidence.\n"
        "If the class is a living creature, choose animal.\n"
        "If it is a device, furniture, tool, clothing item, place, sign, container, or general artifact, "
        "choose generic_object unless vehicle / instrument / food is clearly more specific.\n"
        "Return JSON only in this exact structure:\n"
        f"{json.dumps(template, ensure_ascii=False, indent=2)}\n"
        "Rules:\n"
        "- Output JSON only\n"
        "- Keep raw_label and readable_name unchanged\n"
        "- archetype must be exactly one allowed label\n"
    )


def load_local_qwen(model_name: str, torch_dtype: str, device_map: str):
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
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def classify_one(model, processor, raw_label: str, readable_name: str) -> tuple[dict, str]:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": build_user_prompt(raw_label, readable_name)},
            ],
        },
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], padding=True, return_tensors="pt")
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    payload = parse_json_object(output_text)
    return payload, output_text


def main() -> None:
    args = build_parser().parse_args()
    classes = json.loads(Path(args.input).read_text(encoding="utf-8-sig"))
    if not isinstance(classes, dict):
        raise ValueError("Input classes file must be a JSON object")

    model, processor = load_local_qwen(args.model_name, args.torch_dtype, args.device_map)

    output_path = Path(args.output)
    detail_output_path = Path(args.detail_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    detail_output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(detail_output_path, [])

    final_mapping: dict[str, str] = {}
    pending_rows: list[dict] = []

    items = list(classes.items())
    for index, (raw_label, readable_name) in enumerate(items, start=1):
        try:
            payload, raw_output = classify_one(model, processor, str(raw_label), str(readable_name))
            predicted = str(payload.get("archetype", "generic_object")).strip()
            if predicted not in ALLOWED_ARCHETYPES:
                predicted = "generic_object"
            final_mapping[str(raw_label)] = predicted
            pending_rows.append(
                {
                    "raw_label": str(raw_label),
                    "readable_name": str(readable_name),
                    "archetype": predicted,
                    "status": "success",
                    "raw_response": raw_output,
                }
            )
        except Exception as exc:  # noqa: BLE001
            final_mapping[str(raw_label)] = "generic_object"
            pending_rows.append(
                {
                    "raw_label": str(raw_label),
                    "readable_name": str(readable_name),
                    "archetype": "generic_object",
                    "status": "failed_fallback_generic_object",
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
