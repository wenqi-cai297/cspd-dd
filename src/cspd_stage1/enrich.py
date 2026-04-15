"""Stage 1D — VLM-based caption enrichment.

Takes Stage 1C render records.jsonl, loads each image, and uses a VLM to
expand the template caption with specific visual details from the image.
The template caption provides structural constraints (subject, basic attributes)
while the VLM adds image-specific details (lighting, texture, spatial layout, etc.).

This produces richer, more diverse captions for Stage 2 LoRA training,
solving the caption homogeneity problem without changing the caption format
drastically enough to cause OOD issues.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from cspd_stage1.io_utils import write_json


ENRICH_PROMPT_TEMPLATE = (
    'You are enriching a template caption with visual details from the image.\n'
    '\n'
    'Template: "{template_caption}"\n'
    'Subject: {class_name}\n'
    '\n'
    'Rewrite as ONE sentence (under 40 words) following these rules:\n'
    '- Keep "{class_name}" as the subject, do not rename it\n'
    '- Keep the template\'s basic content (color, pose, background, viewpoint)\n'
    '- ADD details visible in the image that the template lacks:\n'
    '  specific shades/patterns, textures, lighting, background objects,\n'
    '  people or hands if present, spatial layout, weather/atmosphere\n'
    '- Do NOT invent details not visible in the image\n'
    '- Do NOT use lists or multiple sentences'
)


@dataclass(slots=True)
class EnrichResult:
    """Result of caption enrichment."""

    output_path: str
    num_enriched: int
    num_skipped: int
    num_failed: int


def _build_enrich_prompt(class_name: str, template_caption: str) -> str:
    """Build the VLM enrichment prompt for one image."""
    return ENRICH_PROMPT_TEMPLATE.format(
        template_caption=template_caption,
        class_name=class_name,
    )


def enrich_captions(
    *,
    render_input: str | Path,
    dataset_root: str | Path,
    output_path: str | Path,
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    device: str = "cuda",
    max_new_tokens: int = 100,
    batch_size: int = 1,
    resume: bool = True,
) -> EnrichResult:
    """Enrich template captions with VLM-generated visual details.

    Reads Stage 1C records.jsonl, enriches each caption using the VLM,
    and writes records_enriched.jsonl with the enriched canonical_caption.

    Args:
        render_input: Path to Stage 1C records.jsonl.
        dataset_root: ImageFolder dataset root (for resolving image paths).
        output_path: Path for enriched output JSONL.
        model_name: VLM model identifier.
        device: Torch device.
        max_new_tokens: Max tokens for VLM generation.
        batch_size: Not used yet (single-image processing).
        resume: Skip already-enriched records if output file exists.

    Returns:
        EnrichResult with counts.
    """
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from qwen_vl_utils import process_vision_info

    render_input = Path(render_input)
    dataset_root = Path(dataset_root)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load render records
    records: list[dict[str, Any]] = []
    with open(render_input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"[Stage 1D] Loaded {len(records)} render records")

    # Resume: load already-enriched record_ids
    enriched_ids: set[str] = set()
    if resume and output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    rid = row.get("record_id", "")
                    if rid:
                        enriched_ids.add(rid)
        print(f"[Stage 1D] Resume: {len(enriched_ids)} records already enriched")

    # Filter to records that need enrichment
    to_enrich = [
        r for r in records
        if r.get("render_status") == "success"
        and r.get("record_id", "") not in enriched_ids
    ]
    skipped = len(records) - len(to_enrich) - len([r for r in records if r.get("render_status") != "success"])
    print(f"[Stage 1D] To enrich: {len(to_enrich)}, already done: {len(enriched_ids)}")

    if not to_enrich:
        print("[Stage 1D] Nothing to enrich.")
        return EnrichResult(
            output_path=str(output_path),
            num_enriched=0,
            num_skipped=len(enriched_ids),
            num_failed=0,
        )

    # Load VLM
    print(f"[Stage 1D] Loading VLM: {model_name}...")
    vlm = AutoModelForImageTextToText.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map=device,
    )
    processor = AutoProcessor.from_pretrained(model_name)

    # Process records
    num_enriched = 0
    num_failed = 0

    # Open output file in append mode for resume support
    with open(output_path, "a", encoding="utf-8") as out_f:
        for record in tqdm(to_enrich, desc="Enriching captions"):
            record_id = record.get("record_id", "")
            class_name = record.get("class_name", "")
            template_caption = record.get("canonical_caption", "")

            # Resolve image path
            parts = record_id.split("::", 1)
            if len(parts) == 2:
                image_path = dataset_root / parts[1]
            else:
                image_path = dataset_root / record.get("sample_id", "")

            if not image_path.exists():
                num_failed += 1
                continue

            # Build prompt
            prompt = _build_enrich_prompt(class_name, template_caption)

            try:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": f"file://{image_path}"},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]

                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text], images=image_inputs, videos=video_inputs,
                    padding=True, return_tensors="pt",
                ).to(device)

                generated_ids = vlm.generate(**inputs, max_new_tokens=max_new_tokens)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                enriched_caption = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0].strip()

                # Write enriched record
                enriched_record = {**record}
                enriched_record["template_caption"] = template_caption
                enriched_record["canonical_caption"] = enriched_caption
                enriched_record["enrichment_status"] = "success"

                out_f.write(json.dumps(enriched_record, ensure_ascii=False) + "\n")
                num_enriched += 1

            except Exception as e:
                # On failure, write original record with failure status
                failed_record = {**record}
                failed_record["enrichment_status"] = "failed"
                failed_record["enrichment_error"] = str(e)
                out_f.write(json.dumps(failed_record, ensure_ascii=False) + "\n")
                num_failed += 1

            # Flush periodically
            if (num_enriched + num_failed) % 50 == 0:
                out_f.flush()

    # Free VLM
    del vlm, processor
    torch.cuda.empty_cache()

    print(f"[Stage 1D] Enrichment complete: {num_enriched} enriched, {num_failed} failed")
    print(f"[Stage 1D] Output: {output_path}")

    return EnrichResult(
        output_path=str(output_path),
        num_enriched=num_enriched,
        num_skipped=len(enriched_ids),
        num_failed=num_failed,
    )
