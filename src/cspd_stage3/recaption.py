"""Stage 3D — Re-caption medoid images with VLM for richer descriptions.

Takes the modes_index.json from Stage 3B/3C clustering, loads each medoid image,
and generates a detailed free-form caption using a VLM (Qwen2.5-VL by default).
The original template-based representative_caption is preserved as
`original_caption`, and the new VLM caption becomes `representative_caption`.

This enriches the text diversity for Stage 4 text2img generation.
"""

from __future__ import annotations

import json
from pathlib import Path

from cspd_stage1.io_utils import write_json


def _build_recaption_prompt(class_name: str, archetype: str, original_caption: str) -> str:
    """Build a grounded recaption prompt using known attributes from Stage 1."""
    return (
        f"This image shows a {class_name} (category: {archetype}).\n"
        f"A template description is: \"{original_caption}\"\n\n"
        f"Based on what you see in the image, write a single detailed sentence "
        f"describing this {class_name}. Keep the class name \"{class_name}\" as the subject. "
        f"Add specific visual details the template misses: lighting, texture, composition, "
        f"surrounding objects, spatial relationships, and atmosphere. "
        f"Be concrete and specific to THIS particular image, not generic."
    )


def recaption_modes(
    *,
    modes_dir: str | Path,
    encode_dir: str | Path,
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    device: str = "cuda",
    max_new_tokens: int = 150,
) -> None:
    """Re-caption medoid images with VLM and update modes_index.json.

    Args:
        modes_dir: Directory containing modes_index.json.
        encode_dir: Directory containing encode_index.json (for image paths).
        model_name: VLM model identifier.
        device: Torch device.
        max_new_tokens: Max tokens for VLM generation.
    """
    modes_dir = Path(modes_dir)
    encode_dir = Path(encode_dir)

    # Load modes index
    modes_index_path = modes_dir / "modes_index.json"
    with open(modes_index_path, encoding="utf-8") as f:
        modes_index = json.load(f)

    modes_list = modes_index.get("modes", [])
    if not modes_list:
        print("[Recaption] No modes found in modes_index.json")
        return

    # Load encode index for image paths
    with open(encode_dir / "encode_index.json", encoding="utf-8") as f:
        encode_index = json.load(f)
    samples = encode_index.get("samples", [])

    # Build record_id → image_path lookup
    record_to_path: dict[str, str] = {}
    for s in samples:
        rid = s.get("record_id", "")
        if rid:
            record_to_path[rid] = s.get("image_path", "")

    # Collect medoid images to re-caption
    medoid_items: list[tuple[int, str, dict]] = []  # (mode_idx, image_path, mode_meta)
    for i, mode in enumerate(modes_list):
        record_id = mode.get("medoid_record_id", mode.get("visual_medoid_record_id", ""))
        image_path = record_to_path.get(record_id, "")
        if image_path and Path(image_path).exists():
            medoid_items.append((i, image_path, mode))
        else:
            print(f"[Recaption] WARNING: Image not found for mode {i} ({record_id})")

    print(f"[Recaption] Re-captioning {len(medoid_items)} medoid images with {model_name}...")

    # Load VLM
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from qwen_vl_utils import process_vision_info

    vlm = AutoModelForImageTextToText.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map=device,
    )
    processor = AutoProcessor.from_pretrained(model_name)

    # Re-caption each medoid
    from tqdm.auto import tqdm
    for mode_idx, image_path, mode_meta in tqdm(medoid_items, desc="Recaptioning"):
        class_name = mode_meta.get("class_name", "object")
        archetype = mode_meta.get("archetype", "")
        original_caption = mode_meta.get("representative_caption", "")

        prompt = _build_recaption_prompt(class_name, archetype, original_caption)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(device)

        generated_ids = vlm.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0].strip()

        # Preserve original caption, update with VLM caption
        mode = modes_list[mode_idx]
        mode["original_caption"] = mode.get("representative_caption", "")
        mode["representative_caption"] = output_text

    # Free VLM
    del vlm, processor
    torch.cuda.empty_cache()

    # Save updated modes_index.json
    write_json(modes_index_path, modes_index)
    print(f"[Recaption] Updated {len(medoid_items)} captions in {modes_index_path}")
    print(f"[Recaption] Original captions preserved in 'original_caption' field")
