"""Single-image inference test for local Qwen2.5-VL attribute extraction.

Purpose:
- run one real image through the local VLM,
- prompt it with the Stage 1 attribute schema,
- inspect the raw text output,
- check whether the output is valid JSON.

Usage example on the server:
    python scripts/vlm/test_single_image_infer.py --image /path/to/example.jpg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

# Stage 1 prompt contract: JSON only, fixed schema, no reasoning text.
SYSTEM_PROMPT = (
    "You are a vision-language attribute extractor for dataset distillation. "
    "Inspect the given image and output JSON only. "
    "Never include reasoning. Never hallucinate invisible attributes. "
    "If a field is unclear, use 'unknown'. If not applicable, use 'not_applicable'."
)

USER_PROMPT_TEMPLATE = """
Class name: {class_name}
Class id: {class_id}

Return a JSON object with exactly these fields:
- subject: short phrase
- color: short phrase
- shape_or_body_trait: short phrase
- action_or_pose_or_state: short phrase
- background_or_context: short phrase
- viewpoint: short phrase
- material: short phrase

Rules:
- Output JSON only
- Use short phrases, not sentences
- Keep semantics image-grounded
- Use unknown / not_applicable when necessary
""".strip()


def build_parser() -> argparse.ArgumentParser:
    """Build CLI args for the single-image inference smoke test."""
    parser = argparse.ArgumentParser(description="Run local Qwen2.5-VL on one test image.")
    parser.add_argument("--image", required=True, help="Path to the test image")
    parser.add_argument("--class-name", default="unknown", help="Optional class name hint")
    parser.add_argument("--class-id", type=int, default=-1, help="Optional class id hint")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Generation length cap")
    return parser


def load_image(image_path: str) -> Image.Image:
    """Open the target image as RGB and fail early if the path is wrong."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


def main() -> None:
    """Load model, run one image through it, and print raw + parsed outputs."""
    args = build_parser().parse_args()

    print("Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    image = load_image(args.image)
    user_prompt = USER_PROMPT_TEMPLATE.format(class_name=args.class_name, class_id=args.class_id)

    # Qwen-VL expects a chat-style multimodal message list.
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )

    # Move tensor inputs onto the same device placement used by the model.
    inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}

    print("Generating...")
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )

    # Only decode newly generated tokens, not the original prompt tokens.
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    print("\n===== RAW OUTPUT =====")
    print(output_text)

    print("\n===== JSON PARSE TEST =====")
    cleaned_text = output_text.strip()

    # Some VLMs wrap JSON in markdown code fences or prepend invisible junk.
    # We strip the most common wrappers first so the parse test reflects the
    # model's actual structured output quality rather than formatting noise.
    if cleaned_text.startswith("```json"):
        cleaned_text = cleaned_text[len("```json"):].strip()
    elif cleaned_text.startswith("```"):
        cleaned_text = cleaned_text[len("```"):].strip()

    if cleaned_text.endswith("```"):
        cleaned_text = cleaned_text[:-3].strip()

    print("RAW REPR:", repr(cleaned_text[:200]))

    try:
        parsed = json.loads(cleaned_text)
        print(json.dumps(parsed, ensure_ascii=False, indent=2))
    except Exception as exc:  # noqa: BLE001
        print("JSON parse failed:", exc)


if __name__ == "__main__":
    main()
