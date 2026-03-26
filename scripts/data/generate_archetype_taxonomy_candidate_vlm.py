"""Generate an archetype taxonomy candidate from a full classes.json file using a local Qwen model.

Input:
    classes.json
        {
          "n01440764": "tench, Tinca tinca",
          ...
        }

Output:
    archetype_taxonomy_candidate.json
        {
          "archetypes": [
            {
              "name": "animal",
              "definition": "...",
              "inclusion_guidelines": ["..."],
              "example_classes": ["..."]
            }
          ],
          "notes": ["..."]
        }

This script asks the VLM to inspect the full class list and propose a compact,
reasonably complete archetype taxonomy for downstream Stage 1 schema design.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cspd_stage1.vlm.json_utils import parse_json_object

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
SYSTEM_PROMPT = (
    "You are designing a semantic archetype taxonomy for image dataset classes. "
    "Return JSON only. Do not include markdown or explanations outside JSON."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an archetype taxonomy candidate from classes.json")
    parser.add_argument("--input", required=True, help="Path to classes.json")
    parser.add_argument("--output", required=True, help="Path to output taxonomy JSON")
    parser.add_argument("--model-name", default=MODEL_NAME, help="Local Qwen model name")
    parser.add_argument("--torch-dtype", default="float16", help="Torch dtype for local model loading")
    parser.add_argument("--device-map", default="auto", help="Transformers device_map")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Generation length cap")
    parser.add_argument("--max-classes-in-prompt", type=int, default=1000, help="Maximum number of classes included in the prompt")
    return parser


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


def build_user_prompt(classes: dict[str, str], max_classes_in_prompt: int) -> str:
    items = list(classes.items())[:max_classes_in_prompt]
    classes_payload = [{"raw_label": raw_label, "readable_name": readable_name} for raw_label, readable_name in items]
    template = {
        "archetypes": [
            {
                "name": "animal",
                "definition": "living creature classes with clear biological identity",
                "inclusion_guidelines": ["use for mammals, birds, fish, reptiles, insects"],
                "example_classes": ["tench, Tinca tinca", "goldfish, Carassius auratus"],
            }
        ],
        "notes": [
            "Keep the archetype list compact but expressive.",
            "Avoid overlapping archetypes when possible.",
        ],
    }
    return (
        "You are given dataset class names from an ImageNet-style classification dataset.\n"
        "Your task is to propose a reasonable semantic archetype taxonomy for downstream structured attribute extraction.\n"
        "The taxonomy should be compact, expressive, and stable enough to support fixed schemas per archetype.\n"
        "Do NOT classify each class individually yet. First design the archetype system itself.\n"
        "Requirements:\n"
        "- Prefer roughly 8 to 20 archetypes if justified by the class set\n"
        "- Avoid a single over-broad generic bucket unless truly necessary\n"
        "- Keep archetypes semantically interpretable and visually useful\n"
        "- Archetypes should be suitable for later slot-schema design\n"
        "- Use example_classes to illustrate each archetype\n"
        "- Return JSON only\n"
        "Return JSON in this exact top-level structure:\n"
        f"{json.dumps(template, ensure_ascii=False, indent=2)}\n"
        "Dataset classes:\n"
        f"{json.dumps(classes_payload, ensure_ascii=False, indent=2)}\n"
    )


def generate_taxonomy(model, processor, user_prompt: str, max_new_tokens: int) -> tuple[dict, str]:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], padding=True, return_tensors="pt")
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
    return payload, output_text


def main() -> None:
    args = build_parser().parse_args()
    classes = json.loads(Path(args.input).read_text(encoding="utf-8-sig"))
    if not isinstance(classes, dict):
        raise ValueError("Input classes file must be a JSON object")

    model, processor = load_local_qwen(args.model_name, args.torch_dtype, args.device_map)
    user_prompt = build_user_prompt(classes, args.max_classes_in_prompt)
    payload, raw_text = generate_taxonomy(model, processor, user_prompt, args.max_new_tokens)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_payload = {
        "source_classes_count": len(classes),
        "classes_in_prompt": min(len(classes), args.max_classes_in_prompt),
        "model_name": args.model_name,
        "taxonomy_candidate": payload,
        "raw_response": raw_text,
    }
    output_path.write_text(json.dumps(final_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Wrote taxonomy candidate to {output_path}")


if __name__ == "__main__":
    main()
