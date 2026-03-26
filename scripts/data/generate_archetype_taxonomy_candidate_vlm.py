"""Generate an archetype taxonomy candidate using a multi-round local Qwen workflow.

This script is designed to reduce single-step pressure on the VLM:
1. Round 1 reads the full classes.json and writes a global summary.
2. Later rounds propose a few new archetypes at a time.
3. Each round writes a summary file into a task-specific timestamp directory.
4. Each new archetype is checked against existing ones for obvious conflicts before acceptance.

Outputs are stored under a task directory like:
    runs/taxonomy_tasks/2026-03-26_133500_taxonomy/
      round_001_summary.json
      round_002_summary.json
      ...
      archetype_taxonomy_candidate.json
      conflict_checks.jsonl
      review.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from cspd_stage1.io_utils import append_jsonl, read_jsonl, write_json, write_jsonl
from cspd_stage1.vlm.json_utils import parse_json_object

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
SYSTEM_PROMPT = (
    "You are designing a semantic archetype taxonomy for image dataset classes. "
    "Return JSON only. Do not include markdown or explanations outside JSON."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an archetype taxonomy candidate from classes.json")
    parser.add_argument("--input", required=True, help="Path to classes.json")
    parser.add_argument(
        "--task-dir",
        default=None,
        help="Optional task output directory. If omitted, create runs/taxonomy_tasks/<timestamp>_taxonomy",
    )
    parser.add_argument("--model-name", default=MODEL_NAME, help="Local Qwen model name")
    parser.add_argument("--torch-dtype", default="float16", help="Torch dtype for local model loading")
    parser.add_argument("--device-map", default="auto", help="Transformers device_map")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Generation length cap")
    parser.add_argument(
        "--max-classes-in-prompt", type=int, default=1000, help="Maximum number of classes included in the prompt"
    )
    parser.add_argument(
        "--new-archetypes-per-round",
        type=int,
        default=3,
        help="How many new archetypes to request per proposal round",
    )
    parser.add_argument(
        "--proposal-rounds",
        type=int,
        default=4,
        help="How many proposal rounds to run after the initial global summary round",
    )
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


def build_classes_payload(classes: dict[str, str], max_classes_in_prompt: int) -> list[dict[str, str]]:
    items = list(classes.items())[:max_classes_in_prompt]
    return [{"raw_label": raw_label, "readable_name": readable_name} for raw_label, readable_name in items]


def build_round1_prompt(classes_payload: list[dict[str, str]]) -> str:
    template = {
        "major_regions": [
            "animals",
            "vehicles",
            "foods",
            "tools_and_devices",
            "furniture_and_household_objects",
        ],
        "taxonomy_design_principles": [
            "keep archetypes at a similar abstraction level",
            "cover both animal and non-animal regions",
        ],
        "suggested_archetype_count_range": {"min": 8, "max": 20},
        "coverage_gaps_to_watch": ["plants", "clothing", "containers"],
        "notes": ["global first-pass summary only; do not finalize taxonomy yet"],
    }
    return (
        "You are given the full class list of an ImageNet-style dataset.\n"
        "Round 1 goal: build a global understanding of the class space before proposing final archetypes.\n"
        "Do NOT output a final taxonomy yet.\n"
        "Instead, summarize the major semantic regions, the desired abstraction level, and important coverage gaps to watch.\n"
        "Requirements:\n"
        "- Read the full class list provided below\n"
        "- Cover both animal and non-animal regions\n"
        "- Keep the intended archetype level consistent\n"
        "- Return JSON only\n"
        "Return JSON in this exact top-level structure:\n"
        f"{json.dumps(template, ensure_ascii=False, indent=2)}\n"
        "Dataset classes:\n"
        f"{json.dumps(classes_payload, ensure_ascii=False, indent=2)}\n"
    )


def build_proposal_prompt(
    classes_payload: list[dict[str, str]],
    latest_summary: dict[str, Any],
    accepted_archetypes: list[dict[str, Any]],
    new_archetypes_per_round: int,
    round_index: int,
) -> str:
    template = {
        "round_index": round_index,
        "new_archetypes": [
            {
                "name": "vehicle",
                "definition": "mobile man-made transport categories with distinct physical form",
                "inclusion_guidelines": ["cars, trucks, buses, boats, aircraft"],
                "example_classes": ["sports car", "school bus", "airliner"],
                "conflict_risk_with_existing": ["device"],
                "why_needed": "covers a major non-animal semantic region",
            }
        ],
        "remaining_regions_to_cover": ["plants", "furniture", "clothing"],
        "notes": ["propose only a small number of new archetypes this round"],
    }
    return (
        f"You are continuing taxonomy design for round {round_index}.\n"
        "You have already read the full class set, and you are now proposing only a few new archetypes.\n"
        "Important constraints:\n"
        "- Propose only archetypes at the same abstraction level as existing accepted archetypes\n"
        "- Do not create parent-child conflicts with accepted archetypes\n"
        "- Prefer major uncovered semantic regions\n"
        f"- Propose at most {new_archetypes_per_round} new archetypes this round\n"
        "- Return JSON only\n"
        "Current accepted archetypes:\n"
        f"{json.dumps(accepted_archetypes, ensure_ascii=False, indent=2)}\n"
        "Latest round summary:\n"
        f"{json.dumps(latest_summary, ensure_ascii=False, indent=2)}\n"
        "Dataset classes:\n"
        f"{json.dumps(classes_payload, ensure_ascii=False, indent=2)}\n"
        "Return JSON in this exact top-level structure:\n"
        f"{json.dumps(template, ensure_ascii=False, indent=2)}\n"
    )


def build_review_prompt(classes_payload: list[dict[str, str]], accepted_archetypes: list[dict[str, Any]]) -> str:
    template = {
        "coverage_assessment": {
            "animal_regions_covered": True,
            "non_animal_regions_covered": True,
            "major_missing_regions": [],
        },
        "overlap_risks": ["possible overlap between tool and device"],
        "final_notes": ["taxonomy is usable as a candidate for manual review"],
    }
    return (
        "Perform a final review of the current taxonomy candidate.\n"
        "Check coverage, overlap, and abstraction-level consistency.\n"
        "Do not invent new archetypes here unless necessary; focus on review.\n"
        "Return JSON only in this exact top-level structure:\n"
        f"{json.dumps(template, ensure_ascii=False, indent=2)}\n"
        "Accepted archetypes:\n"
        f"{json.dumps(accepted_archetypes, ensure_ascii=False, indent=2)}\n"
        "Dataset classes:\n"
        f"{json.dumps(classes_payload, ensure_ascii=False, indent=2)}\n"
    )


def run_text_round(model, processor, user_prompt: str, max_new_tokens: int) -> tuple[dict[str, Any], str]:
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


def normalize_archetype_name(name: str) -> str:
    return "_".join(name.strip().lower().split())


def conflict_check(candidate: dict[str, Any], accepted: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_name = normalize_archetype_name(str(candidate.get("name", "")))
    candidate_definition = str(candidate.get("definition", "")).strip().lower()
    conflicts: list[str] = []
    if not candidate_name:
        conflicts.append("empty_name")
    for existing in accepted:
        existing_name = normalize_archetype_name(str(existing.get("name", "")))
        if candidate_name == existing_name:
            conflicts.append(f"duplicate_name:{existing_name}")
            continue
        if candidate_name in existing_name or existing_name in candidate_name:
            conflicts.append(f"possible_parent_child_name_overlap:{existing_name}")
        existing_definition = str(existing.get("definition", "")).strip().lower()
        if candidate_definition and existing_definition and candidate_definition == existing_definition:
            conflicts.append(f"duplicate_definition:{existing_name}")
    return {
        "candidate_name": candidate_name,
        "accepted": len(conflicts) == 0,
        "conflicts": conflicts,
    }


def find_latest_summary(task_dir: Path) -> dict[str, Any] | None:
    summary_files = sorted(task_dir.glob("round_*_summary.json"))
    if not summary_files:
        return None
    return json.loads(summary_files[-1].read_text(encoding="utf-8-sig"))


def create_task_dir(task_dir_arg: str | None) -> Path:
    if task_dir_arg:
        task_dir = Path(task_dir_arg)
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        task_dir = Path("runs") / "taxonomy_tasks" / f"{timestamp}_taxonomy"
    task_dir.mkdir(parents=True, exist_ok=True)
    return task_dir


def main() -> None:
    args = build_parser().parse_args()
    classes = json.loads(Path(args.input).read_text(encoding="utf-8-sig"))
    if not isinstance(classes, dict):
        raise ValueError("Input classes file must be a JSON object")

    task_dir = create_task_dir(args.task_dir)
    candidate_path = task_dir / "archetype_taxonomy_candidate.json"
    conflict_path = task_dir / "conflict_checks.jsonl"
    review_path = task_dir / "review.json"

    if not conflict_path.exists():
        write_jsonl(conflict_path, [])

    if candidate_path.exists():
        accepted_archetypes = json.loads(candidate_path.read_text(encoding="utf-8-sig")).get("archetypes", [])
    else:
        accepted_archetypes = []
        write_json(candidate_path, {"archetypes": []})

    classes_payload = build_classes_payload(classes, args.max_classes_in_prompt)
    model, processor = load_local_qwen(args.model_name, args.torch_dtype, args.device_map)

    latest_summary = find_latest_summary(task_dir)
    round_index = 1 if latest_summary is None else int(latest_summary.get("round_index", 0)) + 1

    if latest_summary is None:
        round1_prompt = build_round1_prompt(classes_payload)
        round1_payload, round1_raw = run_text_round(model, processor, round1_prompt, args.max_new_tokens)
        round1_summary = {
            "round_index": 1,
            "round_type": "global_summary",
            "source_classes_count": len(classes),
            "classes_in_prompt": len(classes_payload),
            "summary": round1_payload,
            "accepted_archetype_count": len(accepted_archetypes),
            "raw_response": round1_raw,
        }
        write_json(task_dir / "round_001_summary.json", round1_summary)
        latest_summary = round1_summary
        round_index = 2

    for proposal_round in range(args.proposal_rounds):
        proposal_prompt = build_proposal_prompt(
            classes_payload=classes_payload,
            latest_summary=latest_summary,
            accepted_archetypes=accepted_archetypes,
            new_archetypes_per_round=args.new_archetypes_per_round,
            round_index=round_index,
        )
        proposal_payload, proposal_raw = run_text_round(model, processor, proposal_prompt, args.max_new_tokens)
        proposed_archetypes = proposal_payload.get("new_archetypes", [])
        if not isinstance(proposed_archetypes, list):
            proposed_archetypes = []

        accepted_this_round: list[dict[str, Any]] = []
        conflict_rows: list[dict[str, Any]] = []
        for candidate in proposed_archetypes:
            if not isinstance(candidate, dict):
                continue
            result = conflict_check(candidate, accepted_archetypes)
            conflict_rows.append({
                "round_index": round_index,
                "candidate": candidate,
                "check": result,
            })
            if result["accepted"]:
                accepted_archetypes.append(candidate)
                accepted_this_round.append(candidate)

        if conflict_rows:
            append_jsonl(conflict_path, conflict_rows)
        write_json(candidate_path, {"archetypes": accepted_archetypes})

        latest_summary = {
            "round_index": round_index,
            "round_type": "proposal",
            "accepted_new_archetypes": accepted_this_round,
            "accepted_archetype_count": len(accepted_archetypes),
            "remaining_regions_to_cover": proposal_payload.get("remaining_regions_to_cover", []),
            "notes": proposal_payload.get("notes", []),
            "raw_response": proposal_raw,
        }
        write_json(task_dir / f"round_{round_index:03d}_summary.json", latest_summary)
        round_index += 1

    review_prompt = build_review_prompt(classes_payload, accepted_archetypes)
    review_payload, review_raw = run_text_round(model, processor, review_prompt, args.max_new_tokens)
    write_json(review_path, {
        "review": review_payload,
        "raw_response": review_raw,
        "accepted_archetype_count": len(accepted_archetypes),
    })

    print(f"[OK] Taxonomy task directory: {task_dir}")
    print(f"[OK] Candidate file: {candidate_path}")
    print(f"[OK] Latest summary: {task_dir / f'round_{round_index - 1:03d}_summary.json'}")
    print(f"[OK] Review file: {review_path}")


if __name__ == "__main__":
    main()
