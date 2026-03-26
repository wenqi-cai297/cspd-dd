"""Generate an archetype taxonomy candidate using a multi-round local Qwen workflow.

This script is designed to reduce single-step pressure on the VLM:
1. Round 1 reads the full classes.json and writes a global summary.
2. Later rounds propose a few new archetypes at a time.
3. Each round writes a summary file into a task-specific timestamp directory.
4. Each new archetype is checked against existing ones for obvious conflicts before acceptance.
5. Coverage is tracked programmatically so later rounds are pushed toward uncovered semantic regions.
6. Repair rounds are triggered when the model keeps proposing already-covered regions.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from cspd_stage1.io_utils import append_jsonl, write_json, write_jsonl
from cspd_stage1.vlm.json_utils import parse_json_object

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
SYSTEM_PROMPT = (
    "You are designing a semantic archetype taxonomy for image dataset classes. "
    "Return JSON only. Do not include markdown or explanations outside JSON."
)

TARGET_DEFINITIONS = {
    "animal": "living creatures including mammals, birds, reptiles, amphibians, fish, and invertebrates",
    "plant": "plant life such as trees, flowers, shrubs, fungi-like natural plant categories when visually plant-like",
    "food": "edible items and prepared foods intended primarily for consumption",
    "vehicle": "mobile transport objects such as cars, trucks, buses, boats, trains, and aircraft",
    "clothing": "wearable garments, shoes, and fashion accessories closely tied to dressing the body",
    "furniture": "large household or office furnishing objects such as chairs, tables, beds, sofas, cabinets",
    "tool": "handheld manual implements used to act on, cut, strike, grip, or modify other objects",
    "device_or_appliance": "powered, mechanical, or electronic functional devices and appliances; not simple handheld manual tools",
    "container": "objects primarily designed to hold, store, or carry items, liquids, or materials",
    "instrument": "musical instruments only; not general tools or electronic appliances",
    "structure_or_building": "large built structures, buildings, architectural constructions, or infrastructure objects",
    "sports_or_toy": "sports equipment, game items, hobby objects, and toys",
    "weapon": "objects primarily designed for attack, defense, or combat use",
    "household_object": "small everyday non-furniture household objects that are not better covered by tool, container, or appliance",
    "decorative_or_symbolic_object": "ornamental, symbolic, religious, or display-oriented artifacts",
}

COVERAGE_TARGETS = list(TARGET_DEFINITIONS.keys())
SEMANTIC_OVERLAP_RULES = {
    "tool": {
        "forbidden_terms": ["appliance", "electronic", "powered device"],
        "forbidden_examples": ["microwave", "washer", "computer"],
    },
    "device_or_appliance": {
        "forbidden_terms": ["manual tool", "hand tool"],
        "forbidden_examples": ["hammer", "wrench", "screwdriver", "pliers"],
    },
    "instrument": {
        "required_terms": ["musical"],
        "forbidden_terms": ["general device", "tool"],
    },
    "container": {
        "required_terms": ["hold", "store", "carry"],
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an archetype taxonomy candidate from classes.json")
    parser.add_argument("--input", required=True, help="Path to classes.json")
    parser.add_argument("--task-dir", default=None, help="Optional task output directory")
    parser.add_argument("--model-name", default=MODEL_NAME, help="Local Qwen model name")
    parser.add_argument("--torch-dtype", default="float16", help="Torch dtype for local model loading")
    parser.add_argument("--device-map", default="auto", help="Transformers device_map")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Generation length cap")
    parser.add_argument("--max-classes-in-prompt", type=int, default=1000, help="Maximum classes included in prompt")
    parser.add_argument("--new-archetypes-per-round", type=int, default=3, help="Requested new archetypes per proposal round")
    parser.add_argument("--proposal-rounds", type=int, default=5, help="Proposal rounds after the initial summary")
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


def normalize_archetype_name(name: str) -> str:
    return "_".join(name.strip().lower().split())


def create_task_dir(task_dir_arg: str | None) -> Path:
    if task_dir_arg:
        task_dir = Path(task_dir_arg)
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        task_dir = Path("runs") / "taxonomy_tasks" / f"{timestamp}_taxonomy"
    task_dir.mkdir(parents=True, exist_ok=True)
    return task_dir


def find_latest_summary(task_dir: Path) -> dict[str, Any] | None:
    summary_files = sorted(task_dir.glob("round_*_summary.json"))
    if not summary_files:
        return None
    return json.loads(summary_files[-1].read_text(encoding="utf-8-sig"))


def compute_coverage_state(accepted_archetypes: list[dict[str, Any]]) -> dict[str, list[str]]:
    accepted_names = {normalize_archetype_name(str(item.get("name", ""))) for item in accepted_archetypes}
    covered = [target for target in COVERAGE_TARGETS if target in accepted_names]
    uncovered = [target for target in COVERAGE_TARGETS if target not in accepted_names]
    extra = sorted(name for name in accepted_names if name and name not in COVERAGE_TARGETS)
    return {"covered_targets": covered, "uncovered_targets": uncovered, "accepted_extra_names": extra}


def build_round1_prompt(classes_payload: list[dict[str, str]]) -> str:
    template = {
        "major_regions": ["animals", "vehicles", "foods", "tools_and_devices", "furniture_and_household_objects"],
        "taxonomy_design_principles": ["keep archetypes at a similar abstraction level", "cover both animal and non-animal regions"],
        "suggested_archetype_count_range": {"min": 8, "max": 20},
        "coverage_gaps_to_watch": ["plants", "clothing", "containers"],
        "notes": ["global first-pass summary only; do not finalize taxonomy yet"],
    }
    return (
        "Round 1 goal: build a global understanding of the full class space before proposing final archetypes.\n"
        "Do NOT output a final taxonomy yet. Summarize major semantic regions, desired abstraction level, and likely gaps.\n"
        "Return JSON only in this exact top-level structure:\n"
        f"{json.dumps(template, ensure_ascii=False, indent=2)}\n"
        "Dataset classes:\n"
        f"{json.dumps(classes_payload, ensure_ascii=False, indent=2)}\n"
    )


def build_target_definition_block(targets: list[str]) -> list[dict[str, str]]:
    return [{"target": t, "definition": TARGET_DEFINITIONS[t]} for t in targets]


def build_proposal_prompt(classes_payload: list[dict[str, str]], latest_summary: dict[str, Any], accepted_archetypes: list[dict[str, Any]], coverage_state: dict[str, list[str]], new_archetypes_per_round: int, round_index: int) -> str:
    prioritized_targets = coverage_state["uncovered_targets"][: max(new_archetypes_per_round + 2, 5)]
    template = {
        "round_index": round_index,
        "new_archetypes": [
            {
                "name": "vehicle",
                "definition": TARGET_DEFINITIONS["vehicle"],
                "inclusion_guidelines": ["cars, trucks, buses, boats, aircraft"],
                "example_classes": ["sports car", "school bus", "airliner"],
                "conflict_risk_with_existing": ["device_or_appliance"],
                "why_needed": "covers a major uncovered semantic region",
                "target_coverage_region": "vehicle",
            }
        ],
        "remaining_regions_to_cover": ["food", "plant", "clothing"],
        "notes": ["prioritize uncovered regions and avoid repeating accepted archetypes"],
    }
    return (
        f"You are continuing taxonomy design for round {round_index}.\n"
        "You must propose archetypes ONLY for currently uncovered semantic regions unless a split is absolutely necessary.\n"
        "Hard rules:\n"
        "- Keep archetypes at the same abstraction level\n"
        "- Do not repeat covered targets\n"
        "- Do not propose synonyms of covered targets\n"
        "- Do not mix tool with device_or_appliance\n"
        "- instrument means musical instrument only\n"
        "- device_or_appliance must exclude simple manual tools\n"
        f"- Propose at most {new_archetypes_per_round} archetypes this round\n"
        "Program-maintained covered targets:\n"
        f"{json.dumps(coverage_state['covered_targets'], ensure_ascii=False, indent=2)}\n"
        "Program-maintained uncovered targets:\n"
        f"{json.dumps(coverage_state['uncovered_targets'], ensure_ascii=False, indent=2)}\n"
        "Prioritized targets for this round:\n"
        f"{json.dumps(build_target_definition_block(prioritized_targets), ensure_ascii=False, indent=2)}\n"
        "Current accepted archetypes:\n"
        f"{json.dumps(accepted_archetypes, ensure_ascii=False, indent=2)}\n"
        "Latest summary:\n"
        f"{json.dumps(latest_summary, ensure_ascii=False, indent=2)}\n"
        "Dataset classes:\n"
        f"{json.dumps(classes_payload, ensure_ascii=False, indent=2)}\n"
        "Return JSON only in this exact top-level structure:\n"
        f"{json.dumps(template, ensure_ascii=False, indent=2)}\n"
    )


def build_repair_prompt(classes_payload: list[dict[str, str]], accepted_archetypes: list[dict[str, Any]], coverage_state: dict[str, list[str]], round_index: int) -> str:
    prioritized_targets = coverage_state["uncovered_targets"][:5]
    template = {
        "round_index": round_index,
        "repair_new_archetypes": [
            {
                "name": "plant",
                "definition": TARGET_DEFINITIONS["plant"],
                "inclusion_guidelines": ["trees, flowers, leaves, mushrooms if treated as plant-like visual classes"],
                "example_classes": ["daisy", "corn", "acorn"],
                "target_coverage_region": "plant",
                "why_needed": "repair round focuses on uncovered targets only",
            }
        ],
        "notes": ["repair round: propose only uncovered targets"],
    }
    return (
        f"This is a repair round for round {round_index}.\n"
        "Recent proposal rounds repeated covered targets or failed to expand coverage.\n"
        "You are now restricted to proposing archetypes ONLY from the uncovered targets below.\n"
        "If you propose a covered target, the proposal will be rejected.\n"
        "Uncovered targets with definitions:\n"
        f"{json.dumps(build_target_definition_block(prioritized_targets), ensure_ascii=False, indent=2)}\n"
        "Accepted archetypes:\n"
        f"{json.dumps(accepted_archetypes, ensure_ascii=False, indent=2)}\n"
        "Dataset classes:\n"
        f"{json.dumps(classes_payload, ensure_ascii=False, indent=2)}\n"
        "Return JSON only in this exact top-level structure:\n"
        f"{json.dumps(template, ensure_ascii=False, indent=2)}\n"
    )


def build_review_prompt(classes_payload: list[dict[str, str]], accepted_archetypes: list[dict[str, Any]], coverage_state: dict[str, list[str]]) -> str:
    template = {
        "coverage_assessment": {
            "animal_regions_covered": True,
            "non_animal_regions_covered": True,
            "major_missing_regions": ["food", "plant"],
        },
        "overlap_risks": ["possible overlap between tool and device_or_appliance"],
        "final_notes": ["taxonomy is usable as a candidate for manual review after checking missing regions"],
    }
    return (
        "Perform a strict final review of the taxonomy candidate.\n"
        "You must respect the program-maintained uncovered targets below.\n"
        "If a target is still uncovered, list it under major_missing_regions unless you explicitly argue it should merge elsewhere.\n"
        "Program-maintained covered targets:\n"
        f"{json.dumps(coverage_state['covered_targets'], ensure_ascii=False, indent=2)}\n"
        "Program-maintained uncovered targets:\n"
        f"{json.dumps(coverage_state['uncovered_targets'], ensure_ascii=False, indent=2)}\n"
        "Accepted archetypes:\n"
        f"{json.dumps(accepted_archetypes, ensure_ascii=False, indent=2)}\n"
        "Dataset classes:\n"
        f"{json.dumps(classes_payload, ensure_ascii=False, indent=2)}\n"
        "Return JSON only in this exact top-level structure:\n"
        f"{json.dumps(template, ensure_ascii=False, indent=2)}\n"
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
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    payload = parse_json_object(output_text)
    return payload, output_text


def coverage_gate(candidate: dict[str, Any], coverage_state: dict[str, list[str]]) -> dict[str, Any]:
    target_region = normalize_archetype_name(str(candidate.get("target_coverage_region", "")))
    if not target_region:
        return {"accepted": False, "reason": "missing_target_coverage_region"}
    if target_region in coverage_state["covered_targets"]:
        return {"accepted": False, "reason": f"target_already_covered:{target_region}"}
    if target_region not in coverage_state["uncovered_targets"]:
        return {"accepted": False, "reason": f"target_not_in_uncovered_set:{target_region}"}
    return {"accepted": True, "reason": "ok"}


def semantic_overlap_guard(candidate: dict[str, Any], accepted: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_name = normalize_archetype_name(str(candidate.get("name", "")))
    definition = str(candidate.get("definition", "")).lower()
    examples = " ".join(str(x).lower() for x in candidate.get("example_classes", []))
    text = f"{definition} {examples}"
    conflicts: list[str] = []
    rule = SEMANTIC_OVERLAP_RULES.get(candidate_name)
    if rule:
        for term in rule.get("forbidden_terms", []):
            if term in text:
                conflicts.append(f"forbidden_term:{term}")
        for term in rule.get("forbidden_examples", []):
            if term in text:
                conflicts.append(f"forbidden_example:{term}")
        required_terms = rule.get("required_terms", [])
        if required_terms and not any(term in text for term in required_terms):
            conflicts.append(f"missing_required_terms:{'|'.join(required_terms)}")
    for existing in accepted:
        existing_name = normalize_archetype_name(str(existing.get("name", "")))
        if candidate_name == "device_or_appliance" and existing_name == "tool":
            if any(word in text for word in ["hammer", "wrench", "screwdriver", "pliers", "manual tool", "hand tool"]):
                conflicts.append("device_or_appliance_overlaps_tool")
        if candidate_name == "tool" and existing_name == "device_or_appliance":
            if any(word in text for word in ["appliance", "electronic", "powered device"]):
                conflicts.append("tool_overlaps_device_or_appliance")
    return {"accepted": len(conflicts) == 0, "conflicts": conflicts}


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
    return {"candidate_name": candidate_name, "accepted": len(conflicts) == 0, "conflicts": conflicts}


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
        round1_payload, round1_raw = run_text_round(model, processor, build_round1_prompt(classes_payload), args.max_new_tokens)
        latest_summary = {
            "round_index": 1,
            "round_type": "global_summary",
            "source_classes_count": len(classes),
            "classes_in_prompt": len(classes_payload),
            "summary": round1_payload,
            "accepted_archetype_count": len(accepted_archetypes),
            "coverage_state": compute_coverage_state(accepted_archetypes),
            "raw_response": round1_raw,
        }
        write_json(task_dir / "round_001_summary.json", latest_summary)
        round_index = 2

    consecutive_zero_accept_rounds = 0
    for _ in range(args.proposal_rounds):
        coverage_state = compute_coverage_state(accepted_archetypes)
        use_repair = consecutive_zero_accept_rounds >= 1 and len(coverage_state["uncovered_targets"]) > 0
        if use_repair:
            prompt = build_repair_prompt(classes_payload, accepted_archetypes, coverage_state, round_index)
            proposal_payload, proposal_raw = run_text_round(model, processor, prompt, args.max_new_tokens)
            proposed_archetypes = proposal_payload.get("repair_new_archetypes", [])
            round_mode = "repair"
        else:
            prompt = build_proposal_prompt(classes_payload, latest_summary, accepted_archetypes, coverage_state, args.new_archetypes_per_round, round_index)
            proposal_payload, proposal_raw = run_text_round(model, processor, prompt, args.max_new_tokens)
            proposed_archetypes = proposal_payload.get("new_archetypes", [])
            round_mode = "proposal"
        if not isinstance(proposed_archetypes, list):
            proposed_archetypes = []

        accepted_this_round: list[dict[str, Any]] = []
        conflict_rows: list[dict[str, Any]] = []
        for candidate in proposed_archetypes:
            if not isinstance(candidate, dict):
                continue
            gate = coverage_gate(candidate, coverage_state)
            name_check = conflict_check(candidate, accepted_archetypes)
            overlap_check = semantic_overlap_guard(candidate, accepted_archetypes)
            accepted_flag = gate["accepted"] and name_check["accepted"] and overlap_check["accepted"]
            row = {
                "round_index": round_index,
                "round_mode": round_mode,
                "candidate": candidate,
                "coverage_gate": gate,
                "name_conflict_check": name_check,
                "semantic_overlap_check": overlap_check,
                "accepted": accepted_flag,
            }
            conflict_rows.append(row)
            if accepted_flag:
                accepted_archetypes.append(candidate)
                accepted_this_round.append(candidate)

        if conflict_rows:
            append_jsonl(conflict_path, conflict_rows)
        write_json(candidate_path, {"archetypes": accepted_archetypes})
        if accepted_this_round:
            consecutive_zero_accept_rounds = 0
        else:
            consecutive_zero_accept_rounds += 1

        latest_summary = {
            "round_index": round_index,
            "round_type": round_mode,
            "accepted_new_archetypes": accepted_this_round,
            "accepted_archetype_count": len(accepted_archetypes),
            "coverage_state": compute_coverage_state(accepted_archetypes),
            "model_reported_remaining_regions": proposal_payload.get("remaining_regions_to_cover", []),
            "notes": proposal_payload.get("notes", []),
            "consecutive_zero_accept_rounds": consecutive_zero_accept_rounds,
            "raw_response": proposal_raw,
        }
        write_json(task_dir / f"round_{round_index:03d}_summary.json", latest_summary)
        round_index += 1

    coverage_state = compute_coverage_state(accepted_archetypes)
    review_payload, review_raw = run_text_round(model, processor, build_review_prompt(classes_payload, accepted_archetypes, coverage_state), args.max_new_tokens)
    write_json(review_path, {
        "review": review_payload,
        "raw_response": review_raw,
        "accepted_archetype_count": len(accepted_archetypes),
        "coverage_state": coverage_state,
    })

    print(f"[OK] Taxonomy task directory: {task_dir}")
    print(f"[OK] Candidate file: {candidate_path}")
    print(f"[OK] Latest summary: {task_dir / f'round_{round_index - 1:03d}_summary.json'}")
    print(f"[OK] Review file: {review_path}")


if __name__ == "__main__":
    main()
