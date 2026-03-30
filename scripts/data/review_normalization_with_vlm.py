from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from cspd_stage1.schema import SampleRecord
from cspd_stage1.vlm.base import BaseVLMClient, VLMOutputParseError
from cspd_stage1.vlm.factory import create_vlm_client

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_ALLOWED_ACTIONS = ("keep_normalized", "replace_normalized", "set_unknown", "defer")
REVIEW_SYSTEM_PROMPT = (
    "You are a strict normalization review assistant for image attributes. "
    "Return JSON only. Never change the provided archetype or slot. "
    "Be conservative: if the image is unclear, choose defer or keep_normalized instead of inventing detail."
)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            yield line_number, json.loads(stripped)


def build_review_prompt(row: dict[str, Any], slot: str, slot_meta: dict[str, Any]) -> str:
    allowed_actions = list(DEFAULT_ALLOWED_ACTIONS)
    payload_hint = {
        "record_id": str(row.get("record_id") or ""),
        "archetype": str(row.get("archetype") or ""),
        "slot": slot,
        "action": "one of: " + " | ".join(allowed_actions),
        "reviewed_value": "short string; use current normalized value for keep_normalized, another slot-compatible phrase for replace_normalized, or 'unknown' for set_unknown/defer",
        "confidence": "high | medium | low",
        "needs_manual_followup": False,
        "reason": "one short sentence",
    }
    raw_attributes = row.get("attributes") or {}
    if not isinstance(raw_attributes, dict):
        raw_attributes = {}
    normalized_attributes = row.get("normalized_attributes") or {}
    if not isinstance(normalized_attributes, dict):
        normalized_attributes = {}

    return (
        "Review exactly one ambiguous normalized attribute.\n"
        f"record_id: {row.get('record_id')}\n"
        f"class_name: {row.get('class_name')}\n"
        f"class_name_raw: {row.get('class_name_raw')}\n"
        f"archetype: {row.get('archetype')}\n"
        f"slot: {slot}\n"
        f"slot_schema: {json.dumps(row.get('slot_schema') or [], ensure_ascii=False)}\n"
        f"review_reasons: {json.dumps(slot_meta.get('review_reasons') or [], ensure_ascii=False)}\n"
        f"raw_slot_value: {json.dumps(raw_attributes.get(slot), ensure_ascii=False)}\n"
        f"normalized_slot_value: {json.dumps(slot_meta.get('normalized_value', normalized_attributes.get(slot, 'unknown')), ensure_ascii=False)}\n"
        "Task:\n"
        "- Look at the image and decide whether to keep the deterministic normalized value, replace it with a better short value for this same slot, map it to unknown, or defer.\n"
        "- Do NOT change archetype.\n"
        "- Do NOT answer for any other slot.\n"
        "- Keep output short, slot-compatible, and render-friendly.\n"
        "- If evidence is weak or ambiguous, prefer defer.\n"
        f"Return JSON with this exact schema:\n{json.dumps(payload_hint, ensure_ascii=False, indent=2)}"
    )


def build_sample(row: dict[str, Any]) -> SampleRecord:
    slot_schema = row.get("slot_schema") or []
    if not isinstance(slot_schema, list):
        slot_schema = []
    return SampleRecord(
        image_path=str(row.get("image_path") or ""),
        class_id=int(row.get("class_id") or 0),
        class_name=str(row.get("class_name") or ""),
        class_name_raw=str(row.get("class_name_raw") or ""),
        archetype=str(row.get("archetype") or ""),
        slot_schema=[str(item) for item in slot_schema],
        sample_id=str(row.get("sample_id") or row.get("record_id") or "") or None,
    )


def run_mock_review(row: dict[str, Any], slot: str, slot_meta: dict[str, Any]) -> dict[str, Any]:
    normalized_value = str(slot_meta.get("normalized_value") or "unknown")
    review_reasons = [str(item) for item in slot_meta.get("review_reasons") or []]
    action = "defer" if review_reasons else "keep_normalized"
    reviewed_value = normalized_value if action != "set_unknown" else "unknown"
    return {
        "record_id": str(row.get("record_id") or ""),
        "archetype": str(row.get("archetype") or ""),
        "slot": slot,
        "action": action,
        "reviewed_value": reviewed_value,
        "confidence": "low" if action == "defer" else "medium",
        "needs_manual_followup": action == "defer",
        "reason": "mock backend emits structured placeholder review decisions only",
    }


def parse_review_payload(payload: dict[str, Any], row: dict[str, Any], slot: str, slot_meta: dict[str, Any]) -> dict[str, Any]:
    action = str(payload.get("action") or "defer").strip()
    if action not in DEFAULT_ALLOWED_ACTIONS:
        action = "defer"

    reviewed_value = str(payload.get("reviewed_value") or "").strip()
    current_value = str(slot_meta.get("normalized_value") or "unknown")
    if not reviewed_value:
        reviewed_value = current_value if action in {"keep_normalized", "defer"} else "unknown"

    if action == "keep_normalized":
        reviewed_value = current_value
    elif action == "set_unknown":
        reviewed_value = "unknown"

    confidence = str(payload.get("confidence") or "low").strip().lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"

    archetype = str(payload.get("archetype") or row.get("archetype") or "")
    if archetype != str(row.get("archetype") or ""):
        action = "defer"
        reviewed_value = current_value
        confidence = "low"

    slot_value = str(payload.get("slot") or slot)
    if slot_value != slot:
        action = "defer"
        reviewed_value = current_value
        confidence = "low"

    return {
        "record_id": str(row.get("record_id") or ""),
        "archetype": str(row.get("archetype") or ""),
        "slot": slot,
        "action": action,
        "reviewed_value": reviewed_value,
        "confidence": confidence,
        "needs_manual_followup": bool(payload.get("needs_manual_followup", action == "defer")),
        "reason": str(payload.get("reason") or "").strip() or "no reason provided",
    }


def select_review_items(row: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    meta = row.get("attribute_normalization") or {}
    if not isinstance(meta, dict):
        return []
    items: list[tuple[str, dict[str, Any]]] = []
    for slot, slot_meta in meta.items():
        if not isinstance(slot_meta, dict):
            continue
        status = str(slot_meta.get("status") or "")
        review_reasons = slot_meta.get("review_reasons") or []
        if status == "review_required" or bool(review_reasons):
            items.append((str(slot), slot_meta))
    return items


def review_file(
    input_path: Path,
    output_dir: Path,
    backend: str,
    model_name: str,
    torch_dtype: str,
    device_map: str,
    use_fast_processor: bool,
    max_new_tokens: int,
    limit: int | None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    reviews_path = output_dir / "normalization_review_vlm.jsonl"
    summary_path = output_dir / "normalization_review_vlm_summary.json"

    client: BaseVLMClient | None = None
    if backend != "mock":
        client = create_vlm_client(
            backend,
            model_name=model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            use_fast_processor=use_fast_processor,
            max_new_tokens=max_new_tokens,
        )

    counts = Counter()
    reason_counts = Counter()
    action_counts = Counter()

    with reviews_path.open("w", encoding="utf-8") as handle:
        for _, row in iter_jsonl(input_path):
            review_items = select_review_items(row)
            if not review_items:
                continue
            for slot, slot_meta in review_items:
                if limit is not None and counts["items_total"] >= limit:
                    break
                counts["items_total"] += 1
                review_reasons = [str(item) for item in slot_meta.get("review_reasons") or []]
                for reason in review_reasons:
                    reason_counts[reason] += 1

                record = {
                    "record_id": row.get("record_id"),
                    "image_path": row.get("image_path"),
                    "class_name": row.get("class_name"),
                    "class_name_raw": row.get("class_name_raw"),
                    "archetype": row.get("archetype"),
                    "slot": slot,
                    "slot_schema": row.get("slot_schema") or [],
                    "review_reasons": review_reasons,
                    "raw_value": slot_meta.get("raw_value"),
                    "normalized_value": slot_meta.get("normalized_value"),
                    "review_status": "ok",
                }

                try:
                    if backend == "mock":
                        parsed = run_mock_review(row, slot, slot_meta)
                        raw_text = json.dumps(parsed, ensure_ascii=False)
                    else:
                        assert client is not None
                        prompt = build_review_prompt(row, slot, slot_meta)
                        sample = build_sample(row)
                        response = client.extract_attributes(sample, prompt, REVIEW_SYSTEM_PROMPT)
                        parsed = parse_review_payload(response.payload, row, slot, slot_meta)
                        raw_text = response.raw_text
                    counts["items_ok"] += 1
                except (FileNotFoundError, VLMOutputParseError, ValueError, KeyError) as exc:
                    counts["items_failed"] += 1
                    parsed = {
                        "record_id": str(row.get("record_id") or ""),
                        "archetype": str(row.get("archetype") or ""),
                        "slot": slot,
                        "action": "defer",
                        "reviewed_value": str(slot_meta.get("normalized_value") or "unknown"),
                        "confidence": "low",
                        "needs_manual_followup": True,
                        "reason": f"review_failed: {exc}",
                    }
                    raw_text = None
                    record["review_status"] = "failed"
                    record["error_message"] = str(exc)

                action_counts[parsed["action"]] += 1
                record["vlm_review"] = parsed
                record["raw_model_output"] = raw_text
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if limit is not None and counts["items_total"] >= limit:
                break

    summary = {
        "input_path": str(input_path.resolve()),
        "backend": backend,
        "model_name": model_name,
        "num_review_items": counts["items_total"],
        "num_ok": counts["items_ok"],
        "num_failed": counts["items_failed"],
        "review_reason_counts": dict(reason_counts.most_common()),
        "action_counts": dict(action_counts.most_common()),
        "artifacts": {
            "normalization_review_vlm": str(reviews_path.resolve()),
        },
        "contract": {
            "allowed_actions": list(DEFAULT_ALLOWED_ACTIONS),
            "trigger": "Only slots whose normalization metadata has status=review_required or non-empty review_reasons are sent to the VLM review path.",
            "application_policy": "This helper does not overwrite attributes_normalized.jsonl. It emits sidecar review decisions for ambiguous normalization cases only.",
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a constrained VLM review over normalization_review items only.")
    parser.add_argument("--input", required=True, help="Path to attributes_normalized.jsonl")
    parser.add_argument("--output-dir", required=True, help="Directory for VLM review artifacts")
    parser.add_argument("--backend", default="mock", help="VLM backend name (mock or qwen_local)")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Model name for real VLM backends")
    parser.add_argument("--torch-dtype", default="float16", help="Torch dtype for local model loading")
    parser.add_argument("--device-map", default="auto", help="Transformers device_map for local model loading")
    parser.add_argument("--disable-fast-processor", action="store_true", help="Use the slow processor implementation instead of the fast default")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Generation cap for local VLM backends")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of review items to process")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file does not exist: {input_path}")

    summary = review_file(
        input_path=input_path,
        output_dir=Path(args.output_dir),
        backend=args.backend,
        model_name=args.model_name,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        use_fast_processor=not args.disable_fast_processor,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
