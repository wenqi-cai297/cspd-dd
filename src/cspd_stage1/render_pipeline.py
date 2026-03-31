from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cspd_stage1.io_utils import append_jsonl, read_jsonl, write_json, write_jsonl
from cspd_stage1.render_utils import (
    class_name_to_anchor,
    clean_pre_anchor_value,
    cleanup_caption,
    format_post_slot,
    is_unknown_like,
    should_drop_slot,
    stringify_slot_value,
    with_article,
)
from cspd_stage1.templates import TemplateSpec, get_template_spec


@dataclass(slots=True)
class Stage1RenderConfig:
    input_path: str
    output_dir: str
    resume: bool = True
    fail_on_missing_anchor: bool = False
    fallback_to_raw: bool = False
    fallback_anchor_token: str | None = None
    renderer_version: str = "v1"
    flush_every: int = 100


def run_stage1_render(config: Stage1RenderConfig) -> dict[str, Any]:
    rows = read_jsonl(config.input_path)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    success_path = output_dir / "records.jsonl"
    failed_path = output_dir / "failures.jsonl"
    summary_path = output_dir / "render_summary.json"

    if config.resume:
        success_rows = read_jsonl(success_path) if success_path.exists() else []
        failed_rows = read_jsonl(failed_path) if failed_path.exists() else []
    else:
        success_rows = []
        failed_rows = []
        write_jsonl(success_path, [])
        write_jsonl(failed_path, [])

    processed_ids = {
        str(row.get("record_id"))
        for row in success_rows
        if isinstance(row, dict) and row.get("record_id") is not None
    }

    pending_success: list[dict[str, Any]] = []
    pending_failed: list[dict[str, Any]] = []

    rows_to_process = [row for row in rows if str(row.get("record_id") or row.get("sample_id")) not in processed_ids]

    for index, row in enumerate(rows_to_process, start=1):
        result = render_row(row, config)
        if result["render_status"] == "success":
            success_rows.append(result)
            pending_success.append(result)
        else:
            failed_rows.append(result)
            pending_failed.append(result)

        if index == len(rows_to_process) or index % max(config.flush_every, 1) == 0:
            if pending_success:
                append_jsonl(success_path, pending_success)
                pending_success.clear()
            if pending_failed:
                append_jsonl(failed_path, pending_failed)
                pending_failed.clear()
            write_json(summary_path, build_summary(rows, success_rows, failed_rows, config, len(processed_ids)))

    summary = build_summary(rows, success_rows, failed_rows, config, len(processed_ids))
    write_json(summary_path, summary)
    return summary


def render_row(row: dict[str, Any], config: Stage1RenderConfig) -> dict[str, Any]:
    record_id = row.get("record_id") or row.get("sample_id") or row.get("image_path")
    archetype = row.get("archetype")
    class_name = row.get("class_name")

    effective_normalized_attributes = row.get("effective_normalized_attributes")
    base = {
        "record_id": record_id,
        "sample_id": row.get("sample_id"),
        "class_name": class_name,
        "archetype": archetype,
        "used_normalized_attributes": True,
        "used_effective_normalized_attributes": isinstance(effective_normalized_attributes, dict),
        "normalization_review_required": bool(row.get("normalization_review_required", False)),
        "vlm_review": row.get("vlm_review"),
    }

    if not record_id:
        return {**base, "render_status": "failed", "error_message": "missing record_id", "render_warnings": []}
    if not archetype:
        return {**base, "render_status": "failed", "error_message": "missing archetype", "render_warnings": []}

    try:
        template = get_template_spec(str(archetype))
    except KeyError as exc:
        return {**base, "render_status": "failed", "error_message": str(exc), "render_warnings": []}

    normalized_attributes = row.get("effective_normalized_attributes")
    if not isinstance(normalized_attributes, dict):
        normalized_attributes = row.get("normalized_attributes")
    raw_attributes = row.get("attributes")
    if not isinstance(normalized_attributes, dict):
        if config.fallback_to_raw and isinstance(raw_attributes, dict):
            normalized_attributes = raw_attributes
            base["used_normalized_attributes"] = False
            base["used_effective_normalized_attributes"] = False
        else:
            return {**base, "render_status": "failed", "error_message": "missing normalized_attributes", "render_warnings": []}

    return _render_with_template(base, row, normalized_attributes, template, config)


def _render_with_template(
    base: dict[str, Any],
    raw_row: dict[str, Any],
    normalized_attributes: dict[str, Any],
    template: TemplateSpec,
    config: Stage1RenderConfig,
) -> dict[str, Any]:
    warnings: list[str] = []
    verbalized_slots: list[str] = []
    dropped_slots: list[str] = []
    drop_reasons: dict[str, str] = {}

    anchor_value = _select_value(normalized_attributes.get(template.anchor_slot))
    if anchor_value is None:
        class_anchor = class_name_to_anchor(base.get("class_name"))
        if class_anchor:
            anchor_value = class_anchor
            warnings.append("class_name_anchor_used")
        elif config.fallback_anchor_token:
            anchor_value = config.fallback_anchor_token
            warnings.append("fallback_anchor_used")
        elif config.fail_on_missing_anchor:
            return {
                **base,
                "render_status": "failed",
                "error_message": f"missing anchor slot: {template.anchor_slot}",
                "render_warnings": warnings,
            }
        else:
            anchor_value = template.fallback_anchor or None
            if anchor_value:
                warnings.append("template_fallback_anchor_used")
            else:
                return {
                    **base,
                    "render_status": "failed",
                    "error_message": f"missing anchor slot: {template.anchor_slot}",
                    "render_warnings": warnings,
                }

    pre_parts: list[str] = []
    for slot in template.pre_anchor_slots:
        value = _select_value(normalized_attributes.get(slot))
        if value is None:
            dropped_slots.append(slot)
            drop_reasons[slot] = "unknown_or_empty"
            continue
        review_required = _slot_is_review_required(raw_row, slot)
        should_drop, reason = should_drop_slot(template.archetype, slot, value, review_required=review_required)
        if should_drop:
            dropped_slots.append(slot)
            drop_reasons[slot] = reason or "dropped_by_rule"
            continue
        cleaned_value = clean_pre_anchor_value(slot, value)
        if not cleaned_value:
            dropped_slots.append(slot)
            drop_reasons[slot] = "empty_after_cleaning"
            continue
        pre_parts.append(cleaned_value)
        verbalized_slots.append(slot)

    noun_phrase = " ".join(part for part in [*pre_parts, anchor_value] if part)
    caption_parts = [with_article(noun_phrase)]
    verbalized_slots.append(template.anchor_slot)

    for slot in template.post_anchor_slots:
        value = _select_value(normalized_attributes.get(slot))
        if value is None:
            dropped_slots.append(slot)
            drop_reasons[slot] = "unknown_or_empty"
            continue
        review_required = _slot_is_review_required(raw_row, slot)
        should_drop, reason = should_drop_slot(template.archetype, slot, value, review_required=review_required)
        if should_drop:
            dropped_slots.append(slot)
            drop_reasons[slot] = reason or "dropped_by_rule"
            continue
        phrase = format_post_slot(slot, value)
        if not phrase:
            dropped_slots.append(slot)
            drop_reasons[slot] = "empty_after_formatting"
            continue
        caption_parts.append(phrase)
        verbalized_slots.append(slot)

    caption = cleanup_caption(" ".join(part for part in caption_parts if part))
    if not caption:
        return {
            **base,
            "render_status": "failed",
            "error_message": "empty caption after rendering",
            "render_warnings": warnings,
        }

    if base.get("normalization_review_required"):
        warnings.append("normalization_review_required")

    return {
        **base,
        "canonical_caption": caption,
        "renderer": {
            "renderer_version": config.renderer_version,
            "template_family": template.archetype,
            "template_id": template.template_id,
        },
        "anchor_slot": template.anchor_slot,
        "verbalized_slots": verbalized_slots,
        "dropped_slots": dropped_slots,
        "drop_reasons": drop_reasons,
        "render_warnings": sorted(set(warnings)),
        "render_status": "success",
    }


def _select_value(value: Any) -> str | None:
    text = stringify_slot_value(value)
    if text is None or is_unknown_like(text):
        return None
    return text


def _slot_is_review_required(row: dict[str, Any], slot: str) -> bool:
    meta = row.get("attribute_normalization")
    if not isinstance(meta, dict):
        return False
    slot_meta = meta.get(slot)
    if not isinstance(slot_meta, dict):
        return False
    status = slot_meta.get("status")
    review_reasons = slot_meta.get("review_reasons")
    return status == "review_required" or bool(review_reasons)


def build_summary(
    input_rows: list[dict[str, Any]],
    success_rows: list[dict[str, Any]],
    failed_rows: list[dict[str, Any]],
    config: Stage1RenderConfig,
    resumed_success_count: int,
) -> dict[str, Any]:
    counts_by_archetype: dict[str, int] = {}
    counts_by_template_id: dict[str, int] = {}
    dropped_slot_counts: dict[str, int] = {}

    for row in success_rows:
        archetype = str(row.get("archetype"))
        counts_by_archetype[archetype] = counts_by_archetype.get(archetype, 0) + 1
        template_id = str(((row.get("renderer") or {}).get("template_id")))
        counts_by_template_id[template_id] = counts_by_template_id.get(template_id, 0) + 1
        for slot in row.get("dropped_slots", []):
            dropped_slot_counts[str(slot)] = dropped_slot_counts.get(str(slot), 0) + 1

    total_input = len(input_rows)
    success_count = len(success_rows)
    failure_count = len(failed_rows)
    avg_verbalized = (
        sum(len(row.get("verbalized_slots", [])) for row in success_rows) / success_count
        if success_count else 0.0
    )
    review_count = sum(1 for row in success_rows if row.get("normalization_review_required"))

    return {
        "input_path": str(Path(config.input_path).resolve()),
        "output_dir": str(Path(config.output_dir).resolve()),
        "renderer_version": config.renderer_version,
        "total_input_rows": total_input,
        "num_rows_skipped_via_resume": resumed_success_count,
        "success_count": success_count,
        "failure_count": failure_count,
        "success_rate": (success_count / total_input) if total_input else 0.0,
        "counts_by_archetype": counts_by_archetype,
        "counts_by_template_id": counts_by_template_id,
        "rows_with_normalization_review_required": review_count,
        "average_verbalized_slot_count": avg_verbalized,
        "dropped_slot_counts": dropped_slot_counts,
        "artifacts": {
            "records": str((Path(config.output_dir) / "records.jsonl").resolve()),
            "failures": str((Path(config.output_dir) / "failures.jsonl").resolve()),
            "summary": str((Path(config.output_dir) / "render_summary.json").resolve()),
        },
    }


def config_from_args(args: Any) -> Stage1RenderConfig:
    return Stage1RenderConfig(
        input_path=args.input,
        output_dir=args.output_dir,
        resume=not args.no_resume,
        fail_on_missing_anchor=args.fail_on_missing_anchor,
        fallback_to_raw=args.fallback_to_raw,
        fallback_anchor_token=args.fallback_anchor_token,
        renderer_version=args.renderer_version,
        flush_every=args.flush_every,
    )
