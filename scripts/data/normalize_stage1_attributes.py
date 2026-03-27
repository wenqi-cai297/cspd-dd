from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

DEFAULT_RULES_PATH = Path("configs/stage1/normalization/stage1_attribute_normalization_rules.json")
STATUS_CHANGED = {"canonicalized", "class_inferred", "mapped_to_unknown", "review_required"}


class Normalizer:
    def __init__(self, rules: dict[str, Any]) -> None:
        self.rules = rules
        slot_groups = rules.get("slot_alias_groups", {})
        self.type_slots = set(slot_groups.get("type_slots", []))
        self.material_slots = set(slot_groups.get("material_slots", []))
        self.color_slots = set(slot_groups.get("color_slots", []))
        self.state_slots = set(slot_groups.get("state_slots", []))
        self.shape_slots = set(slot_groups.get("shape_slots", []))
        self.part_slots = set(slot_groups.get("part_slots", []))
        self.background_slots = set(slot_groups.get("background_slots", []))
        self.viewpoint_slots = set(slot_groups.get("viewpoint", []))
        self.unknown_like = {self._clean_key(v) for v in rules.get("unknown_like_values", [])}
        self.viewpoint_map = self._clean_mapping(rules.get("viewpoint_map", {}))
        self.state_map = self._clean_mapping(rules.get("state_map", {}))
        self.shape_map = self._clean_mapping(rules.get("shape_map", {}))
        self.part_map = self._clean_mapping(rules.get("part_map", {}))
        self.background_map = self._clean_mapping(rules.get("background_map", {}))
        self.class_slot_maps = {
            class_name: {slot: self._clean_mapping(slot_map) for slot, slot_map in slot_maps.items()}
            for class_name, slot_maps in rules.get("class_slot_maps", {}).items()
        }
        self.review_value_sets = {
            class_name: {slot: {self._clean_key(v) for v in values} for slot, values in slot_maps.items()}
            for class_name, slot_maps in rules.get("review_value_sets", {}).items()
        }
        review_substrings = rules.get("review_substrings", {})
        self.person_markers = tuple(review_substrings.get("part_or_state_person_mentions", []))
        self.narrative_markers = tuple(review_substrings.get("mixed_or_narrative_markers", []))
        self.background_color_markers = tuple(review_substrings.get("slot_contamination_color_background", []))
        refinement_rules = rules.get("v2_refinement", {})
        self.person_to_unknown_slots = set(refinement_rules.get("person_to_unknown_slots", []))
        self.person_unknown_phrases = tuple(self._clean_key(v) for v in refinement_rules.get("person_unknown_phrases", []))
        self.mixed_state_primary_values = {
            self._clean_key(v) for v in refinement_rules.get("mixed_state_primary_values", [])
        }

    @staticmethod
    def _clean_key(value: Any) -> str:
        text = "" if value is None else str(value)
        text = text.strip().casefold()
        text = text.replace("_", " ")
        text = re.sub(r"\s+", " ", text)
        return text

    def _clean_mapping(self, mapping: dict[str, str]) -> dict[str, str]:
        return {self._clean_key(k): v for k, v in mapping.items()}

    def normalize_field(self, class_name_raw: str, slot: str, raw_value: Any) -> dict[str, Any]:
        original = raw_value
        cleaned = self.clean_value(raw_value)
        status = "unchanged"
        normalized = cleaned
        applied_rules: list[str] = []
        review_reasons: list[str] = []

        if self.is_unknown_like(cleaned):
            normalized = "unknown"
            status = "mapped_to_unknown"
            applied_rules.append("global.placeholder_to_unknown")
            return self._result(original, cleaned, normalized, status, applied_rules, review_reasons)

        if cleaned != normalized:
            status = "canonicalized"

        normalized, status, applied_rules = self.normalize_low_value_contamination(slot, normalized, status, applied_rules)
        if status == "mapped_to_unknown":
            return self._result(original, cleaned, normalized, status, applied_rules, review_reasons)

        slot_map = self.class_slot_maps.get(class_name_raw, {}).get(slot, {})
        key = self._clean_key(normalized)

        if slot in self.viewpoint_slots:
            normalized, status, applied_rules = self.apply_simple_map(
                normalized, status, applied_rules, self.viewpoint_map, "slot.viewpoint"
            )
        elif slot in self.state_slots:
            normalized, status, applied_rules = self.apply_simple_map(
                normalized, status, applied_rules, self.state_map, "slot.state"
            )
        elif slot in self.shape_slots:
            normalized, status, applied_rules = self.apply_simple_map(
                normalized, status, applied_rules, self.shape_map, "slot.shape"
            )
        elif slot in self.part_slots:
            normalized, status, applied_rules = self.apply_simple_map(
                normalized, status, applied_rules, self.part_map, "slot.part"
            )
        elif slot in self.background_slots:
            normalized, status, applied_rules = self.apply_simple_map(
                normalized, status, applied_rules, self.background_map, "slot.background"
            )

        if slot in self.state_slots:
            normalized, status, applied_rules = self.normalize_mixed_state(normalized, status, applied_rules)
            normalized, status, applied_rules = self.normalize_state_pattern(normalized, status, applied_rules)

        if slot in self.material_slots:
            normalized, status, applied_rules = self.normalize_material(normalized, status, applied_rules)
        elif slot in self.color_slots:
            normalized, status, applied_rules = self.normalize_color(normalized, status, applied_rules)
        elif slot in self.type_slots:
            normalized, status, applied_rules = self.normalize_type_like(normalized, status, applied_rules)

        key = self._clean_key(normalized)
        if key in slot_map:
            mapped = slot_map[key]
            if mapped != normalized:
                normalized = mapped
                status = "class_inferred"
                applied_rules.append(f"class.{class_name_raw}.{slot}")

        review_reasons.extend(self.detect_review_reasons(class_name_raw, slot, normalized))
        if review_reasons:
            status = "review_required"
            applied_rules.extend(sorted(set(review_reasons)))

        return self._result(original, cleaned, normalized, status, applied_rules, review_reasons)

    def _result(
        self,
        original: Any,
        cleaned: str,
        normalized: str,
        status: str,
        applied_rules: list[str],
        review_reasons: list[str],
    ) -> dict[str, Any]:
        changed = status in STATUS_CHANGED or normalized != cleaned
        return {
            "raw_value": original,
            "cleaned_value": cleaned,
            "normalized_value": normalized,
            "status": status,
            "changed": changed,
            "applied_rules": list(dict.fromkeys(applied_rules)),
            "review_reasons": list(dict.fromkeys(review_reasons)),
        }

    def clean_value(self, value: Any) -> str:
        if value is None:
            return "unknown"
        text = str(value).strip()
        text = text.replace("_", " ")
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\s*/\s*", "/", text)
        text = re.sub(r"\s*,\s*", ", ", text)
        return text.casefold()

    def is_unknown_like(self, value: str) -> bool:
        return self._clean_key(value) in self.unknown_like

    def normalize_low_value_contamination(
        self, slot: str, value: str, status: str, applied_rules: list[str]
    ) -> tuple[str, str, list[str]]:
        key = self._clean_key(value)
        if slot in self.background_slots and key in self.background_color_markers:
            applied_rules.append("slot.background.low_value_to_unknown")
            return "unknown", "mapped_to_unknown", applied_rules
        if slot in self.person_to_unknown_slots and (key in self.person_unknown_phrases or any(marker in key for marker in self.person_markers)):
            applied_rules.append("slot.person_contamination_to_unknown")
            return "unknown", "mapped_to_unknown", applied_rules
        return value, status, applied_rules

    def apply_simple_map(
        self,
        value: str,
        status: str,
        applied_rules: list[str],
        mapping: dict[str, str],
        rule_prefix: str,
    ) -> tuple[str, str, list[str]]:
        key = self._clean_key(value)
        if key in mapping and mapping[key] != value:
            applied_rules.append(f"{rule_prefix}.canonicalize")
            return mapping[key], "canonicalized", applied_rules
        return value, status, applied_rules

    def normalize_type_like(self, value: str, status: str, applied_rules: list[str]) -> tuple[str, str, list[str]]:
        singular_map = {
            "golf balls": "golf ball",
            "crosses": "cross",
            "towers": "tower",
        }
        key = self._clean_key(value)
        if key in singular_map:
            applied_rules.append("slot.type.singularize")
            return singular_map[key], "canonicalized", applied_rules
        return value, status, applied_rules

    def normalize_mixed_state(self, value: str, status: str, applied_rules: list[str]) -> tuple[str, str, list[str]]:
        key = self._clean_key(value)
        if "," not in key:
            return value, status, applied_rules

        parts = [part.strip() for part in key.split(",") if part.strip()]
        if len(parts) != 2:
            return value, status, applied_rules

        primary, secondary = parts
        if primary in self.mixed_state_primary_values and len(secondary) <= 32 and not re.search(r"\d", secondary):
            applied_rules.append("slot.state.primary_clause")
            return primary, "canonicalized", applied_rules
        return value, status, applied_rules

    def normalize_state_pattern(self, value: str, status: str, applied_rules: list[str]) -> tuple[str, str, list[str]]:
        key = self._clean_key(value)
        if key.startswith("held") or key.startswith("being held"):
            applied_rules.append("slot.state.held_family")
            return "being held", "canonicalized", applied_rules
        if key.startswith("powered on") or key == "active":
            applied_rules.append("slot.state.on_family")
            return "on", "canonicalized", applied_rules
        if key.startswith("powered off") or key == "inactive":
            applied_rules.append("slot.state.off_family")
            return "off", "canonicalized", applied_rules
        return value, status, applied_rules

    def normalize_material(self, value: str, status: str, applied_rules: list[str]) -> tuple[str, str, list[str]]:
        key = self._clean_key(value)
        base_map = {
            "metallic": "metal",
            "wooden": "wood",
        }
        if key in base_map:
            applied_rules.append("slot.material.base_material")
            return base_map[key], "canonicalized", applied_rules

        if any(sep in key for sep in [" and ", "/", ","]):
            parts = self._split_compound_value(key)
            part_map = {"metallic": "metal", "wooden": "wood"}
            canonical_parts = [part_map.get(part, part) for part in parts if part not in self.unknown_like]
            deduped = sorted(dict.fromkeys(canonical_parts))
            if len(deduped) >= 2:
                applied_rules.append("slot.material.separator_unify")
                return "+".join(deduped), "canonicalized", applied_rules
        return value, status, applied_rules

    def normalize_color(self, value: str, status: str, applied_rules: list[str]) -> tuple[str, str, list[str]]:
        key = self._clean_key(value)
        base_map = {
            "grey": "gray",
            "golden": "gold",
        }
        if key in base_map:
            applied_rules.append("slot.color.base_color")
            return base_map[key], "canonicalized", applied_rules

        if any(sep in key for sep in [" and ", ","]):
            parts = self._split_compound_value(key)
            deduped = sorted(dict.fromkeys(parts))
            if len(deduped) >= 2:
                applied_rules.append("slot.color.separator_unify")
                return "+".join(deduped), "canonicalized", applied_rules
        return value, status, applied_rules

    def _split_compound_value(self, value: str) -> list[str]:
        tmp = value.replace(" and ", ",")
        tmp = tmp.replace("/", ",")
        return [piece.strip() for piece in tmp.split(",") if piece.strip()]

    def detect_review_reasons(self, class_name_raw: str, slot: str, value: str) -> list[str]:
        reasons: list[str] = []
        key = self._clean_key(value)
        class_review_values = self.review_value_sets.get(class_name_raw, {}).get(slot, set())
        if key in class_review_values:
            reasons.append("review.wrong_object_candidate")

        if slot in self.part_slots | self.state_slots and any(marker in key for marker in self.person_markers):
            reasons.append("review.person_mention")

        if slot in self.background_slots and key in self.background_color_markers:
            reasons.append("review.slot_contamination")

        if slot in self.viewpoint_slots and re.search(r"\d", key):
            reasons.append("review.misplaced_spec_text")

        if len(key) > 40 and any(marker in key for marker in self.narrative_markers):
            reasons.append("review.narrative_value")

        if slot in self.state_slots and "," in key:
            reasons.append("review.mixed_state")

        return reasons


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield line_number, json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize Stage 1 attributes.jsonl with conservative deterministic rules.")
    parser.add_argument("--input", required=True, help="Path to Stage 1 attributes.jsonl")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for normalized artifacts. Defaults to sibling directory named <input_stem>_normalized.",
    )
    parser.add_argument(
        "--rules",
        default=str(DEFAULT_RULES_PATH),
        help="Path to normalization rules JSON.",
    )
    return parser


def normalize_file(input_path: Path, output_dir: Path, rules_path: Path) -> dict[str, Any]:
    rules = json.loads(rules_path.read_text(encoding="utf-8"))
    normalizer = Normalizer(rules)
    output_dir.mkdir(parents=True, exist_ok=True)

    normalized_path = output_dir / "attributes_normalized.jsonl"
    audit_path = output_dir / "normalization_audit.jsonl"
    review_path = output_dir / "normalization_review_queue.jsonl"
    summary_path = output_dir / "normalization_summary.json"
    snapshot_path = output_dir / "normalization_rules_snapshot.json"

    summary: dict[str, Any] = {
        "input_path": str(input_path.resolve()),
        "rules_path": str(rules_path.resolve()),
        "num_rows": 0,
        "num_success_rows": 0,
        "status_counts": Counter(),
        "slot_status_counts": defaultdict(Counter),
        "class_status_counts": defaultdict(Counter),
        "rule_counts": Counter(),
        "review_reason_counts": Counter(),
    }

    with (
        normalized_path.open("w", encoding="utf-8") as normalized_handle,
        audit_path.open("w", encoding="utf-8") as audit_handle,
        review_path.open("w", encoding="utf-8") as review_handle,
    ):
        for line_number, row in iter_jsonl(input_path):
            summary["num_rows"] += 1
            if row.get("extraction_status") == "success":
                summary["num_success_rows"] += 1

            class_name_raw = str(row.get("class_name_raw") or "")
            raw_attributes = row.get("attributes", {})
            if not isinstance(raw_attributes, dict):
                raw_attributes = {}

            normalized_attributes: dict[str, str] = {}
            normalization_meta: dict[str, Any] = {}
            row_has_review = False

            for slot, raw_value in raw_attributes.items():
                result = normalizer.normalize_field(class_name_raw, slot, raw_value)
                normalized_attributes[slot] = result["normalized_value"]
                normalization_meta[slot] = {
                    "raw_value": result["raw_value"],
                    "normalized_value": result["normalized_value"],
                    "status": result["status"],
                    "applied_rules": result["applied_rules"],
                    "review_reasons": result["review_reasons"],
                }
                summary["status_counts"][result["status"]] += 1
                summary["slot_status_counts"][slot][result["status"]] += 1
                summary["class_status_counts"][class_name_raw][result["status"]] += 1
                for rule in result["applied_rules"]:
                    summary["rule_counts"][rule] += 1
                for reason in result["review_reasons"]:
                    summary["review_reason_counts"][reason] += 1

                if result["changed"] or result["review_reasons"]:
                    audit_record = {
                        "line_number": line_number,
                        "record_id": row.get("record_id"),
                        "class_name_raw": class_name_raw,
                        "slot": slot,
                        "raw_value": result["raw_value"],
                        "normalized_value": result["normalized_value"],
                        "status": result["status"],
                        "applied_rules": result["applied_rules"],
                        "review_reasons": result["review_reasons"],
                    }
                    audit_handle.write(json.dumps(audit_record, ensure_ascii=False) + "\n")

                if result["status"] == "review_required":
                    row_has_review = True
                    review_record = {
                        "line_number": line_number,
                        "record_id": row.get("record_id"),
                        "class_name_raw": class_name_raw,
                        "slot": slot,
                        "raw_value": result["raw_value"],
                        "normalized_value": result["normalized_value"],
                        "review_reasons": result["review_reasons"],
                    }
                    review_handle.write(json.dumps(review_record, ensure_ascii=False) + "\n")

            normalized_row = dict(row)
            normalized_row["normalized_attributes"] = normalized_attributes
            normalized_row["attribute_normalization"] = normalization_meta
            normalized_row["normalization_review_required"] = row_has_review
            normalized_handle.write(json.dumps(normalized_row, ensure_ascii=False) + "\n")

    summary_payload = {
        "input_path": summary["input_path"],
        "rules_path": summary["rules_path"],
        "num_rows": summary["num_rows"],
        "num_success_rows": summary["num_success_rows"],
        "status_counts": dict(summary["status_counts"]),
        "slot_status_counts": {slot: dict(counter) for slot, counter in summary["slot_status_counts"].items()},
        "class_status_counts": {cls: dict(counter) for cls, counter in summary["class_status_counts"].items()},
        "rule_counts": dict(summary["rule_counts"].most_common()),
        "review_reason_counts": dict(summary["review_reason_counts"].most_common()),
        "artifacts": {
            "attributes_normalized": str(normalized_path.resolve()),
            "normalization_audit": str(audit_path.resolve()),
            "normalization_review_queue": str(review_path.resolve()),
            "normalization_rules_snapshot": str(snapshot_path.resolve()),
        },
    }

    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    snapshot_path.write_text(json.dumps(rules, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_payload


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file does not exist: {input_path}")

    rules_path = Path(args.rules)
    if not rules_path.exists():
        raise SystemExit(f"Rules file does not exist: {rules_path}")

    output_dir = Path(args.output_dir) if args.output_dir else input_path.with_name(f"{input_path.stem}_normalized")
    summary = normalize_file(input_path=input_path, output_dir=output_dir, rules_path=rules_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
