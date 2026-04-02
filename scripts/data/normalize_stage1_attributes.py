from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from cspd_stage1.schema import SampleRecord
from cspd_stage1.templates import TEMPLATE_SPECS
from cspd_stage1.vlm.base import BaseVLMClient, VLMOutputParseError
from cspd_stage1.vlm.factory import create_vlm_client

DEFAULT_RULES_PATH = Path("configs/stage1/normalization/stage1_attribute_normalization_rules.json")
STATUS_CHANGED = {"canonicalized", "class_inferred", "mapped_to_unknown", "review_required"}

REVIEW_REASON_PRIORITIES = {
    "review.food_anchor_context_drift": 120,
    "review.structure_type_style_conflict": 115,
    "review.weapon_anchor_related_object": 115,
    "review.cross_slot_value_conflict": 100,
    "review.wrong_object_candidate": 95,
    "review.archetype_anchor_mismatch": 90,
    "review.person_mention": 60,
    "review.slot_contamination": 55,
    "review.misplaced_spec_text": 45,
    "review.narrative_value": 40,
    "review.mixed_state": 35,
}

REVIEW_SLOT_PRIORITIES = {
    "food_or_drink_type": 60,
    "structure_or_building_type": 60,
    "weapon_type": 60,
    "architectural_style_or_form": 55,
    "shape_or_structure": 45,
    "container_or_context": 40,
    "background_or_context": 30,
    "surrounding_environment": 25,
}


class Normalizer:
    def __init__(self, rules: dict[str, Any]) -> None:
        self.rules = rules
        slot_groups = rules.get("slot_alias_groups", {})
        self.type_slots = set(slot_groups.get("type_slots", []))
        self.anchor_type_slots = {
            spec.anchor_slot for spec in TEMPLATE_SPECS.values() if spec.anchor_slot in self.type_slots
        }
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
        self.type_map = self._clean_mapping(rules.get("type_map", {}))
        self.archetype_review_value_sets = {
            archetype: {slot: {self._clean_key(v) for v in values} for slot, values in slot_maps.items()}
            for archetype, slot_maps in rules.get("archetype_review_value_sets", {}).items()
        }
        self.archetype_unknown_value_sets = {
            archetype: {slot: {self._clean_key(v) for v in values} for slot, values in slot_maps.items()}
            for archetype, slot_maps in rules.get("archetype_unknown_value_sets", {}).items()
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
        v3_rules = rules.get("v3_refinement", {})
        self.low_value_background_phrases = {self._clean_key(v) for v in v3_rules.get("low_value_background_phrases", [])}
        self.low_value_state_phrases = {self._clean_key(v) for v in v3_rules.get("low_value_state_phrases", [])}
        self.low_value_shape_phrases = {self._clean_key(v) for v in v3_rules.get("low_value_shape_phrases", [])}
        self.low_value_part_phrases = {self._clean_key(v) for v in v3_rules.get("low_value_part_phrases", [])}
        self.non_ascii_to_unknown_slots = set(v3_rules.get("non_ascii_to_unknown_slots", []))
        self.background_phrase_map = self._clean_mapping(v3_rules.get("background_phrase_map", {}))
        self.state_phrase_map = self._clean_mapping(v3_rules.get("state_phrase_map", {}))

    @staticmethod
    def _clean_key(value: Any) -> str:
        text = "" if value is None else str(value)
        text = text.strip().casefold()
        text = text.replace("_", " ")
        text = re.sub(r"\s+", " ", text)
        return text

    def _clean_mapping(self, mapping: dict[str, str]) -> dict[str, str]:
        return {self._clean_key(k): v for k, v in mapping.items()}

    def normalize_field(self, class_name: str, class_name_raw: str, archetype: str, slot: str, raw_value: Any) -> dict[str, Any]:
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

        normalized, status, applied_rules = self.normalize_v3_render_awareness(slot, normalized, status, applied_rules)
        if status == "mapped_to_unknown":
            return self._result(original, cleaned, normalized, status, applied_rules, review_reasons)

        normalized, status, applied_rules = self.normalize_archetype_awareness(archetype, slot, normalized, status, applied_rules)
        if status == "mapped_to_unknown":
            return self._result(original, cleaned, normalized, status, applied_rules, review_reasons)

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
            normalized, status, applied_rules = self.normalize_type_like(
                class_name, slot, normalized, status, applied_rules
            )

        review_reasons.extend(self.detect_review_reasons(class_name_raw, archetype, slot, normalized))
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

    def normalize_v3_render_awareness(
        self, slot: str, value: str, status: str, applied_rules: list[str]
    ) -> tuple[str, str, list[str]]:
        key = self._clean_key(value)

        if slot in self.non_ascii_to_unknown_slots and any(ord(ch) > 127 for ch in value):
            applied_rules.append(f"slot.{slot}.non_ascii_to_unknown")
            return "unknown", "mapped_to_unknown", applied_rules

        if slot in self.background_slots:
            if key in self.background_phrase_map:
                mapped = self.background_phrase_map[key]
                if mapped == "unknown":
                    applied_rules.append("slot.background.low_value_to_unknown")
                    return "unknown", "mapped_to_unknown", applied_rules
                if mapped != value:
                    applied_rules.append("slot.background.canonicalize")
                    return mapped, "canonicalized", applied_rules
            if key in self.low_value_background_phrases:
                applied_rules.append("slot.background.low_value_to_unknown")
                return "unknown", "mapped_to_unknown", applied_rules

        if slot in self.state_slots:
            if key in self.state_phrase_map:
                mapped = self.state_phrase_map[key]
                if mapped != value:
                    applied_rules.append("slot.state.canonicalize")
                    return mapped, "canonicalized", applied_rules
            if key in self.low_value_state_phrases:
                applied_rules.append("slot.state.low_value_preserved")
                return value, status, applied_rules

        if slot in self.shape_slots and key in self.low_value_shape_phrases:
            applied_rules.append("slot.shape.low_value_preserved")
            return value, status, applied_rules

        if slot in self.part_slots and key in self.low_value_part_phrases:
            applied_rules.append("slot.part.low_value_to_unknown")
            return "unknown", "mapped_to_unknown", applied_rules

        return value, status, applied_rules

    def normalize_archetype_awareness(
        self, archetype: str, slot: str, value: str, status: str, applied_rules: list[str]
    ) -> tuple[str, str, list[str]]:
        key = self._clean_key(value)
        unknown_values = self.archetype_unknown_value_sets.get(archetype, {}).get(slot, set())
        if key in unknown_values:
            applied_rules.append(f"archetype.{archetype}.{slot}.to_unknown")
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

    def normalize_type_like(
        self,
        class_name: str,
        slot: str,
        value: str,
        status: str,
        applied_rules: list[str],
    ) -> tuple[str, str, list[str]]:
        key = self._clean_key(value)

        if slot in self.anchor_type_slots:
            class_short_name = self.extract_class_short_name(class_name)
            if class_short_name and class_short_name != value:
                applied_rules.append("slot.anchor_type.from_class_name_first_item")
                return class_short_name, "canonicalized", applied_rules
            return value, status, applied_rules

        if key in self.type_map:
            mapped = self.type_map[key]
            if mapped != value:
                applied_rules.append("slot.type.canonicalize")
                return mapped, "canonicalized", applied_rules

        singular_map = {
            "golf balls": "golf ball",
            "crosses": "cross",
            "towers": "tower",
        }
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

    def extract_class_short_name(self, class_name: Any) -> str:
        if class_name is None:
            return ""
        text = str(class_name).strip()
        if not text:
            return ""
        first_item = text.split(",", 1)[0].strip()
        if not first_item:
            return ""
        return self.clean_value(first_item)

    def _split_compound_value(self, value: str) -> list[str]:
        tmp = value.replace(" and ", ",")
        tmp = tmp.replace("/", ",")
        return [piece.strip() for piece in tmp.split(",") if piece.strip()]

    def detect_review_reasons(self, class_name_raw: str, archetype: str, slot: str, value: str) -> list[str]:
        reasons: list[str] = []
        key = self._clean_key(value)

        archetype_review_values = self.archetype_review_value_sets.get(archetype, {}).get(slot, set())
        if key in archetype_review_values:
            reasons.append("review.archetype_anchor_mismatch")

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


def _append_review_reason(slot_meta: dict[str, Any], reason: str) -> None:
    reasons = [str(item) for item in slot_meta.get("review_reasons") or []]
    if reason not in reasons:
        reasons.append(reason)
    slot_meta["review_reasons"] = reasons
    slot_meta["status"] = "review_required"
    applied_rules = [str(item) for item in slot_meta.get("applied_rules") or []]
    if reason not in applied_rules:
        applied_rules.append(reason)
    slot_meta["applied_rules"] = applied_rules


def _tokenize_review_value(value: Any) -> set[str]:
    text = "" if value is None else str(value).casefold()
    return {token for token in re.findall(r"[a-z]+", text) if len(token) >= 3}


def apply_consistency_review_pass(archetype: str, normalized_attributes: dict[str, str], normalization_meta: dict[str, Any]) -> None:
    if not isinstance(normalized_attributes, dict) or not isinstance(normalization_meta, dict):
        return

    def slot_value(slot: str) -> str:
        return str(normalized_attributes.get(slot) or "").strip()

    def clean(slot: str) -> str:
        return Normalizer._clean_key(slot_value(slot))

    def add(slot: str, reason: str) -> None:
        slot_meta = normalization_meta.get(slot)
        if isinstance(slot_meta, dict):
            _append_review_reason(slot_meta, reason)

    if archetype == "food_and_drink":
        anchor_slot = "food_or_drink_type"
        anchor_value = clean(anchor_slot)
        food_shape = clean("shape_or_structure")
        food_context = clean("container_or_context")
        vessel_tokens = {"plate", "bowl", "tray", "cup", "mug", "glass", "spoon", "fork", "knife", "skewer"}
        if anchor_value and anchor_value != "unknown":
            if _tokenize_review_value(anchor_value) & vessel_tokens:
                add(anchor_slot, "review.food_anchor_context_drift")
            if anchor_value in {food_shape, food_context} and anchor_value not in {"", "unknown"}:
                add(anchor_slot, "review.cross_slot_value_conflict")
                if anchor_value == food_context:
                    add("container_or_context", "review.food_anchor_context_drift")

    if archetype == "structure_or_building":
        type_slot = "structure_or_building_type"
        style_slot = "architectural_style_or_form"
        type_value = clean(type_slot)
        style_value = clean(style_slot)
        structure_type_tokens = {"bridge", "tower", "church", "mosque", "temple", "castle", "palace", "stadium", "barn", "house", "building", "skyscraper", "lighthouse", "warehouse", "dome"}
        style_tokens = {"gothic", "baroque", "victorian", "modern", "modernist", "roman", "romanesque", "classical", "suspension", "arched"}
        if style_value and style_value != "unknown" and (_tokenize_review_value(style_value) & structure_type_tokens):
            add(style_slot, "review.structure_type_style_conflict")
        if type_value and type_value != "unknown" and (_tokenize_review_value(type_value) & style_tokens):
            add(type_slot, "review.structure_type_style_conflict")
        if type_value and style_value and type_value == style_value and type_value != "unknown":
            add(type_slot, "review.cross_slot_value_conflict")
            add(style_slot, "review.cross_slot_value_conflict")

    if archetype == "weapon":
        anchor_slot = "weapon_type"
        anchor_value = clean(anchor_slot)
        context_value = clean("background_or_context")
        related_object_tokens = {"holster", "scabbard", "sheath", "ammo", "ammunition", "bullet", "target", "case"}
        if anchor_value and anchor_value != "unknown":
            if _tokenize_review_value(anchor_value) & related_object_tokens:
                add(anchor_slot, "review.weapon_anchor_related_object")
            if anchor_value == context_value and anchor_value not in {"", "unknown"}:
                add(anchor_slot, "review.cross_slot_value_conflict")
                add("background_or_context", "review.weapon_anchor_related_object")


def compute_review_priority(slot: str, slot_meta: dict[str, Any]) -> int:
    reasons = [str(item) for item in slot_meta.get("review_reasons") or []]
    base = REVIEW_SLOT_PRIORITIES.get(slot, 0)
    if not reasons and str(slot_meta.get("status") or "") == "review_required":
        base += 10
    return base + sum(REVIEW_REASON_PRIORITIES.get(reason, 20) for reason in reasons)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield line_number, json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc


DEFAULT_REVIEW_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_ALLOWED_ACTIONS = ("keep_normalized", "replace_normalized", "set_unknown", "defer")
REVIEW_SYSTEM_PROMPT = (
    "You are a careful normalization review assistant for image attributes. "
    "Return JSON only. Never change the provided archetype or slot. "
    "The deterministic normalized value is a candidate, not ground truth. "
    "Prefer preserving a coarse, slot-compatible semantic value when it is reasonably supported by the image, even if fine detail is uncertain. "
    "Use keep_normalized when the current value is broadly supported. "
    "Use replace_normalized when another short slot-compatible value is more clearly supported. "
    "Use set_unknown only when the slot is truly unverifiable, contradicted, contaminated by the wrong object/background, or too unclear to support even a coarse semantic label. "
    "Reserve defer for true contract problems, unreadable images, or cases that genuinely need manual follow-up."
)


def build_review_prompt(row: dict[str, Any], slot: str, slot_meta: dict[str, Any]) -> str:
    payload_hint = {
        "record_id": str(row.get("record_id") or ""),
        "archetype": str(row.get("archetype") or ""),
        "slot": slot,
        "action": "one of: " + " | ".join(DEFAULT_ALLOWED_ACTIONS),
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
    current_value = slot_meta.get("normalized_value", normalized_attributes.get(slot, "unknown"))
    context_attributes = {k: normalized_attributes.get(k, "unknown") for k in (row.get("slot_schema") or []) if k != slot}
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
        f"normalized_slot_value: {json.dumps(current_value, ensure_ascii=False)}\n"
        f"other_normalized_slots: {json.dumps(context_attributes, ensure_ascii=False)}\n"
        "Decision policy:\n"
        "- keep_normalized: use when the current normalized value is broadly supported for this exact slot, even if some fine detail remains uncertain.\n"
        "- replace_normalized: use when another short value is more clearly supported by the image. Do not elaborate or rewrite beyond this slot.\n"
        "- set_unknown: use only when the slot is truly not verifiable, clearly conflicting, heavily contaminated by the wrong object/background, or too unclear to support even a coarse slot-compatible label.\n"
        "- defer: use only for exceptional manual-review situations such as unreadable/corrupted image, slot/archetype contract confusion, or impossible judgment even after applying the rules above. Do not use defer for normal ambiguity.\n"
        "Constraints:\n"
        "- The current normalized value is only a candidate. It may be wrong precisely because this item was routed for review.\n"
        "- Prefer preserving coarse semantics over collapsing to unknown when the image reasonably supports them.\n"
        "- This especially applies to high-level state/action/context/viewpoint labels.\n"
        "- Do NOT change archetype.\n"
        "- Do NOT answer for any other slot.\n"
        "- Keep output short, slot-compatible, and render-friendly.\n"
        "- Do NOT invent hidden details. If you truly cannot verify the slot visually, choose set_unknown.\n"
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
    return {
        "record_id": str(row.get("record_id") or ""),
        "archetype": str(row.get("archetype") or ""),
        "slot": slot,
        "action": "keep_normalized",
        "reviewed_value": normalized_value,
        "confidence": "medium",
        "needs_manual_followup": False,
        "reason": "mock backend preserves deterministic candidate for plumbing-only review tests",
    }


def should_apply_review_decision(slot: str, slot_meta: dict[str, Any], parsed: dict[str, Any]) -> tuple[bool, str]:
    action = str(parsed.get("action") or "defer")
    confidence = str(parsed.get("confidence") or "low").strip().lower()
    review_reasons = {str(item) for item in slot_meta.get("review_reasons") or []}
    severity_unknown_ok = {
        "review.wrong_object_candidate",
        "review.archetype_anchor_mismatch",
        "review.food_anchor_context_drift",
        "review.weapon_anchor_related_object",
        "review.cross_slot_value_conflict",
        "review.person_mention",
        "review.slot_contamination",
    }
    if action == "replace_normalized":
        return confidence in {"high", "medium"}, "accepted_replace" if confidence in {"high", "medium"} else "rejected_low_confidence_replace"
    if action == "set_unknown":
        allow = confidence == "high" or bool(review_reasons & severity_unknown_ok)
        return allow, "accepted_set_unknown" if allow else "rejected_low_confidence_set_unknown"
    return False, "not_applicable"


def parse_review_payload(payload: dict[str, Any], row: dict[str, Any], slot: str, slot_meta: dict[str, Any]) -> dict[str, Any]:
    action = str(payload.get("action") or "defer").strip()
    if action not in DEFAULT_ALLOWED_ACTIONS:
        action = "defer"
    current_value = str(slot_meta.get("normalized_value") or "unknown")
    reviewed_value = str(payload.get("reviewed_value") or "").strip()
    if not reviewed_value:
        reviewed_value = current_value if action in {"keep_normalized", "defer"} else "unknown"
    if action == "keep_normalized":
        reviewed_value = current_value
    elif action == "set_unknown":
        reviewed_value = "unknown"
    confidence = str(payload.get("confidence") or "low").strip().lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"
    if str(payload.get("archetype") or row.get("archetype") or "") != str(row.get("archetype") or ""):
        action = "defer"
        reviewed_value = current_value
        confidence = "low"
    if str(payload.get("slot") or slot) != slot:
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
        if str(slot_meta.get("status") or "") == "review_required" or bool(slot_meta.get("review_reasons") or []):
            slot_meta = dict(slot_meta)
            slot_meta["review_priority"] = compute_review_priority(str(slot), slot_meta)
            items.append((str(slot), slot_meta))
    items.sort(key=lambda item: (-int(item[1].get("review_priority") or 0), item[0]))
    return items


def run_inline_vlm_review(rows: list[dict[str, Any]], review_backend: str, review_model_name: str, review_torch_dtype: str, review_device_map: str, review_use_fast_processor: bool, review_max_new_tokens: int, review_limit: int | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    client: BaseVLMClient | None = None
    if review_backend != "mock":
        client = create_vlm_client(
            review_backend,
            model_name=review_model_name,
            torch_dtype=review_torch_dtype,
            device_map=review_device_map,
            use_fast_processor=review_use_fast_processor,
            max_new_tokens=review_max_new_tokens,
        )
    review_records: list[dict[str, Any]] = []
    counts = Counter()
    reason_counts = Counter()
    action_counts = Counter()
    for row in rows:
        items = select_review_items(row)
        if not items:
            row["vlm_review"] = {"reviewed_slots": {}, "review_status": "not_needed"}
            row["effective_normalized_attributes"] = dict(row.get("normalized_attributes") or {})
            continue
        reviewed_slots: dict[str, Any] = {}
        effective_attributes = dict(row.get("normalized_attributes") or {})
        for slot, slot_meta in items:
            if review_limit is not None and counts["items_total"] >= review_limit:
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
                "review_priority": int(slot_meta.get("review_priority") or compute_review_priority(slot, slot_meta)),
                "raw_value": slot_meta.get("raw_value"),
                "normalized_value": slot_meta.get("normalized_value"),
                "review_status": "ok",
            }
            try:
                if review_backend == "mock":
                    parsed = run_mock_review(row, slot, slot_meta)
                    raw_text = json.dumps(parsed, ensure_ascii=False)
                else:
                    assert client is not None
                    response = client.extract_attributes(build_sample(row), build_review_prompt(row, slot, slot_meta), REVIEW_SYSTEM_PROMPT)
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
            apply_decision, application_reason = should_apply_review_decision(slot, slot_meta, parsed)
            parsed["applied_to_effective_normalized_attributes"] = apply_decision
            parsed["application_reason"] = application_reason
            if apply_decision:
                effective_attributes[slot] = parsed["reviewed_value"]
            reviewed_slots[slot] = parsed
            record["vlm_review"] = parsed
            record["raw_model_output"] = raw_text
            review_records.append(record)
        row["effective_normalized_attributes"] = effective_attributes
        row["vlm_review"] = {
            "reviewed_slots": reviewed_slots,
            "review_status": "completed" if reviewed_slots else "not_needed",
            "backend": review_backend,
            "model_name": review_model_name,
            "allowed_actions": list(DEFAULT_ALLOWED_ACTIONS),
        }
        if review_limit is not None and counts["items_total"] >= review_limit:
            break
    summary = {
        "backend": review_backend,
        "model_name": review_model_name,
        "num_review_items": counts["items_total"],
        "num_ok": counts["items_ok"],
        "num_failed": counts["items_failed"],
        "review_reason_counts": dict(reason_counts.most_common()),
        "action_counts": dict(action_counts.most_common()),
        "contract": {
            "allowed_actions": list(DEFAULT_ALLOWED_ACTIONS),
            "trigger": "Only slots whose normalization metadata has status=review_required or non-empty review_reasons are sent to the VLM review path.",
            "application_policy": "Deterministic normalization is preserved in normalized_attributes. effective_normalized_attributes applies inline VLM decisions for render/use while vlm_review stores the constrained review metadata.",
        },
    }
    return review_records, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize Stage 1 attributes.jsonl with deterministic rules plus inline constrained VLM review by default.")
    parser.add_argument("--input", required=True, help="Path to Stage 1 attributes.jsonl")
    parser.add_argument("--output-dir", default=None, help="Directory for normalized artifacts. Defaults to <input_dir>/normalization/<timestamp>.")
    parser.add_argument("--rules", default=str(DEFAULT_RULES_PATH), help="Path to normalization rules JSON.")
    parser.add_argument("--disable-vlm-review", action="store_true", help="Disable the default inline VLM review pass and keep the output purely deterministic.")
    parser.add_argument("--review-backend", default="qwen_local", help="VLM backend used for the inline review pass. Use mock for plumbing-only smoke tests.")
    parser.add_argument("--review-model-name", default=DEFAULT_REVIEW_MODEL_NAME, help="Model name for inline VLM review.")
    parser.add_argument("--review-torch-dtype", default="float16", help="Torch dtype for local inline review backends.")
    parser.add_argument("--review-device-map", default="auto", help="Transformers device_map for local inline review backends.")
    parser.add_argument("--disable-review-fast-processor", action="store_true", help="Use the slow processor for inline VLM review.")
    parser.add_argument("--review-max-new-tokens", type=int, default=256, help="Generation cap for inline VLM review.")
    parser.add_argument("--review-limit", type=int, default=None, help="Optional maximum number of ambiguous slots to send through inline VLM review.")
    return parser


def normalize_file(input_path: Path, output_dir: Path, rules_path: Path, *, enable_vlm_review: bool, review_backend: str, review_model_name: str, review_torch_dtype: str, review_device_map: str, review_use_fast_processor: bool, review_max_new_tokens: int, review_limit: int | None) -> dict[str, Any]:
    rules = json.loads(rules_path.read_text(encoding="utf-8"))
    normalizer = Normalizer(rules)
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_path = output_dir / "attributes_normalized.jsonl"
    audit_path = output_dir / "normalization_audit.jsonl"
    review_path = output_dir / "normalization_review_queue.jsonl"
    review_vlm_path = output_dir / "normalization_review_vlm.jsonl"
    summary_path = output_dir / "normalization_summary.json"
    review_vlm_summary_path = output_dir / "normalization_review_vlm_summary.json"
    snapshot_path = output_dir / "normalization_rules_snapshot.json"
    summary: dict[str, Any] = {"input_path": str(input_path.resolve()), "rules_path": str(rules_path.resolve()), "num_rows": 0, "num_success_rows": 0, "status_counts": Counter(), "slot_status_counts": defaultdict(Counter), "class_status_counts": defaultdict(Counter), "rule_counts": Counter(), "review_reason_counts": Counter()}
    output_rows: list[dict[str, Any]] = []
    audit_records: list[dict[str, Any]] = []
    review_records: list[dict[str, Any]] = []
    for line_number, row in iter_jsonl(input_path):
        summary["num_rows"] += 1
        if row.get("extraction_status") == "success":
            summary["num_success_rows"] += 1
        class_name = str(row.get("class_name") or "")
        class_name_raw = str(row.get("class_name_raw") or "")
        archetype = str(row.get("archetype") or "")
        raw_attributes = row.get("attributes", {})
        if not isinstance(raw_attributes, dict):
            raw_attributes = {}
        normalized_attributes: dict[str, str] = {}
        normalization_meta: dict[str, Any] = {}
        per_slot_results: dict[str, dict[str, Any]] = {}
        for slot, raw_value in raw_attributes.items():
            result = normalizer.normalize_field(class_name, class_name_raw, archetype, slot, raw_value)
            normalized_attributes[slot] = result["normalized_value"]
            normalization_meta[slot] = {"raw_value": result["raw_value"], "normalized_value": result["normalized_value"], "status": result["status"], "applied_rules": result["applied_rules"], "review_reasons": result["review_reasons"]}
            per_slot_results[slot] = result

        apply_consistency_review_pass(archetype, normalized_attributes, normalization_meta)

        row_has_review = False
        for slot, slot_meta in normalization_meta.items():
            result = per_slot_results[slot]
            status = str(slot_meta.get("status") or result["status"])
            reasons = [str(item) for item in slot_meta.get("review_reasons") or []]
            applied_rules = [str(item) for item in slot_meta.get("applied_rules") or []]
            summary["status_counts"][status] += 1
            summary["slot_status_counts"][slot][status] += 1
            summary["class_status_counts"][class_name_raw][status] += 1
            for rule in applied_rules:
                summary["rule_counts"][rule] += 1
            for reason in reasons:
                summary["review_reason_counts"][reason] += 1
            changed = status in STATUS_CHANGED or str(slot_meta.get("normalized_value") or "") != str(result["cleaned_value"])
            if changed or reasons:
                audit_records.append({"line_number": line_number, "record_id": row.get("record_id"), "class_name_raw": class_name_raw, "slot": slot, "raw_value": slot_meta.get("raw_value"), "normalized_value": slot_meta.get("normalized_value"), "status": status, "applied_rules": applied_rules, "review_reasons": reasons, "review_priority": compute_review_priority(slot, slot_meta) if reasons or status == "review_required" else 0})
            if status == "review_required" or reasons:
                row_has_review = True
                review_records.append({"line_number": line_number, "record_id": row.get("record_id"), "class_name_raw": class_name_raw, "slot": slot, "raw_value": slot_meta.get("raw_value"), "normalized_value": slot_meta.get("normalized_value"), "review_reasons": reasons, "review_priority": compute_review_priority(slot, slot_meta)})
        normalized_row = dict(row)
        normalized_row["normalized_attributes"] = normalized_attributes
        normalized_row["attribute_normalization"] = normalization_meta
        normalized_row["normalization_review_required"] = row_has_review
        normalized_row["effective_normalized_attributes"] = dict(normalized_attributes)
        normalized_row["vlm_review"] = {"reviewed_slots": {}, "review_status": "disabled" if not enable_vlm_review else "pending"}
        output_rows.append(normalized_row)
    if enable_vlm_review:
        vlm_review_records, vlm_review_summary = run_inline_vlm_review(output_rows, review_backend, review_model_name, review_torch_dtype, review_device_map, review_use_fast_processor, review_max_new_tokens, review_limit)
    else:
        vlm_review_records = []
        vlm_review_summary = {"backend": None, "model_name": None, "num_review_items": 0, "num_ok": 0, "num_failed": 0, "review_reason_counts": {}, "action_counts": {}, "contract": {"allowed_actions": list(DEFAULT_ALLOWED_ACTIONS), "trigger": "disabled", "application_policy": "Inline VLM review disabled; effective_normalized_attributes matches deterministic normalized_attributes."}}
    normalized_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows), encoding="utf-8")
    audit_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in audit_records), encoding="utf-8")
    review_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in review_records), encoding="utf-8")
    review_vlm_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in vlm_review_records), encoding="utf-8")
    summary_payload = {"input_path": summary["input_path"], "rules_path": summary["rules_path"], "num_rows": summary["num_rows"], "num_success_rows": summary["num_success_rows"], "status_counts": dict(summary["status_counts"]), "slot_status_counts": {slot: dict(counter) for slot, counter in summary["slot_status_counts"].items()}, "class_status_counts": {cls: dict(counter) for cls, counter in summary["class_status_counts"].items()}, "rule_counts": dict(summary["rule_counts"].most_common()), "review_reason_counts": dict(summary["review_reason_counts"].most_common()), "vlm_review": {"enabled": enable_vlm_review, **vlm_review_summary}, "artifacts": {"attributes_normalized": str(normalized_path.resolve()), "normalization_audit": str(audit_path.resolve()), "normalization_review_queue": str(review_path.resolve()), "normalization_review_vlm": str(review_vlm_path.resolve()), "normalization_review_vlm_summary": str(review_vlm_summary_path.resolve()), "normalization_rules_snapshot": str(snapshot_path.resolve())}}
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    review_vlm_summary_path.write_text(json.dumps(vlm_review_summary, ensure_ascii=False, indent=2), encoding="utf-8")
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
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent / "normalization" / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    summary = normalize_file(input_path=input_path, output_dir=output_dir, rules_path=rules_path, enable_vlm_review=not args.disable_vlm_review, review_backend=args.review_backend, review_model_name=args.review_model_name, review_torch_dtype=args.review_torch_dtype, review_device_map=args.review_device_map, review_use_fast_processor=not args.disable_review_fast_processor, review_max_new_tokens=args.review_max_new_tokens, review_limit=args.review_limit)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
