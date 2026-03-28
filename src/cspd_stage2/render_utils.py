from __future__ import annotations

import re
from typing import Any

UNKNOWN_VALUES = {"unknown", "not_applicable", "n/a", "none", "null", ""}
LOW_VALUE_FOCUS_VALUES = {
    "fish",
    "entire fish",
    "entire body",
    "entire object",
    "object",
    "body",
}
LOW_VALUE_BACKGROUND_VALUES = {
    "grass",
    "water",
    "outdoor",
    "indoors",
    "indoor",
    "greenery",
}
LOW_VALUE_POSE_VALUES = {
    "being held",
    "standing",
    "resting",
}
VIEWPOINT_MAP = {
    "frontal": "front view",
    "side view": "side view",
    "top-down": "top-down view",
    "rear view": "rear view",
    "close-up": "close-up view",
}


def normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = " ".join(text.split())
    return text


def is_unknown_like(value: Any) -> bool:
    text = normalize_text(value)
    if text is None:
        return True
    return text.casefold() in UNKNOWN_VALUES


def needs_an(text: str) -> bool:
    lowered = text.strip().casefold()
    return lowered.startswith(("a", "e", "i", "o", "u"))


def with_article(noun_phrase: str) -> str:
    phrase = normalize_text(noun_phrase)
    if not phrase:
        return ""
    article = "an" if needs_an(phrase) else "a"
    return f"{article} {phrase}"


def cleanup_caption(text: str) -> str:
    cleaned = " ".join(text.split())
    cleaned = cleaned.replace(" ,", ",")
    cleaned = cleaned.replace(" .", ".")
    cleaned = cleaned.replace("  ", " ")
    return cleaned.strip(" ,.")


def stringify_slot_value(value: Any) -> str | None:
    text = normalize_text(value)
    if text is None:
        return None
    return text


def class_name_to_anchor(class_name: Any) -> str | None:
    text = normalize_text(class_name)
    if not text:
        return None
    primary = text.split(",", 1)[0].strip()
    return primary.casefold() if primary else None


def clean_pre_anchor_value(slot: str, value: str) -> str | None:
    text = normalize_text(value)
    if not text:
        return None
    lowered = text.casefold()
    if slot in {"body_trait", "shape_or_structure", "shape_or_growth_form", "architectural_style_or_form"}:
        text = text.replace(",", " and")
        text = re.sub(r"\s+and\s+", " and ", text)
        text = text.replace(" and and ", " and ")
        return text
    return text


def should_drop_slot(archetype: str, slot: str, value: str, review_required: bool = False) -> tuple[bool, str | None]:
    text = normalize_text(value)
    if not text:
        return True, "unknown_or_empty"
    lowered = text.casefold()

    if review_required and slot != "species_or_category" and not slot.endswith("_type"):
        return True, "review_required"

    if slot in {"salient_part_or_focus", "salient_part_or_accessory", "salient_structural_part"} and lowered in LOW_VALUE_FOCUS_VALUES:
        return True, "low_value_focus"

    if slot == "viewpoint" and lowered in {"frontal", "front", "front view"}:
        return True, "default_viewpoint"

    if archetype == "animal" and slot == "background_or_habitat" and lowered in LOW_VALUE_BACKGROUND_VALUES:
        return True, "low_value_background"

    if archetype == "animal" and slot == "pose_or_state" and lowered in LOW_VALUE_POSE_VALUES:
        return True, "low_value_pose"

    if slot in {"body_trait", "shape_or_structure"} and lowered in {"fish", "object"}:
        return True, "low_value_trait"

    return False, None


def format_post_slot(slot: str, value: str) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    lowered = text.casefold()

    if slot == "viewpoint":
        return VIEWPOINT_MAP.get(lowered, f"{text} view" if not lowered.endswith("view") else text)
    if slot in {"background_or_habitat", "background_or_context", "environment", "surrounding_environment", "background_or_room_context", "background_or_activity_context", "vegetation_or_natural_context", "container_or_context", "display_or_usage_context"}:
        return f"in {text}"
    if slot in {"salient_part_or_focus", "salient_part_or_accessory", "salient_structural_part", "held_object_or_equipment", "salient_geographic_feature"}:
        return f"with {text}"
    if slot in {"content_or_symbol_type", "visible_part"}:
        return f"showing {text}"
    if slot in {"weather_or_water_state"}:
        return f"with {text}"
    return text
