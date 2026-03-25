from __future__ import annotations

"""Core schema utilities for CSPD Stage 1.

Stage 1 now supports class-adaptive slot schemas instead of forcing every image
through one universal attribute template. The overall idea is:
- infer a semantic archetype from the class name,
- choose the slot family associated with that archetype,
- ask the VLM to fill only those slots.
"""

from dataclasses import dataclass
from typing import Any

ALLOWED_SPECIAL_VALUES = {"unknown", "not_applicable", None}

# Slot schemas keyed by semantic archetype. The exact slot names are chosen to
# be descriptive enough for downstream rendering while remaining short enough
# for reliable VLM JSON generation.
ARCHETYPE_SLOT_SCHEMAS: dict[str, list[str]] = {
    "animal": [
        "species_or_category",
        "color",
        "body_trait",
        "pose_or_state",
        "background_or_habitat",
        "viewpoint",
        "material",
    ],
    "vehicle": [
        "vehicle_type",
        "color",
        "shape_or_structure",
        "state_or_action",
        "environment",
        "viewpoint",
        "material",
    ],
    "food": [
        "food_type",
        "color",
        "shape_or_structure",
        "state_or_serving_style",
        "container_or_context",
        "viewpoint",
        "material",
    ],
    "instrument": [
        "instrument_type",
        "color",
        "shape_or_structure",
        "playing_state_or_pose",
        "background_or_context",
        "viewpoint",
        "material",
    ],
    "generic_object": [
        "object_type",
        "color",
        "shape_or_structure",
        "state_or_usage",
        "background_or_context",
        "viewpoint",
        "material",
    ],
}

DEFAULT_ARCHETYPE = "generic_object"

# Lightweight heuristics for mapping readable class names to semantic archetypes.
ARCHETYPE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "animal": (
        "dog",
        "cat",
        "bird",
        "fish",
        "tench",
        "springer",
        "shark",
        "hen",
        "cock",
        "ostrich",
        "snake",
        "lizard",
        "frog",
        "monkey",
        "bear",
        "lion",
        "tiger",
        "wolf",
        "fox",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "penguin",
        "duck",
        "eagle",
        "ray",
        "stingray",
    ),
    "vehicle": (
        "car",
        "truck",
        "bus",
        "bicycle",
        "bike",
        "motorcycle",
        "ship",
        "boat",
        "airplane",
        "plane",
        "train",
        "locomotive",
        "ambulance",
        "taxi",
        "jeep",
        "van",
        "tractor",
        "submarine",
        "scooter",
    ),
    "food": (
        "pizza",
        "hotdog",
        "banana",
        "apple",
        "orange",
        "sandwich",
        "burger",
        "cake",
        "bread",
        "ice cream",
        "icecream",
        "dish",
        "plate",
    ),
    "instrument": (
        "guitar",
        "piano",
        "violin",
        "trumpet",
        "drum",
        "flute",
        "sax",
        "saxophone",
        "harp",
        "accordion",
        "cello",
    ),
}


@dataclass(slots=True)
class SampleRecord:
    """Single dataset sample consumed by Stage 1."""

    image_path: str
    class_id: int
    class_name: str
    class_name_raw: str
    archetype: str
    slot_schema: list[str]
    sample_id: str | None = None


def infer_archetype(class_name: str) -> str:
    """Infer a semantic archetype from a readable class name.

    This is intentionally heuristic and class-level, not image-level. The goal
    is to keep slot schemas stable within a class so downstream aggregation does
    not become a mess.
    """
    lowered = class_name.lower()
    for archetype, keywords in ARCHETYPE_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return archetype
    return DEFAULT_ARCHETYPE


def get_slot_schema(archetype: str) -> list[str]:
    """Return the slot schema associated with an archetype."""
    if archetype not in ARCHETYPE_SLOT_SCHEMAS:
        raise KeyError(f"Unknown archetype: {archetype}")
    return ARCHETYPE_SLOT_SCHEMAS[archetype]


def normalize_attributes(payload: dict[str, Any], expected_fields: list[str]) -> dict[str, str | None]:
    """Normalize a raw attribute dictionary against the expected slot schema."""
    normalized: dict[str, str | None] = {}
    for field in expected_fields:
        normalized[field] = _normalize_value(payload.get(field, "unknown"))
    return normalized


def validate_attribute_payload(payload: dict[str, Any], expected_fields: list[str]) -> tuple[bool, list[str]]:
    """Validate a VLM response against the current sample's slot schema.

    The VLM may return either:
    - a flat object with the slot fields directly at top level, or
    - {"archetype": ..., "attributes": {...}}

    We accept both to keep the parser tolerant while migrating prompts.
    """
    errors: list[str] = []
    attribute_payload = payload.get("attributes", payload)
    if not isinstance(attribute_payload, dict):
        return False, ["Payload 'attributes' must be a JSON object"]

    for field in expected_fields:
        if field not in attribute_payload:
            errors.append(f"Missing attribute field: {field}")
            continue
        value = attribute_payload[field]
        if value in ALLOWED_SPECIAL_VALUES:
            continue
        if not isinstance(value, str):
            errors.append(f"Field {field} must be str | null | special token, got {type(value).__name__}")
            continue
        if len(value.strip()) == 0:
            errors.append(f"Field {field} is empty")
    return len(errors) == 0, errors


def extract_attribute_mapping(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the actual attribute mapping from a parsed VLM JSON object."""
    attribute_payload = payload.get("attributes", payload)
    if not isinstance(attribute_payload, dict):
        raise ValueError("Payload 'attributes' must be a JSON object")
    return attribute_payload


def _normalize_value(value: Any) -> str | None:
    if value in ALLOWED_SPECIAL_VALUES:
        return value
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = " ".join(value.strip().split())
        return cleaned if cleaned else "unknown"
    return str(value)
