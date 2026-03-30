from __future__ import annotations

"""Core schema utilities for CSPD Stage 1.

Stage 1 uses class-adaptive slot schemas:
- map each class to a fixed semantic archetype,
- choose the slot family associated with that archetype,
- ask the VLM to fill only those slots.
"""

from dataclasses import dataclass
from typing import Any

ALLOWED_SPECIAL_VALUES = {"unknown", "not_applicable", None}

ARCHETYPE_SLOT_SCHEMAS: dict[str, list[str]] = {
    "animal": [
        "species_or_category",
        "color_or_pattern",
        "body_trait",
        "pose_or_state",
        "background_or_habitat",
        "viewpoint",
        "salient_part_or_focus",
    ],
    "plant_or_fungus": [
        "plant_or_fungus_type",
        "color",
        "shape_or_growth_form",
        "visible_part",
        "growth_state",
        "background_or_habitat",
        "viewpoint",
    ],
    "food_and_drink": [
        "food_or_drink_type",
        "color",
        "shape_or_structure",
        "preparation_or_serving_style",
        "container_or_context",
        "viewpoint",
        "salient_topping_or_ingredient",
    ],
    "vehicle": [
        "vehicle_type",
        "color",
        "shape_or_structure",
        "state_or_action",
        "environment",
        "viewpoint",
        "salient_part_or_accessory",
    ],
    "clothing_and_wearable": [
        "wearable_type",
        "color_or_pattern",
        "material_or_texture",
        "shape_or_style",
        "wearing_state_or_pose",
        "background_or_context",
        "viewpoint",
    ],
    "furniture": [
        "furniture_type",
        "color",
        "material",
        "shape_or_structure",
        "usage_state",
        "background_or_room_context",
        "viewpoint",
    ],
    "container": [
        "container_type",
        "color",
        "material",
        "shape_or_structure",
        "fill_state_or_contents_visibility",
        "background_or_context",
        "viewpoint",
    ],
    "tool": [
        "tool_type",
        "color",
        "material",
        "shape_or_structure",
        "usage_state",
        "background_or_context",
        "viewpoint",
    ],
    "device_or_appliance": [
        "device_or_appliance_type",
        "color",
        "material_or_finish",
        "shape_or_structure",
        "operating_state_or_display_state",
        "background_or_context",
        "viewpoint",
    ],
    "instrument": [
        "instrument_type",
        "color",
        "material",
        "shape_or_structure",
        "playing_state_or_pose",
        "background_or_context",
        "viewpoint",
    ],
    "weapon": [
        "weapon_type",
        "color",
        "material",
        "shape_or_structure",
        "usage_or_display_state",
        "background_or_context",
        "viewpoint",
    ],
    "sports_or_toy": [
        "sports_or_toy_type",
        "color_or_pattern",
        "material",
        "shape_or_structure",
        "activity_or_usage_state",
        "background_or_context",
        "viewpoint",
    ],
    "household_object": [
        "household_object_type",
        "color",
        "material",
        "shape_or_structure",
        "usage_state",
        "background_or_room_context",
        "viewpoint",
    ],
    "structure_or_building": [
        "structure_or_building_type",
        "material_or_surface",
        "architectural_style_or_form",
        "scale_or_extent",
        "surrounding_environment",
        "viewpoint",
        "salient_structural_part",
    ],
    "natural_scene_or_landform": [
        "scene_or_landform_type",
        "dominant_color_or_tone",
        "terrain_or_surface_trait",
        "weather_or_water_state",
        "vegetation_or_natural_context",
        "viewpoint",
        "salient_geographic_feature",
    ],
    "human_or_person": [
        "person_type_or_role",
        "clothing_or_gear",
        "body_pose_or_action",
        "visible_body_trait",
        "background_or_activity_context",
        "viewpoint",
        "held_object_or_equipment",
    ],
    "text_or_media_object": [
        "text_or_media_object_type",
        "dominant_color",
        "layout_or_format",
        "content_or_symbol_type",
        "physical_or_display_state",
        "background_or_context",
        "viewpoint",
    ],
    "decorative_or_symbolic_object": [
        "decorative_or_symbolic_object_type",
        "color_or_pattern",
        "material",
        "ornamentation_or_symbolic_trait",
        "display_or_usage_context",
        "background_or_context",
        "viewpoint",
    ],
}

DEFAULT_ARCHETYPE = "household_object"

ARCHETYPE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "animal": (
        "dog", "cat", "bird", "fish", "shark", "ray", "hen", "cock", "ostrich", "snake", "lizard",
        "frog", "monkey", "bear", "lion", "tiger", "wolf", "fox", "horse", "sheep", "cow",
        "elephant", "penguin", "duck", "eagle", "whale", "jellyfish", "crab", "lobster", "spider",
    ),
    "plant_or_fungus": (
        "daisy", "corn", "acorn", "rapeseed", "fungus", "mushroom", "bolete", "earthstar", "flower",
        "tree", "leaf", "plant", "cactus",
    ),
    "food_and_drink": (
        "pizza", "hotdog", "banana", "apple", "orange", "burger", "cake", "bread", "ice cream",
        "icecream", "espresso", "wine", "eggnog", "potpie", "burrito", "guacamole",
    ),
    "vehicle": (
        "car", "truck", "bus", "bicycle", "bike", "motorcycle", "ship", "boat", "airplane", "plane",
        "train", "locomotive", "ambulance", "taxi", "jeep", "van", "tractor", "submarine", "scooter",
        "airliner", "warplane", "tank", "cart", "canoe", "cab",
    ),
    "instrument": (
        "guitar", "piano", "violin", "trumpet", "drum", "flute", "sax", "saxophone", "harp",
        "accordion", "cello", "banjo", "oboe", "trombone", "marimba", "harmonica", "ocarina",
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
    """Heuristic fallback for class-level archetype assignment.

    This remains as a weak fallback only. The preferred path is to pass an
    explicit fixed class->archetype mapping file.
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


def complete_missing_attributes(payload: dict[str, Any], expected_fields: list[str]) -> dict[str, Any]:
    """Return a payload where missing expected attribute fields are filled with ``unknown``.

    Real VLM outputs are often almost correct but omit one or two requested slots.
    Since the prompt explicitly instructs the model to use ``unknown`` when a slot
    is unclear, an omitted field is usually better interpreted as an implicit
    ``unknown`` than treated as a hard extraction failure.
    """
    attribute_payload = payload.get("attributes", payload)
    if not isinstance(attribute_payload, dict):
        return payload

    completed_attributes = dict(attribute_payload)
    for field in expected_fields:
        completed_attributes.setdefault(field, "unknown")

    completed_payload = dict(payload)
    completed_payload["attributes"] = completed_attributes
    return completed_payload


def validate_attribute_payload(payload: dict[str, Any], expected_fields: list[str]) -> tuple[bool, list[str]]:
    """Validate a VLM response against the current sample's slot schema."""
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
