from __future__ import annotations

"""Core data schema for CSPD Stage 1.

This module defines:
- the unified attribute field list used by the first implementation,
- the input sample record consumed by the pipeline, and
- validation / normalization helpers for VLM outputs.

The design goal is to keep Stage 1 rigid enough for downstream code to rely on,
while still allowing uncertain fields to fall back to `unknown` or `not_applicable`.
"""

from dataclasses import asdict, dataclass
from typing import Any

# Special values that are explicitly allowed in Stage 1 outputs.
# They let us distinguish between "the model could not tell" and
# "this slot does not make sense for this image/category".
ALLOWED_SPECIAL_VALUES = {"unknown", "not_applicable", None}

# Unified attribute schema for the first coding iteration.
# We intentionally keep it global rather than class-adaptive so the
# first pipeline is simple, stable, and easy to debug.
ATTRIBUTE_FIELDS = [
    "subject",
    "color",
    "shape_or_body_trait",
    "action_or_pose_or_state",
    "background_or_context",
    "viewpoint",
    "material",
]


@dataclass(slots=True)
class SampleRecord:
    """Single dataset sample consumed by Stage 1.

    Attributes:
        image_path: Path to the source image.
        class_id: Integer class index used by the dataset.
        class_name: Human-readable category name.
        sample_id: Optional stable identifier if the dataset already has one.
    """

    image_path: str
    class_id: int
    class_name: str
    sample_id: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SampleRecord":
        """Build a validated sample record from raw JSON input.

        We fail early if required metadata is missing, because downstream VLM
        prompting assumes these fields exist.
        """
        missing = [key for key in ("image_path", "class_id", "class_name") if key not in payload]
        if missing:
            raise ValueError(f"Missing required sample fields: {missing}")
        return cls(
            image_path=str(payload["image_path"]),
            class_id=int(payload["class_id"]),
            class_name=str(payload["class_name"]),
            sample_id=str(payload["sample_id"]) if payload.get("sample_id") is not None else None,
        )


@dataclass(slots=True)
class AttributeRecord:
    """Normalized attribute payload produced by Stage 1.

    Every field is present so later stages do not need to guess whether a slot
    was omitted accidentally or intentionally left unknown.
    """

    subject: str | None = "unknown"
    color: str | None = "unknown"
    shape_or_body_trait: str | None = "unknown"
    action_or_pose_or_state: str | None = "unknown"
    background_or_context: str | None = "unknown"
    viewpoint: str | None = "unknown"
    material: str | None = "unknown"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AttributeRecord":
        """Normalize a raw attribute dictionary into the frozen Stage 1 schema."""
        normalized: dict[str, Any] = {}
        for field in ATTRIBUTE_FIELDS:
            value = payload.get(field, "unknown")
            normalized[field] = _normalize_value(value)
        return cls(**normalized)

    def to_dict(self) -> dict[str, Any]:
        """Convert the dataclass back to a plain dict for JSON serialization."""
        return asdict(self)


def _normalize_value(value: Any) -> str | None:
    """Normalize one slot value into a compact JSON-friendly representation.

    Rules:
    - preserve the accepted special tokens,
    - collapse extra whitespace in strings,
    - turn empty strings into `unknown`,
    - stringify any unexpected primitive rather than crashing immediately.
    """
    if value in ALLOWED_SPECIAL_VALUES:
        return value
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = " ".join(value.strip().split())
        return cleaned if cleaned else "unknown"
    return str(value)


def validate_attribute_payload(payload: dict[str, Any]) -> tuple[bool, list[str]]:
    """Check whether a VLM response matches the required Stage 1 schema.

    Validation is intentionally lightweight:
    - every required field must exist,
    - each field must be either a string, null, or one of the special tokens,
    - empty strings are rejected.

    More semantic validation can be added later, but this is enough to keep the
    first pipeline implementation from silently accepting malformed outputs.
    """
    errors: list[str] = []
    for field in ATTRIBUTE_FIELDS:
        if field not in payload:
            errors.append(f"Missing attribute field: {field}")
            continue
        value = payload[field]
        if value in ALLOWED_SPECIAL_VALUES:
            continue
        if not isinstance(value, str):
            errors.append(f"Field {field} must be str | null | special token, got {type(value).__name__}")
            continue
        if len(value.strip()) == 0:
            errors.append(f"Field {field} is empty")
    return len(errors) == 0, errors
