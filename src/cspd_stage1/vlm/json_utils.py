from __future__ import annotations

"""Helpers for extracting structured JSON from VLM text outputs.

Real VLMs often produce nearly-correct JSON wrapped in markdown fences or padded
with harmless formatting junk. Stage 1 should be robust to that noise without
silently accepting completely malformed outputs.
"""

import json
import re
from typing import Any


def clean_json_text(raw_text: str) -> str:
    """Remove the most common formatting wrappers around a JSON object."""
    cleaned_text = raw_text.strip()

    if cleaned_text.startswith("```json"):
        cleaned_text = cleaned_text[len("```json"):].strip()
    elif cleaned_text.startswith("```"):
        cleaned_text = cleaned_text[len("```"):].strip()

    if cleaned_text.endswith("```"):
        cleaned_text = cleaned_text[:-3].strip()

    return cleaned_text


def parse_json_object(raw_text: str) -> dict[str, Any]:
    """Parse a JSON object from VLM output text.

    First try strict JSON parsing. If that yields a top-level JSON list, accept
    the first dictionary item as the payload. This makes Stage 1 tolerant to
    outputs like `[ { ... } ]` or `[ { ... }, { ... } ]`.

    If strict parsing fails, try to salvage the first complete JSON object from
    a truncated response such as `[ { ... }, { ...` where the model already
    emitted one valid candidate before being cut off.

    If that also fails, apply a conservative fallback parser for the
    pseudo-JSON pattern we have observed from the VLM.
    """
    cleaned_text = clean_json_text(raw_text)
    try:
        parsed = json.loads(cleaned_text)
        parsed = _coerce_top_level_list_to_object(parsed)
    except json.JSONDecodeError:
        salvaged_object_text = _extract_first_complete_json_object(cleaned_text)
        if salvaged_object_text is not None:
            parsed = json.loads(salvaged_object_text)
            parsed = _coerce_top_level_list_to_object(parsed)
        else:
            parsed = _parse_bullet_style_attributes(cleaned_text)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a JSON object, got {type(parsed).__name__}")
    return parsed


def _coerce_top_level_list_to_object(parsed: Any) -> Any:
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                return item
        raise ValueError("Expected a JSON object or a list containing at least one object")
    return parsed


def _extract_first_complete_json_object(text: str) -> str | None:
    """Extract the first complete top-level JSON object from noisy/truncated text.

    This is used for cases where the model emits a valid first object and then
    keeps generating additional candidates before getting truncated, e.g.:
    `[ { ... }, { ...`.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for index in range(start, len(text)):
        char = text[index]

        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]

    return None


def _parse_bullet_style_attributes(text: str) -> dict[str, Any]:
    """Fallback parser for a narrow pseudo-JSON pattern.

    Example supported input:
        {
          "archetype": "animal",
          "attributes": {
            - species_or_category: tench,
            - color: yellow-orange,
            ...
          }
        }

    This parser is intentionally narrow so we do not silently over-correct truly
    broken outputs into something misleading.
    """
    archetype_match = re.search(r'"archetype"\s*:\s*"([^"]+)"', text)
    if archetype_match is None:
        raise ValueError("Fallback parse failed: missing quoted archetype field")

    attributes_block_match = re.search(r'"attributes"\s*:\s*\{(.*?)\}', text, flags=re.DOTALL)
    if attributes_block_match is None:
        raise ValueError("Fallback parse failed: missing attributes block")

    attributes_block = attributes_block_match.group(1)
    attributes: dict[str, str] = {}
    for line in attributes_block.splitlines():
        stripped = line.strip().rstrip(",")
        if not stripped:
            continue
        bullet_match = re.match(r'^-\s*([A-Za-z0-9_]+)\s*:\s*(.+)$', stripped)
        if bullet_match is None:
            raise ValueError(f"Fallback parse failed: unsupported attribute line: {stripped}")
        key = bullet_match.group(1)
        value = bullet_match.group(2).strip()
        value = value.strip('"')
        attributes[key] = value

    return {
        "archetype": archetype_match.group(1),
        "attributes": attributes,
    }
