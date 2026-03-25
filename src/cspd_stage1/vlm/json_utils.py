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

    First try strict JSON parsing. If that fails, apply a very conservative
    fallback for the specific pseudo-JSON pattern we have observed from the VLM:
    bullet-style lines such as `- key: value` inside the `attributes` block.
    """
    cleaned_text = clean_json_text(raw_text)
    try:
        parsed = json.loads(cleaned_text)
    except json.JSONDecodeError:
        parsed = _parse_bullet_style_attributes(cleaned_text)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a JSON object, got {type(parsed).__name__}")
    return parsed


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
