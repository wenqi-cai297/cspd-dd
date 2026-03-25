from __future__ import annotations

"""Helpers for extracting structured JSON from VLM text outputs.

Real VLMs often produce nearly-correct JSON wrapped in markdown fences or padded
with harmless formatting junk. Stage 1 should be robust to that noise without
silently accepting completely malformed outputs.
"""

import json
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

    We require the final parsed value to be a dictionary because Stage 1 expects
    a fixed attribute mapping, not a list or scalar.
    """
    cleaned_text = clean_json_text(raw_text)
    parsed = json.loads(cleaned_text)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a JSON object, got {type(parsed).__name__}")
    return parsed
