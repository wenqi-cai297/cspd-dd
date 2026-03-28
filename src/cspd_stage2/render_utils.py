from __future__ import annotations

from typing import Any

UNKNOWN_VALUES = {"unknown", "not_applicable", "n/a", "none", "null", ""}


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
