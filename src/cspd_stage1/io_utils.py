from __future__ import annotations

"""Lightweight file IO helpers for Stage 1 artifacts.

We keep these utilities tiny and dependency-free because they are used by both
CLI execution and future unit tests.
"""

import json
from pathlib import Path
from typing import Iterable


def read_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file into memory.

    We use `utf-8-sig` instead of plain `utf-8` because PowerShell and some
    Windows editors like to sneak in a BOM. That encoding choice avoids a very
    annoying class of avoidable parse failures.
    """
    records: list[dict] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return records


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    """Write rows to JSONL, one JSON object per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, payload: dict) -> None:
    """Write a pretty-printed JSON file for human inspection and debugging."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
