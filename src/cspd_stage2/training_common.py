from __future__ import annotations

"""Shared Stage 2 helpers kept after the SDXL-only cleanup (2026-04-18).

Only path-derivation utilities and a safe-JSON write wrapper remain here.
All adapter-injection / module-targeting / optimizer-building helpers were
removed along with the self-built FLUX/PixArt training loop.
"""

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from cspd_stage1.io_utils import write_json


SPLIT_ONLY_DATASET_ROOT_NAMES = {"train", "val", "valid", "validation", "test", "testing"}


def derive_stage2_dataset_label(dataset_root: str | os.PathLike[str]) -> str:
    dataset_path = Path(dataset_root).expanduser().resolve()
    base_name = dataset_path.name
    parent_name = dataset_path.parent.name
    if base_name.lower() in SPLIT_ONLY_DATASET_ROOT_NAMES and parent_name:
        return f"{parent_name}_{base_name}"
    return base_name


def sanitize_stage2_backbone_slug(backbone_name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]", "_", backbone_name.strip().replace("/", "__").replace(" ", "__"))
    slug = re.sub(r"_+", "_", slug)
    return slug.strip("._-") or "backbone"


def derive_stage2_output_dir(dataset_root: str | os.PathLike[str], backbone_name: str, *, timestamp: str | None = None) -> str:
    resolved_timestamp = timestamp or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return str(
        Path("runs") / "stage2" / "train"
        / derive_stage2_dataset_label(dataset_root)
        / sanitize_stage2_backbone_slug(backbone_name)
        / resolved_timestamp
    )


def _safe_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    try:
        write_json(path, payload)
    except Exception:
        pass
