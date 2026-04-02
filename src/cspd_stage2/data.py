from __future__ import annotations

"""Data loading and pairing utilities for CSPD Stage 2.

Stage 2 now means diffusion adaptation / canonical-semantic-space familiarization.
This module implements the parts we can make real without claiming a complete
FLUX Kontext fine-tuning stack:
- scan an ImageFolder-style visual dataset,
- load Stage 1 render records,
- pair images to canonical captions conservatively,
- emit a training-ready manifest and summary,
- provide a lightweight dataset surface for future trainers.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from cspd_stage1.io_utils import write_json, write_jsonl
from cspd_stage1.pipeline import IMAGE_EXTENSIONS, build_samples_from_imagefolder, load_string_mapping


@dataclass(slots=True)
class Stage2PairRecord:
    """Joined Stage 2 training record built from a real image and Stage 1 text."""

    pair_id: str
    record_id: str
    sample_id: str
    relative_image_path: str
    image_path: str
    class_id: int
    class_name_raw: str
    class_name: str
    archetype: str
    canonical_caption: str
    render_source_record_id: str
    render_input_path: str
    matched_via: str
    image_width: int | None = None
    image_height: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "record_id": self.record_id,
            "sample_id": self.sample_id,
            "relative_image_path": self.relative_image_path,
            "image_path": self.image_path,
            "class_id": self.class_id,
            "class_name_raw": self.class_name_raw,
            "class_name": self.class_name,
            "archetype": self.archetype,
            "canonical_caption": self.canonical_caption,
            "render_source_record_id": self.render_source_record_id,
            "render_input_path": self.render_input_path,
            "matched_via": self.matched_via,
            "image_width": self.image_width,
            "image_height": self.image_height,
        }


@dataclass(slots=True)
class PairingResult:
    pairs: list[Stage2PairRecord]
    summary: dict[str, Any]
    unmatched_images: list[dict[str, Any]]
    unmatched_render_records: list[dict[str, Any]]


@dataclass(slots=True)
class ManifestPaths:
    manifest_path: str
    summary_path: str
    unmatched_images_path: str
    unmatched_render_records_path: str


class Stage2PairedDataset:
    """Minimal dataset wrapper for downstream Stage 2 trainers.

    This intentionally returns plain Python dicts and PIL images rather than
    pretending a specific FLUX Kontext pipeline is wired up in this repo.
    """

    def __init__(self, pairs: Iterable[Stage2PairRecord]):
        self._pairs = list(pairs)

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        pair = self._pairs[index]
        image = Image.open(pair.image_path).convert("RGB")
        return {
            "pair_id": pair.pair_id,
            "record_id": pair.record_id,
            "sample_id": pair.sample_id,
            "image": image,
            "image_path": pair.image_path,
            "caption": pair.canonical_caption,
            "class_name": pair.class_name,
            "class_name_raw": pair.class_name_raw,
            "archetype": pair.archetype,
        }


def build_stage2_pairs(
    *,
    dataset_root: str,
    render_input: str,
    class_name_map: str | None = None,
    class_archetype_map: str | None = None,
    verify_images: bool = False,
    strict: bool = False,
) -> PairingResult:
    """Pair ImageFolder samples with Stage 1 render records.

    Pairing priority is intentionally conservative:
    1. exact `record_id`
    2. exact `sample_id`
    3. exact normalized relative image path

    This follows the user's requirement to rely on stable identifiers from
    Stage 1 artifacts when available, rather than inventing brittle heuristics.
    """

    class_name_mapping = load_string_mapping(class_name_map, "class-name map")
    class_archetype_mapping = load_string_mapping(class_archetype_map, "class-archetype map")
    samples = build_samples_from_imagefolder(dataset_root, class_name_mapping, class_archetype_mapping)
    render_rows = _read_jsonl(render_input)

    render_by_record_id: dict[str, dict[str, Any]] = {}
    render_by_sample_id: dict[str, dict[str, Any]] = {}
    render_by_relative_path: dict[str, dict[str, Any]] = {}
    render_row_keys: dict[int, str] = {}

    for index, row in enumerate(render_rows):
        if not isinstance(row, dict):
            continue
        record_id = _string_or_none(row.get("record_id"))
        sample_id = _normalize_relpath(_string_or_none(row.get("sample_id")))
        rel_image = _normalize_relpath(_string_or_none(row.get("relative_image_path")))
        render_status = _string_or_none(row.get("render_status"))
        canonical_caption = _string_or_none(row.get("canonical_caption"))
        if render_status not in {None, "success"}:
            continue
        if not canonical_caption:
            continue
        key = record_id or sample_id or rel_image or f"render_row_{index}"
        render_row_keys[index] = key
        if record_id and record_id not in render_by_record_id:
            render_by_record_id[record_id] = row
        if sample_id and sample_id not in render_by_sample_id:
            render_by_sample_id[sample_id] = row
        if rel_image and rel_image not in render_by_relative_path:
            render_by_relative_path[rel_image] = row

    used_keys: set[str] = set()
    pairs: list[Stage2PairRecord] = []
    unmatched_images: list[dict[str, Any]] = []

    for sample in samples:
        record_id = f"{sample.class_name_raw}::{sample.sample_id or Path(sample.image_path).name}"
        sample_id = _normalize_relpath(sample.sample_id)
        relative_image_path = sample_id or _normalize_relpath(Path(sample.image_path).name)

        matched_row = None
        matched_via = None
        if record_id and record_id in render_by_record_id:
            matched_row = render_by_record_id[record_id]
            matched_via = "record_id"
        elif sample_id and sample_id in render_by_sample_id:
            matched_row = render_by_sample_id[sample_id]
            matched_via = "sample_id"
        elif relative_image_path and relative_image_path in render_by_relative_path:
            matched_row = render_by_relative_path[relative_image_path]
            matched_via = "relative_image_path"

        if matched_row is None:
            unmatched_images.append(
                {
                    "record_id": record_id,
                    "sample_id": sample.sample_id,
                    "relative_image_path": relative_image_path,
                    "image_path": sample.image_path,
                    "class_name_raw": sample.class_name_raw,
                    "class_name": sample.class_name,
                    "archetype": sample.archetype,
                    "reason": "no_matching_render_record",
                }
            )
            continue

        matched_key = (
            _string_or_none(matched_row.get("record_id"))
            or _normalize_relpath(_string_or_none(matched_row.get("sample_id")))
            or _normalize_relpath(_string_or_none(matched_row.get("relative_image_path")))
        )
        if matched_key:
            used_keys.add(matched_key)

        width, height = (None, None)
        if verify_images:
            width, height = _probe_image_size(sample.image_path)

        pairs.append(
            Stage2PairRecord(
                pair_id=record_id,
                record_id=record_id,
                sample_id=sample.sample_id or relative_image_path,
                relative_image_path=relative_image_path,
                image_path=sample.image_path,
                class_id=sample.class_id,
                class_name_raw=sample.class_name_raw,
                class_name=sample.class_name,
                archetype=sample.archetype,
                canonical_caption=str(matched_row["canonical_caption"]),
                render_source_record_id=str(matched_row.get("record_id") or record_id),
                render_input_path=str(Path(render_input).resolve()),
                matched_via=str(matched_via or "unknown"),
                image_width=width,
                image_height=height,
            )
        )

    unmatched_render_records: list[dict[str, Any]] = []
    for row in render_rows:
        if not isinstance(row, dict):
            continue
        key = (
            _string_or_none(row.get("record_id"))
            or _normalize_relpath(_string_or_none(row.get("sample_id")))
            or _normalize_relpath(_string_or_none(row.get("relative_image_path")))
        )
        canonical_caption = _string_or_none(row.get("canonical_caption"))
        render_status = _string_or_none(row.get("render_status"))
        if render_status not in {None, "success"} or not canonical_caption:
            continue
        if key and key not in used_keys:
            unmatched_render_records.append(
                {
                    "record_id": row.get("record_id"),
                    "sample_id": row.get("sample_id"),
                    "relative_image_path": row.get("relative_image_path"),
                    "canonical_caption": canonical_caption,
                    "archetype": row.get("archetype"),
                    "class_name": row.get("class_name"),
                    "reason": "no_matching_image",
                }
            )

    summary = _build_pairing_summary(
        dataset_root=dataset_root,
        render_input=render_input,
        samples=samples,
        pairs=pairs,
        unmatched_images=unmatched_images,
        unmatched_render_records=unmatched_render_records,
        strict=strict,
        verify_images=verify_images,
    )

    if strict and (unmatched_images or unmatched_render_records):
        raise ValueError(
            "Strict Stage 2 pairing failed: unmatched images or render records remain. "
            f"unmatched_images={len(unmatched_images)}, unmatched_render_records={len(unmatched_render_records)}"
        )

    return PairingResult(
        pairs=pairs,
        summary=summary,
        unmatched_images=unmatched_images,
        unmatched_render_records=unmatched_render_records,
    )


def write_pairing_artifacts(result: PairingResult, output_dir: str | Path) -> ManifestPaths:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "train_manifest.jsonl"
    summary_path = output_dir / "train_manifest_summary.json"
    unmatched_images_path = output_dir / "unmatched_images.jsonl"
    unmatched_render_records_path = output_dir / "unmatched_render_records.jsonl"

    write_jsonl(manifest_path, [_pair_to_manifest_row(pair) for pair in result.pairs])
    write_json(summary_path, result.summary)
    write_jsonl(unmatched_images_path, result.unmatched_images)
    write_jsonl(unmatched_render_records_path, result.unmatched_render_records)
    return ManifestPaths(
        manifest_path=str(manifest_path.resolve()),
        summary_path=str(summary_path.resolve()),
        unmatched_images_path=str(unmatched_images_path.resolve()),
        unmatched_render_records_path=str(unmatched_render_records_path.resolve()),
    )


def _pair_to_manifest_row(pair: Stage2PairRecord) -> dict[str, Any]:
    row = pair.to_dict()
    row["text"] = pair.canonical_caption
    row["image"] = pair.image_path
    return row


def _build_pairing_summary(
    *,
    dataset_root: str,
    render_input: str,
    samples: list[Any],
    pairs: list[Stage2PairRecord],
    unmatched_images: list[dict[str, Any]],
    unmatched_render_records: list[dict[str, Any]],
    strict: bool,
    verify_images: bool,
) -> dict[str, Any]:
    counts_by_class: dict[str, int] = {}
    counts_by_archetype: dict[str, int] = {}
    for pair in pairs:
        counts_by_class[pair.class_name_raw] = counts_by_class.get(pair.class_name_raw, 0) + 1
        counts_by_archetype[pair.archetype] = counts_by_archetype.get(pair.archetype, 0) + 1

    return {
        "dataset_root": str(Path(dataset_root).resolve()),
        "render_input": str(Path(render_input).resolve()),
        "num_imagefolder_samples": len(samples),
        "num_pairs": len(pairs),
        "num_unmatched_images": len(unmatched_images),
        "num_unmatched_render_records": len(unmatched_render_records),
        "pairing_success_rate": (len(pairs) / len(samples)) if samples else 0.0,
        "counts_by_class": counts_by_class,
        "counts_by_archetype": counts_by_archetype,
        "strict": strict,
        "verify_images": verify_images,
    }


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            rows.append(payload)
    return rows


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_relpath(value: str | None) -> str | None:
    if not value:
        return None
    return value.replace('\\', '/').lstrip('./')


def _probe_image_size(path: str | Path) -> tuple[int | None, int | None]:
    with Image.open(path) as image:
        width, height = image.size
    return int(width), int(height)
