from __future__ import annotations

"""Data loading and pairing utilities for CSPD Stage 2.

Stage 2 now means generative-backbone adaptation / canonical-semantic-space familiarization.
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

import numpy as np
from PIL import Image

from cspd_stage1.io_utils import write_json, write_jsonl
from cspd_stage1.pipeline import build_samples_from_imagefolder, load_string_mapping


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
    dataset_archetype: str | None = None
    render_archetype: str | None = None
    caption_template_family: str | None = None
    caption_template_id: str | None = None
    caption_anchor_slot: str | None = None
    caption_slot_count: int | None = None
    caption_source_stage: str = "stage1_render"
    conditioning_text_source: str = "canonical_caption"
    render_source_record_id: str = ""
    render_input_path: str = ""
    matched_via: str = ""
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
            "dataset_archetype": self.dataset_archetype,
            "render_archetype": self.render_archetype,
            "canonical_caption": self.canonical_caption,
            "caption_template_family": self.caption_template_family,
            "caption_template_id": self.caption_template_id,
            "caption_anchor_slot": self.caption_anchor_slot,
            "caption_slot_count": self.caption_slot_count,
            "caption_source_stage": self.caption_source_stage,
            "conditioning_text_source": self.conditioning_text_source,
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

    This returns real image/caption pairs from Stage 1 render outputs.
    When a resolution is provided, images are resized/cropped into a tensor
    suitable for a real Stage 2 training loop.
    """

    def __init__(self, pairs: Iterable[Stage2PairRecord], *, resolution: int | None = None):
        self._pairs = list(pairs)
        self._resolution = resolution

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        pair = self._pairs[index]
        image = Image.open(pair.image_path).convert("RGB")
        item = {
            "pair_id": pair.pair_id,
            "record_id": pair.record_id,
            "sample_id": pair.sample_id,
            "image": image,
            "image_path": pair.image_path,
            "caption": pair.canonical_caption,
            "conditioning_text": pair.canonical_caption,
            "conditioning_text_source": pair.conditioning_text_source,
            "caption_template_family": pair.caption_template_family,
            "caption_template_id": pair.caption_template_id,
            "caption_anchor_slot": pair.caption_anchor_slot,
            "caption_slot_count": pair.caption_slot_count,
            "class_name": pair.class_name,
            "class_name_raw": pair.class_name_raw,
            "archetype": pair.archetype,
        }
        if self._resolution is not None:
            item["pixel_values"] = pil_to_normalized_tensor(image, self._resolution)
        return item


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

        renderer = matched_row.get("renderer") if isinstance(matched_row.get("renderer"), dict) else {}
        verbalized_slots = matched_row.get("verbalized_slots")
        caption_slot_count = len(verbalized_slots) if isinstance(verbalized_slots, list) else None
        render_archetype = _string_or_none(matched_row.get("archetype"))
        template_family = _string_or_none(renderer.get("template_family"))
        effective_archetype = render_archetype or template_family or sample.archetype

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
                archetype=effective_archetype,
                dataset_archetype=sample.archetype,
                render_archetype=render_archetype,
                canonical_caption=str(matched_row["canonical_caption"]),
                caption_template_family=template_family,
                caption_template_id=_string_or_none(renderer.get("template_id")),
                caption_anchor_slot=_string_or_none(matched_row.get("anchor_slot")),
                caption_slot_count=caption_slot_count,
                caption_source_stage="stage1_render",
                conditioning_text_source="canonical_caption",
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
    row["conditioning_text"] = pair.canonical_caption
    row["image"] = pair.image_path
    row["conditioning_target"] = "text_conditioning"
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
    counts_by_template_family: dict[str, int] = {}
    counts_by_match_strategy: dict[str, int] = {}
    counts_by_dataset_archetype: dict[str, int] = {}
    counts_by_render_archetype: dict[str, int] = {}
    num_pairs_with_dataset_render_archetype_mismatch = 0
    for pair in pairs:
        counts_by_class[pair.class_name_raw] = counts_by_class.get(pair.class_name_raw, 0) + 1
        counts_by_archetype[pair.archetype] = counts_by_archetype.get(pair.archetype, 0) + 1
        template_family = pair.caption_template_family or "unknown"
        counts_by_template_family[template_family] = counts_by_template_family.get(template_family, 0) + 1
        counts_by_match_strategy[pair.matched_via] = counts_by_match_strategy.get(pair.matched_via, 0) + 1

        dataset_archetype = pair.dataset_archetype or "unknown"
        render_archetype = pair.render_archetype or template_family
        counts_by_dataset_archetype[dataset_archetype] = counts_by_dataset_archetype.get(dataset_archetype, 0) + 1
        counts_by_render_archetype[render_archetype] = counts_by_render_archetype.get(render_archetype, 0) + 1
        if pair.dataset_archetype and render_archetype and pair.dataset_archetype != render_archetype:
            num_pairs_with_dataset_render_archetype_mismatch += 1

    return {
        "dataset_root": str(Path(dataset_root).resolve()),
        "render_input": str(Path(render_input).resolve()),
        "conditioning_text_source": "stage1_render.canonical_caption",
        "num_imagefolder_samples": len(samples),
        "num_pairs": len(pairs),
        "num_unmatched_images": len(unmatched_images),
        "num_unmatched_render_records": len(unmatched_render_records),
        "pairing_success_rate": (len(pairs) / len(samples)) if samples else 0.0,
        "counts_by_class": counts_by_class,
        "counts_by_archetype": counts_by_archetype,
        "counts_by_template_family": counts_by_template_family,
        "counts_by_dataset_archetype": counts_by_dataset_archetype,
        "counts_by_render_archetype": counts_by_render_archetype,
        "archetype_count_source": "matched_stage1_render_record",
        "dataset_archetype_source": "imagefolder_sample_or_class_archetype_map",
        "render_archetype_source": "matched_stage1_render_record.archetype_or_template_family",
        "num_pairs_with_dataset_render_archetype_mismatch": num_pairs_with_dataset_render_archetype_mismatch,
        "counts_by_match_strategy": counts_by_match_strategy,
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


def make_stage2_dataloader(
    pairs: Iterable[Stage2PairRecord],
    *,
    resolution: int,
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool = True,
) -> Any:
    import torch
    from torch.utils.data import DataLoader

    dataset = Stage2PairedDataset(pairs, resolution=resolution)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate_stage2_batch,
    )


def pil_to_normalized_tensor(image: Image.Image, resolution: int) -> Any:
    image = image.convert("RGB")
    width, height = image.size
    scale = float(resolution) / float(min(width, height))
    resized_width = max(int(round(width * scale)), resolution)
    resized_height = max(int(round(height * scale)), resolution)
    image = image.resize((resized_width, resized_height), Image.BICUBIC)

    left = max((resized_width - resolution) // 2, 0)
    top = max((resized_height - resolution) // 2, 0)
    image = image.crop((left, top, left + resolution, top + resolution))

    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array * 2.0) - 1.0
    array = np.transpose(array, (2, 0, 1))

    import torch

    return torch.from_numpy(array)



def _collate_stage2_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    batch = {
        "pair_id": [item["pair_id"] for item in items],
        "record_id": [item["record_id"] for item in items],
        "sample_id": [item["sample_id"] for item in items],
        "image_path": [item["image_path"] for item in items],
        "caption": [item["caption"] for item in items],
        "conditioning_text": [item["conditioning_text"] for item in items],
        "class_name": [item["class_name"] for item in items],
        "archetype": [item["archetype"] for item in items],
    }
    pixel_values = [item.get("pixel_values") for item in items if item.get("pixel_values") is not None]
    if len(pixel_values) == len(items):
        batch["pixel_values"] = torch.stack(pixel_values, dim=0)
    return batch



def _probe_image_size(path: str | Path) -> tuple[int | None, int | None]:
    with Image.open(path) as image:
        width, height = image.size
    return int(width), int(height)
