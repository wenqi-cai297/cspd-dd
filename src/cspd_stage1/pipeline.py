from __future__ import annotations

"""Main execution pipeline for CSPD Stage 1.

This module ties together:
- ImageFolder-style dataset scanning,
- optional class-label mapping for synset-style folders,
- optional explicit class->archetype mapping,
- class-adaptive slot schema selection,
- VLM invocation,
- response validation / normalization,
- incremental artifact writing,
- progress reporting for long-running CLI jobs.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from cspd_stage1.io_utils import append_jsonl, write_json, write_jsonl
from cspd_stage1.prompting import SYSTEM_PROMPT, build_user_prompt
from cspd_stage1.schema import (
    SampleRecord,
    extract_attribute_mapping,
    get_slot_schema,
    infer_archetype,
    normalize_attributes,
    validate_attribute_payload,
)
from cspd_stage1.vlm.base import BaseVLMClient, VLMOutputParseError
from cspd_stage1.vlm.factory import create_vlm_client

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(slots=True)
class Stage1Config:
    """Runtime configuration for Stage 1."""

    dataset_root: str
    output_dir: str
    backend: str = "mock"
    max_retries: int = 2
    save_raw_response: bool = True
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    torch_dtype: str = "float16"
    device_map: str = "auto"
    use_fast_processor: bool = True
    max_new_tokens: int = 256
    class_name_map: str | None = None
    class_archetype_map: str | None = None
    flush_every: int = 10


def run_stage1(config: Stage1Config) -> dict[str, Any]:
    """Run attribute extraction over an ImageFolder-style dataset."""
    class_name_map = load_string_mapping(config.class_name_map, "class-name map")
    class_archetype_map = load_string_mapping(config.class_archetype_map, "class-archetype map")
    samples = build_samples_from_imagefolder(config.dataset_root, class_name_map, class_archetype_map)
    client = create_vlm_client(
        config.backend,
        model_name=config.model_name,
        torch_dtype=config.torch_dtype,
        device_map=config.device_map,
        use_fast_processor=config.use_fast_processor,
        max_new_tokens=config.max_new_tokens,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    success_path = output_dir / "attributes.jsonl"
    failed_path = output_dir / "failed_samples.jsonl"
    stats_path = output_dir / "stage1_stats.json"

    write_jsonl(success_path, [])
    write_jsonl(failed_path, [])

    dataset_root_resolved = str(Path(config.dataset_root).resolve())
    success_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    pending_success_rows: list[dict[str, Any]] = []
    pending_failed_rows: list[dict[str, Any]] = []

    progress = tqdm(samples, desc="Stage1 attribute extraction", unit="img", dynamic_ncols=True)
    for index, sample in enumerate(progress, start=1):
        progress.set_postfix_str(_build_progress_postfix(sample, len(success_rows), len(failed_rows)))
        result = _process_sample(sample=sample, client=client, max_retries=config.max_retries)
        base = _build_sample_metadata(sample, config, dataset_root_resolved)
        if result["status"] == "success":
            row = {
                **base,
                "attributes": result["attributes"],
                "extraction_status": "success",
            }
            if config.save_raw_response:
                row["raw_response"] = result.get("raw_response")
            success_rows.append(row)
            pending_success_rows.append(row)
        else:
            row = {
                **base,
                "extraction_status": "failed",
                "error_message": result.get("error_message", "unknown error"),
                "raw_response": result.get("raw_response"),
            }
            failed_rows.append(row)
            pending_failed_rows.append(row)

        progress.set_postfix_str(_build_progress_postfix(sample, len(success_rows), len(failed_rows)))
        should_flush = index == len(samples) or index % max(config.flush_every, 1) == 0
        if should_flush:
            _flush_partial_results(
                success_path=success_path,
                failed_path=failed_path,
                stats_path=stats_path,
                pending_success_rows=pending_success_rows,
                pending_failed_rows=pending_failed_rows,
                stats=_build_stats(config, samples, success_rows, failed_rows),
            )
            progress.write(
                f"[progress] {index}/{len(samples)} done | success={len(success_rows)} | failed={len(failed_rows)}"
            )

    progress.close()

    final_stats = _build_stats(config, samples, success_rows, failed_rows)
    write_json(stats_path, final_stats)
    return final_stats


def _build_sample_metadata(sample: SampleRecord, config: Stage1Config, dataset_root_resolved: str) -> dict[str, Any]:
    image_path = Path(sample.image_path)
    relative_path = sample.sample_id or image_path.name
    split = Path(dataset_root_resolved).name
    extracted_at = datetime.now(timezone.utc).isoformat()
    record_id = f"{sample.class_name_raw}::{relative_path}"
    return {
        "record_id": record_id,
        "dataset_root": dataset_root_resolved,
        "split": split,
        "sample_id": sample.sample_id,
        "relative_image_path": relative_path,
        "image_path": sample.image_path,
        "file_name": image_path.name,
        "class_id": sample.class_id,
        "class_name_raw": sample.class_name_raw,
        "class_name": sample.class_name,
        "archetype": sample.archetype,
        "slot_schema": sample.slot_schema,
        "backend": config.backend,
        "model_name": config.model_name,
        "extracted_at": extracted_at,
    }


def _flush_partial_results(
    *,
    success_path: Path,
    failed_path: Path,
    stats_path: Path,
    pending_success_rows: list[dict[str, Any]],
    pending_failed_rows: list[dict[str, Any]],
    stats: dict[str, Any],
) -> None:
    if pending_success_rows:
        append_jsonl(success_path, pending_success_rows)
        pending_success_rows.clear()
    if pending_failed_rows:
        append_jsonl(failed_path, pending_failed_rows)
        pending_failed_rows.clear()
    write_json(stats_path, stats)


def _build_stats(
    config: Stage1Config,
    samples: list[SampleRecord],
    success_rows: list[dict[str, Any]],
    failed_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "dataset_root": str(Path(config.dataset_root).resolve()),
        "num_samples": len(samples),
        "num_success": len(success_rows),
        "num_failed": len(failed_rows),
        "backend": config.backend,
        "max_retries": config.max_retries,
        "model_name": config.model_name,
        "torch_dtype": config.torch_dtype,
        "device_map": config.device_map,
        "num_classes": len({sample.class_name_raw for sample in samples}),
        "class_name_map": config.class_name_map,
        "class_archetype_map": config.class_archetype_map,
        "archetypes": sorted({sample.archetype for sample in samples}),
        "flush_every": config.flush_every,
    }


def load_string_mapping(path: str | None, mapping_name: str) -> dict[str, str]:
    """Load an optional string->string JSON mapping file."""
    if path is None:
        return {}
    mapping_path = Path(path)
    if not mapping_path.exists():
        raise FileNotFoundError(f"{mapping_name} not found: {mapping_path}")
    payload = json.loads(mapping_path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{mapping_name} must be a JSON object")
    normalized: dict[str, str] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError(f"{mapping_name} must be a string-to-string mapping")
        normalized[key] = value
    return normalized


def build_samples_from_imagefolder(
    dataset_root: str,
    class_name_map: dict[str, str],
    class_archetype_map: dict[str, str],
) -> list[SampleRecord]:
    """Scan an ImageFolder-style dataset and build Stage 1 sample records."""
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {root}")

    class_dirs = sorted([path for path in root.iterdir() if path.is_dir()], key=lambda p: p.name)
    if not class_dirs:
        raise ValueError(f"No class subdirectories found under dataset root: {root}")

    samples: list[SampleRecord] = []
    for class_id, class_dir in enumerate(class_dirs):
        class_name_raw = class_dir.name
        class_name = class_name_map.get(class_name_raw, class_name_raw)
        archetype = class_archetype_map.get(class_name_raw, infer_archetype(class_name))
        slot_schema = get_slot_schema(archetype)
        image_files = sorted(
            [path for path in class_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
        )
        for image_path in image_files:
            sample_id = str(image_path.relative_to(root)).replace("\\", "/")
            samples.append(
                SampleRecord(
                    image_path=str(image_path),
                    class_id=class_id,
                    class_name=class_name,
                    class_name_raw=class_name_raw,
                    archetype=archetype,
                    slot_schema=list(slot_schema),
                    sample_id=sample_id,
                )
            )

    if not samples:
        raise ValueError(f"No image files found under dataset root: {root}")
    return samples


def _process_sample(sample: SampleRecord, client: BaseVLMClient, max_retries: int) -> dict[str, Any]:
    system_prompt = SYSTEM_PROMPT
    user_prompt = build_user_prompt(sample)

    attempts = max_retries + 1
    last_error = ""
    last_raw = None

    for _ in range(attempts):
        try:
            response = client.extract_attributes(sample, user_prompt=user_prompt, system_prompt=system_prompt)
            last_raw = response.raw_text
            valid, errors = validate_attribute_payload(response.payload, sample.slot_schema)
            if not valid:
                last_error = "; ".join(errors)
                continue
            attribute_mapping = extract_attribute_mapping(response.payload)
            attributes = normalize_attributes(attribute_mapping, sample.slot_schema)
            return {
                "status": "success",
                "attributes": attributes,
                "raw_response": response.raw_text,
            }
        except VLMOutputParseError as exc:
            last_error = str(exc)
            last_raw = exc.raw_text
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)

    return {
        "status": "failed",
        "error_message": last_error or "attribute extraction failed",
        "raw_response": last_raw,
    }


def _build_progress_postfix(sample: SampleRecord, num_success: int, num_failed: int) -> str:
    sample_label = sample.sample_id or Path(sample.image_path).name
    return (
        f"class={sample.class_name_raw}→{sample.archetype} | success={num_success} | failed={num_failed} | "
        f"sample={_truncate_text(sample_label, 36)}"
    )


def _truncate_text(text: str, max_length: int) -> str:
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def config_from_args(args) -> Stage1Config:
    return Stage1Config(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        backend=args.backend,
        max_retries=args.max_retries,
        save_raw_response=not args.no_raw_response,
        model_name=args.model_name,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        use_fast_processor=not args.disable_fast_processor,
        max_new_tokens=args.max_new_tokens,
        class_name_map=args.class_name_map,
        class_archetype_map=args.class_archetype_map,
        flush_every=args.flush_every,
    )
