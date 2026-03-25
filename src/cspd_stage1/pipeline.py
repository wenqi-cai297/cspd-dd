from __future__ import annotations

"""Main execution pipeline for CSPD Stage 1.

This module ties together:
- ImageFolder-style dataset scanning,
- prompt construction,
- VLM invocation,
- response validation / normalization,
- artifact writing,
- progress reporting for long-running CLI jobs.

The current implementation assumes the input dataset uses a simple ImageFolder
layout:
    dataset_root/
      class_a/
        img1.jpg
        img2.jpg
      class_b/
        img3.jpg

That assumption is deliberate for now because the user's current datasets all
follow that structure, and there is no point pretending we need a general data
abstraction layer before the actual method is even running.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from cspd_stage1.io_utils import write_json, write_jsonl
from cspd_stage1.prompting import SYSTEM_PROMPT, build_user_prompt
from cspd_stage1.schema import AttributeRecord, SampleRecord, validate_attribute_payload
from cspd_stage1.vlm.base import BaseVLMClient
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


def run_stage1(config: Stage1Config) -> dict[str, Any]:
    """Run attribute extraction over an ImageFolder-style dataset.

    Output artifacts:
    - attributes.jsonl: successful extractions
    - failed_samples.jsonl: failures that need inspection or rerun
    - stage1_stats.json: high-level summary for quick debugging
    """
    samples = build_samples_from_imagefolder(config.dataset_root)
    client = create_vlm_client(
        config.backend,
        model_name=config.model_name,
        torch_dtype=config.torch_dtype,
        device_map=config.device_map,
        use_fast_processor=config.use_fast_processor,
        max_new_tokens=config.max_new_tokens,
    )

    success_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []

    progress = tqdm(samples, desc="Stage1 attribute extraction", unit="img", dynamic_ncols=True)
    for index, sample in enumerate(progress, start=1):
        progress.set_postfix_str(_build_progress_postfix(sample, len(success_rows), len(failed_rows)))
        result = _process_sample(sample=sample, client=client, max_retries=config.max_retries)
        base = {
            "sample_id": sample.sample_id,
            "image_path": sample.image_path,
            "class_id": sample.class_id,
            "class_name": sample.class_name,
        }
        if result["status"] == "success":
            row = {
                **base,
                "attributes": result["attributes"],
                "extraction_status": "success",
            }
            if config.save_raw_response:
                row["raw_response"] = result.get("raw_response")
            success_rows.append(row)
        else:
            failed_rows.append(
                {
                    **base,
                    "extraction_status": "failed",
                    "error_message": result.get("error_message", "unknown error"),
                    "raw_response": result.get("raw_response"),
                }
            )

        progress.set_postfix_str(_build_progress_postfix(sample, len(success_rows), len(failed_rows)))
        if index == len(samples) or index % 10 == 0:
            progress.write(
                f"[progress] {index}/{len(samples)} done | success={len(success_rows)} | failed={len(failed_rows)}"
            )

    progress.close()

    output_dir = Path(config.output_dir)
    write_jsonl(output_dir / "attributes.jsonl", success_rows)
    write_jsonl(output_dir / "failed_samples.jsonl", failed_rows)
    stats = {
        "dataset_root": str(Path(config.dataset_root).resolve()),
        "num_samples": len(samples),
        "num_success": len(success_rows),
        "num_failed": len(failed_rows),
        "backend": config.backend,
        "max_retries": config.max_retries,
        "model_name": config.model_name,
        "torch_dtype": config.torch_dtype,
        "device_map": config.device_map,
        "num_classes": len({sample.class_name for sample in samples}),
    }
    write_json(output_dir / "stage1_stats.json", stats)
    return stats


def build_samples_from_imagefolder(dataset_root: str) -> list[SampleRecord]:
    """Scan an ImageFolder-style dataset and build Stage 1 sample records.

    Class ids are assigned by sorting subdirectory names alphabetically, which is
    deterministic and matches common ImageFolder conventions.
    """
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
        class_name = class_dir.name
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
                    sample_id=sample_id,
                )
            )

    if not samples:
        raise ValueError(f"No image files found under dataset root: {root}")
    return samples


def _process_sample(sample: SampleRecord, client: BaseVLMClient, max_retries: int) -> dict[str, Any]:
    """Run extraction for a single sample with retry and schema validation.

    We treat malformed payloads and backend exceptions the same way at the
    control-flow level: retry a limited number of times, then mark the sample as
    failed with the last observed error.
    """
    system_prompt = SYSTEM_PROMPT
    user_prompt = build_user_prompt(sample)

    # `max_retries=2` means 1 initial attempt + 2 retries = 3 total tries.
    attempts = max_retries + 1
    last_error = ""
    last_raw = None

    for _ in range(attempts):
        try:
            response = client.extract_attributes(sample, user_prompt=user_prompt, system_prompt=system_prompt)
            last_raw = response.raw_text
            valid, errors = validate_attribute_payload(response.payload)
            if not valid:
                last_error = "; ".join(errors)
                continue
            attributes = AttributeRecord.from_dict(response.payload)
            return {
                "status": "success",
                "attributes": attributes.to_dict(),
                "raw_response": response.raw_text,
            }
        except Exception as exc:  # noqa: BLE001
            # We keep broad exception capture here because backend integrations
            # may fail in many messy ways: transport, auth, parsing, timeouts...
            last_error = str(exc)

    return {
        "status": "failed",
        "error_message": last_error or "attribute extraction failed",
        "raw_response": last_raw,
    }


def _build_progress_postfix(sample: SampleRecord, num_success: int, num_failed: int) -> str:
    """Build a compact progress summary shown next to the tqdm bar."""
    sample_label = sample.sample_id or Path(sample.image_path).name
    return (
        f"class={sample.class_name} | success={num_success} | failed={num_failed} | "
        f"sample={_truncate_text(sample_label, 48)}"
    )


def _truncate_text(text: str, max_length: int) -> str:
    """Shorten long sample ids so the progress bar stays readable."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def config_from_args(args) -> Stage1Config:
    """Map parsed CLI args into a structured config object."""
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
    )
