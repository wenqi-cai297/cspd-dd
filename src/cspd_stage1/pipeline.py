from __future__ import annotations

"""Main execution pipeline for CSPD Stage 1.

This module ties together:
- dataset loading,
- prompt construction,
- VLM invocation,
- response validation / normalization,
- artifact writing.

The current implementation is intentionally conservative: correctness and clear
artifacts matter more than premature optimization.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cspd_stage1.io_utils import read_jsonl, write_json, write_jsonl
from cspd_stage1.prompting import SYSTEM_PROMPT, build_user_prompt
from cspd_stage1.schema import AttributeRecord, SampleRecord, validate_attribute_payload
from cspd_stage1.vlm.base import BaseVLMClient
from cspd_stage1.vlm.factory import create_vlm_client


@dataclass(slots=True)
class Stage1Config:
    """Runtime configuration for Stage 1."""

    input_path: str
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
    """Run attribute extraction over the full input dataset.

    Output artifacts:
    - attributes.jsonl: successful extractions
    - failed_samples.jsonl: failures that need inspection or rerun
    - stage1_stats.json: high-level summary for quick debugging
    """
    raw_samples = read_jsonl(config.input_path)
    samples = [SampleRecord.from_dict(item) for item in raw_samples]
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

    for sample in samples:
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

    output_dir = Path(config.output_dir)
    write_jsonl(output_dir / "attributes.jsonl", success_rows)
    write_jsonl(output_dir / "failed_samples.jsonl", failed_rows)
    stats = {
        "num_samples": len(samples),
        "num_success": len(success_rows),
        "num_failed": len(failed_rows),
        "backend": config.backend,
        "max_retries": config.max_retries,
        "model_name": config.model_name,
        "torch_dtype": config.torch_dtype,
        "device_map": config.device_map,
    }
    write_json(output_dir / "stage1_stats.json", stats)
    return stats


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


def config_from_args(args) -> Stage1Config:
    """Map parsed CLI args into a structured config object."""
    return Stage1Config(
        input_path=args.input,
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
