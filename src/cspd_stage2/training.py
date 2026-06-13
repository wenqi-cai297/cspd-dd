from __future__ import annotations

"""Stage 2 training orchestration (SDXL LoRA only).

Responsibilities:
- build the Stage 2 paired manifest from Stage 1 render records,
- write a run directory with config snapshot + manifest,
- delegate actual training to the official diffusers SDXL LoRA trainer via
  `cspd_stage2.families.sdxl.training.run_stage2_sdxl_official_training`.

The self-built FLUX / PixArt / SD v1.5 loops that used to live here were
removed on 2026-04-18 along with the corresponding family subpackages.
"""

import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from cspd_stage1.io_utils import write_json
from cspd_stage2.backbone import infer_backbone_family
from cspd_stage2.data import ManifestPaths, build_stage2_pairs, write_pairing_artifacts
from cspd_stage2.families.sdxl.training import run_stage2_sdxl_official_training
from cspd_stage2.training_common import (
    _safe_write_json,
    derive_stage2_dataset_label,
    derive_stage2_output_dir,
)


@dataclass(slots=True)
class AdapterPlan:
    """Minimal adapter-config record.

    Only `rank` is consumed by the SDXL wrapper (`--rank` on the official
    diffusers trainer). The other fields are kept so existing config
    snapshots remain roundtrip-serializable.
    """

    rank: int = 64
    alpha: float = 64.0
    dropout: float = 0.0
    bias: str = "none"


@dataclass(slots=True)
class Stage2TrainConfig:
    dataset_root: str
    render_input: str
    output_dir: str | None = None
    backbone_name: str = "stabilityai/stable-diffusion-xl-base-1.0"
    batch_size: int = 8
    learning_rate: float = 2e-5
    epochs: int = 9
    max_steps: int | None = None
    num_workers: int = 0
    resolution: int = 512
    seed: int = 42
    save_every: int = 200
    max_train_samples: int | None = None
    class_name_map: str | None = None
    class_archetype_map: str | None = None
    verify_images: bool = False
    strict_pairing: bool = False
    dry_run: bool = False
    generate_manifest_only: bool = False
    use_accelerate: bool = True
    gradient_accumulation_steps: int = 1
    adapter_plan: AdapterPlan = field(default_factory=AdapterPlan)
    # SDXL official trainer knobs
    sdxl_official_script: str | None = None
    sdxl_num_processes: int | None = None
    sdxl_accelerate_extra_args: list[str] = field(default_factory=list)
    sdxl_mixed_precision: str = "fp16"
    sdxl_lr_scheduler: str = "cosine"
    sdxl_lr_warmup_steps: int = 500
    sdxl_validation_epochs: int = 1
    sdxl_validation_prompt: str | None = None
    sdxl_report_to: str = "tensorboard"
    sdxl_use_8bit_adam: bool = False
    sdxl_enable_xformers: bool = False
    sdxl_gradient_checkpointing: bool = True
    sdxl_train_text_encoder: bool = False
    sdxl_caption_dropout_probability: float | None = None
    sdxl_noise_offset: float | None = 0.05
    sdxl_snr_gamma: float | None = 5.0
    sdxl_extra_args: list[str] = field(default_factory=list)


def _config_to_dict(config: Stage2TrainConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["dataset_label"] = derive_stage2_dataset_label(config.dataset_root)
    payload["family"] = infer_backbone_family(config.backbone_name)
    return payload


def _read_text_tail(path_value: str | None, *, max_lines: int = 80) -> list[str]:
    if not path_value:
        return []
    path = Path(path_value)
    if not path.exists() or not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return lines[-max_lines:]


def run_stage2_training(config: Stage2TrainConfig) -> dict[str, Any]:
    """Build Stage 2 artifacts and launch the official SDXL LoRA training."""
    if not config.output_dir:
        config.output_dir = derive_stage2_output_dir(config.dataset_root, config.backbone_name)
    run_dir = Path(config.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    training_result: dict[str, Any] = {
        "status": "manifest_ready",
        "implemented_training": False,
        "message": "Stage 2 paired manifest is ready.",
    }
    manifest_paths: ManifestPaths | None = None
    pairing_summary: dict[str, Any] | None = None
    num_pairs = 0
    top_level_failure: dict[str, Any] | None = None

    try:
        pairing = build_stage2_pairs(
            dataset_root=config.dataset_root,
            render_input=config.render_input,
            class_name_map=config.class_name_map,
            class_archetype_map=config.class_archetype_map,
            verify_images=config.verify_images,
            strict=config.strict_pairing,
        )
        if config.max_train_samples is not None:
            pairing.pairs = pairing.pairs[: max(config.max_train_samples, 0)]
            pairing.summary["max_train_samples_applied"] = config.max_train_samples
            pairing.summary["num_pairs_after_limit"] = len(pairing.pairs)
        else:
            pairing.summary["num_pairs_after_limit"] = len(pairing.pairs)

        pairing_summary = pairing.summary
        num_pairs = len(pairing.pairs)

        manifest_paths = write_pairing_artifacts(pairing, run_dir)
        write_json(run_dir / "stage2_config_snapshot.json", _config_to_dict(config))

        family = infer_backbone_family(config.backbone_name)
        if family == "sdxl" and not config.generate_manifest_only and not config.dry_run:
            try:
                training_result = run_stage2_sdxl_official_training(
                    config=config,
                    pairs=pairing.pairs,
                    run_dir=run_dir,
                    manifest_path=manifest_paths.manifest_path,
                )
            except Exception as exc:  # noqa: BLE001
                training_result = {
                    "status": "failed_before_training",
                    "implemented_training": False,
                    "message": (
                        "SDXL Stage 2 training was attempted but could not start or complete in this environment."
                    ),
                    "training_error": str(exc),
                    "training_traceback": traceback.format_exc(),
                }
        elif family != "sdxl":
            training_result = {
                "status": "unsupported_backbone",
                "implemented_training": False,
                "message": (
                    f"Stage 2 only supports SDXL backbones; got family={family!r} "
                    f"from backbone_name={config.backbone_name!r}."
                ),
            }

        if (
            training_result.get("status") in {"failed", "failed_before_training", "failed_before_training_setup_complete"}
            or int(training_result.get("returncode", 0) or 0) != 0
        ):
            training_result["stderr_tail"] = _read_text_tail(training_result.get("stderr_path"))
            training_result["stdout_tail"] = _read_text_tail(training_result.get("stdout_path"), max_lines=40)
    except Exception as exc:  # noqa: BLE001
        top_level_failure = {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        training_result = {
            "status": "failed_before_training_setup_complete",
            "implemented_training": False,
            "message": "Stage 2 setup failed before training could start. See top_level_failure for details.",
        }

    summary = {
        "config_snapshot": _config_to_dict(config),
        "pairing_summary": pairing_summary,
        "manifest_paths": asdict(manifest_paths) if manifest_paths else None,
        "num_pairs": num_pairs,
        "training_result": training_result,
        "top_level_failure": top_level_failure,
        "run_dir": str(run_dir.resolve()),
    }
    _safe_write_json(run_dir / "stage2_run_summary.json", summary)
    return summary
