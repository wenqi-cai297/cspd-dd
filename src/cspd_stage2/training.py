from __future__ import annotations

"""Training utilities for CSPD Stage 2 (SDXL LoRA only).

This module prepares run directories + paired manifests and hands the actual
training off to the official diffusers SDXL LoRA trainer via
`cspd_stage2.families.sdxl.training.run_stage2_sdxl_official_training`.
Legacy FLUX / PixArt / SD v1.5 paths were removed 2026-04-18.
"""

import fnmatch
import importlib.util
import json
import os
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
import hashlib
import math
import re
from contextlib import nullcontext

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional runtime dependency
    tqdm = None

from cspd_stage1.io_utils import write_json
from cspd_stage2.backbone import (
    apply_trainable_parameter_selection,
    infer_backbone_family,
    inject_lora_adapters,
    inspect_target_modules,
    load_generative_backbone,
    load_module_from_reference,
    load_real_backbone_module,
)
from cspd_stage2.data import ManifestPaths, build_stage2_pairs, make_stage2_dataloader, write_pairing_artifacts
from cspd_stage2.families.sdxl.training import run_stage2_sdxl_official_training
from cspd_stage2.training_common import (
    CONDITIONING_RELATED_GROUP_PATTERNS,
    DEFAULT_EXCLUDE_PATTERNS,
    DEFAULT_LORA_TARGET_GROUPS,
    DEFAULT_TEXT_CONDITIONING_GROUPS,
    _build_lr_scheduler,
    _build_optimizer,
    _freeze_stage2_modules,
    _prompt_slug,
    _resolve_lora_master_weight_dtype,
    _safe_write_json,
    _save_pil_like_image,
    _should_force_full_update_fp32,
    _torch_dtype_label,
    _upcast_trainable_parameters_,
    derive_stage2_dataset_label,
    derive_stage2_output_dir,
    resolve_effective_module_selection,
)


@dataclass(slots=True)
class AdapterPlan:
    adapter_type: str = "lora"
    rank: int = 16
    alpha: float = 16.0
    dropout: float = 0.0
    target_strategy: str = "module_name_patterns"
    target_module_patterns: list[str] = field(default_factory=list)
    exclude_module_patterns: list[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDE_PATTERNS))
    bias: str = "none"
    master_weight_dtype: str | None = None
    task_note: str = "placeholder adapter config until a concrete backbone-specific training stack is integrated"


@dataclass(slots=True)
class Stage2TrainConfig:
    dataset_root: str
    render_input: str
    output_dir: str | None = None
    backbone_name: str = "stabilityai/stable-diffusion-xl-base-1.0"
    memory_log_artifact_name: str = "memory_diagnostics.jsonl"
    wandb_enabled: bool = False
    wandb_project: str = "cspd-stage2"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_tags: list[str] = field(default_factory=list)
    wandb_mode: str = "online"
    wandb_dir: str | None = None
    wandb_resume: str | None = None
    wandb_run_id: str | None = None
    batch_size: int = 4
    learning_rate: float = 2e-5
    epochs: int = 1
    max_steps: int | None = None
    num_workers: int = 0
    resolution: int = 512
    seed: int = 42
    weight_dtype: str = "float16"
    optimizer_name: str = "adamw"
    lr_scheduler: str = "constant_with_warmup"
    lr_warmup_steps: int = 1000
    max_grad_norm: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 0.0
    adam_epsilon: float = 1e-8
    log_every: int = 10
    save_every: int = 200
    max_train_samples: int | None = None
    class_name_map: str | None = None
    class_archetype_map: str | None = None
    verify_images: bool = False
    strict_pairing: bool = False
    dry_run: bool = False
    generate_manifest_only: bool = False
    freeze_text_encoder: bool = True
    freeze_vae: bool = True
    train_transformer_core_only: bool = True
    stage2_focus: str = "transformer_finetuning"
    conditioning_objective: str = "finetune_generative_transformer_on_real_image_and_stage1_canonical_caption_pairs"
    conditioning_text_field: str = "canonical_caption"
    trainable_component_groups: list[str] = field(default_factory=lambda: list(DEFAULT_TEXT_CONDITIONING_GROUPS))
    module_include_patterns: list[str] = field(default_factory=list)
    module_exclude_patterns: list[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDE_PATTERNS))
    training_parameterization: str = "full"
    adapter_plan: AdapterPlan = field(default_factory=AdapterPlan)
    inspect_module_reference: str | None = None
    inspect_limit: int = 200
    apply_real_module_selection: bool = False
    inject_adapters_on_real_module: bool = False
    backbone_torch_dtype: str = "bfloat16"
    backbone_device: str | None = None
    backbone_device_map: str | None = None
    backbone_local_files_only: bool = False
    backbone_component: str | None = None
    use_accelerate: bool = True
    gradient_accumulation_steps: int = 1
    dataloader_drop_last: bool = False
    enable_gradient_checkpointing: bool = True
    sdxl_official_script: str | None = None
    sdxl_num_processes: int | None = None
    sdxl_accelerate_extra_args: list[str] = field(default_factory=list)
    sdxl_mixed_precision: str = "fp16"
    sdxl_lr_scheduler: str = "cosine"
    sdxl_lr_warmup_steps: int = 500
    sdxl_validation_epochs: int = 1
    sdxl_validation_prompt: str | None = None
    sdxl_report_to: str = "none"
    sdxl_use_8bit_adam: bool = False
    sdxl_enable_xformers: bool = False
    sdxl_gradient_checkpointing: bool = True
    sdxl_train_text_encoder: bool = False
    sdxl_caption_dropout_probability: float | None = None
    sdxl_noise_offset: float | None = 0.05
    sdxl_snr_gamma: float | None = 5.0
    sdxl_extra_args: list[str] = field(default_factory=list)


def _try_import_wandb() -> Any | None:
    try:
        import wandb  # type: ignore
    except Exception:
        return None
    return wandb


def _resolve_wandb_dir(config: Stage2TrainConfig, run_dir: Path) -> str:
    return str((Path(config.wandb_dir).expanduser() if config.wandb_dir else (run_dir / "wandb")).resolve())


def _init_wandb_run(*, config: Stage2TrainConfig, run_dir: Path, is_main_process: bool) -> tuple[Any | None, dict[str, Any] | None]:
    if not config.wandb_enabled or not is_main_process:
        return None, None
    wandb = _try_import_wandb()
    if wandb is None:
        return None, {"enabled": False, "requested": True, "status": "wandb_not_installed"}
    wandb_dir = _resolve_wandb_dir(config, run_dir)
    run = wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=config.wandb_run_name,
        tags=config.wandb_tags or None,
        mode=config.wandb_mode,
        dir=wandb_dir,
        resume=config.wandb_resume,
        id=config.wandb_run_id,
        config=_config_to_dict(config),
        settings=wandb.Settings(start_method="thread"),
    )
    return run, {
        "enabled": True,
        "requested": True,
        "status": "initialized",
        "project": config.wandb_project,
        "entity": config.wandb_entity,
        "name": getattr(run, "name", None),
        "id": getattr(run, "id", None),
        "url": getattr(run, "url", None),
        "mode": config.wandb_mode,
        "dir": wandb_dir,
        "tags": list(config.wandb_tags),
    }


def _finish_wandb_run(run: Any | None, summary: dict[str, Any] | None = None) -> None:
    if run is None:
        return
    if summary:
        for key, value in summary.items():
            try:
                run.summary[key] = value
            except Exception:
                continue
    try:
        run.finish()
    except Exception:
        pass


def _collect_step_metrics(*, loss_value: float, optimizer: Any, grad_diagnostics: dict[str, Any] | None = None, parameter_diagnostics: dict[str, Any] | None = None, memory_stats: dict[str, Any] | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    metrics: dict[str, Any] = {"loss": float(loss_value), "lr": float(optimizer.param_groups[0]["lr"])}
    if grad_diagnostics is not None:
        metrics["grad_global_norm"] = grad_diagnostics.get("grad_global_norm")
        metrics["grad_parameter_count"] = grad_diagnostics.get("grad_parameter_count")
        metrics["grad_value_count"] = grad_diagnostics.get("grad_value_count")
    if parameter_diagnostics is not None:
        metrics["trainable_parameter_count"] = parameter_diagnostics.get("trainable_parameter_count")
        metrics["trainable_value_count"] = parameter_diagnostics.get("trainable_value_count")
        metrics["max_abs_trainable_parameter_value"] = parameter_diagnostics.get("max_abs_trainable_parameter_value")
        metrics["all_trainable_parameters_finite"] = 1.0 if parameter_diagnostics.get("all_trainable_parameters_finite") else 0.0
        for dtype_label, count in sorted((parameter_diagnostics.get("trainable_parameter_dtype_counts") or {}).items()):
            metrics[f"trainable_parameter_dtype_count/{dtype_label}"] = count
    if memory_stats is not None and memory_stats.get("memory_stats_available"):
        metrics["cuda_memory_allocated_bytes"] = memory_stats.get("allocated_bytes")
        metrics["cuda_memory_reserved_bytes"] = memory_stats.get("reserved_bytes")
        metrics["cuda_memory_max_allocated_bytes"] = memory_stats.get("max_allocated_bytes")
        metrics["cuda_memory_max_reserved_bytes"] = memory_stats.get("max_reserved_bytes")
    if extra:
        metrics.update(extra)
    return {key: value for key, value in metrics.items() if value is not None}


def _wandb_log(run: Any | None, payload: dict[str, Any], *, step: int) -> None:
    if run is None or not payload:
        return
    try:
        run.log(payload, step=step)
    except Exception:
        pass






def run_stage2_training(config: Stage2TrainConfig) -> dict[str, Any]:
    """Build Stage 2 artifacts and run the smallest honest training path available."""
    if not config.output_dir:
        config.output_dir = derive_stage2_output_dir(config.dataset_root, config.backbone_name)
    run_dir = Path(config.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_paths: ManifestPaths | None = None
    pairing_summary: dict[str, Any] | None = None
    num_pairs = 0
    component_plan: dict[str, Any] | None = None
    backbone_runtime: dict[str, Any] | None = None
    last_known_phase = "run_dir_ready"
    top_level_failure: dict[str, Any] | None = None
    training_result: dict[str, Any] = {
        "status": "manifest_ready",
        "implemented_training": False,
        "placeholder_training": False,
        "message": "Stage 2 paired manifest is ready.",
        "component_plan_status": "implemented_metadata_only",
    }

    try:
        last_known_phase = "before_pairing"
        pairing = build_stage2_pairs(
            dataset_root=config.dataset_root,
            render_input=config.render_input,
            class_name_map=config.class_name_map,
            class_archetype_map=config.class_archetype_map,
            verify_images=config.verify_images,
            strict=config.strict_pairing,
        )
        last_known_phase = "after_pairing"

        if config.max_train_samples is not None:
            pairing.pairs = pairing.pairs[: max(config.max_train_samples, 0)]
            pairing.summary["max_train_samples_applied"] = config.max_train_samples
            pairing.summary["num_pairs_after_limit"] = len(pairing.pairs)
        else:
            pairing.summary["num_pairs_after_limit"] = len(pairing.pairs)

        pairing_summary = pairing.summary
        num_pairs = len(pairing.pairs)

        last_known_phase = "before_write_pairing_artifacts"
        manifest_paths = write_pairing_artifacts(pairing, run_dir)
        last_known_phase = "after_write_pairing_artifacts"

        last_known_phase = "before_backbone_runtime_summary"
        backbone_runtime = _build_backbone_runtime_summary(config)
        last_known_phase = "after_backbone_runtime_summary"

        last_known_phase = "before_component_plan"
        component_plan = _build_component_plan(config, backbone_runtime)
        last_known_phase = "after_component_plan"

        last_known_phase = "before_write_config_snapshot"
        write_json(run_dir / "stage2_config_snapshot.json", _config_to_dict(config))
        last_known_phase = "after_write_config_snapshot"

        trainer_plan = _build_trainer_plan(config, manifest_paths.manifest_path, len(pairing.pairs), component_plan)
        last_known_phase = "before_write_trainer_plan"
        write_json(run_dir / "trainer_plan.json", trainer_plan)
        last_known_phase = "after_write_trainer_plan"

        if infer_backbone_family(config.backbone_name) == 'sdxl' and not config.generate_manifest_only and not config.dry_run:
            try:
                last_known_phase = "before_real_training"
                training_result = run_stage2_sdxl_official_training(
                    config=config,
                    pairs=pairing.pairs,
                    run_dir=run_dir,
                    manifest_path=manifest_paths.manifest_path,
                )
                last_known_phase = training_result.get("last_known_phase", "after_real_training")
            except Exception as exc:  # noqa: BLE001
                last_known_phase = "real_training_exception"
                training_result = {
                    "status": "failed_before_training",
                    "implemented_training": False,
                    "placeholder_training": False,
                    "message": (
                        "SDXL Stage 2 training was attempted but could not start or complete in this environment. "
                        "See training_error for the real runtime failure."
                    ),
                    "training_error": str(exc),
                    "training_traceback": traceback.format_exc(),
                    "component_plan_status": "real_training_attempted",
                    "last_known_phase": last_known_phase,
                }
    except Exception as exc:  # noqa: BLE001
        top_level_failure = {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "last_known_phase": last_known_phase,
        }
        training_result = {
            "status": "failed_before_training_setup_complete",
            "implemented_training": False,
            "placeholder_training": False,
            "message": "Stage 2 setup failed before training could start cleanly. See top_level_failure for traceback and last phase.",
            "component_plan_status": "setup_failed",
            "last_known_phase": last_known_phase,
        }

    summary = _build_stage2_run_summary(
        config=config,
        run_dir=run_dir,
        manifest_paths=manifest_paths,
        num_pairs=num_pairs,
        pairing_summary=pairing_summary,
        component_plan=component_plan,
        backbone_runtime=backbone_runtime,
        training_result=training_result,
        last_known_phase=training_result.get("last_known_phase", last_known_phase),
        top_level_failure=top_level_failure,
    )
    _safe_write_json(run_dir / "stage2_run_summary.json", summary)
    return summary



def _build_stage2_run_summary(
    *,
    config: Stage2TrainConfig,
    run_dir: Path,
    manifest_paths: ManifestPaths | None,
    num_pairs: int,
    pairing_summary: dict[str, Any] | None,
    component_plan: dict[str, Any] | None,
    backbone_runtime: dict[str, Any] | None,
    training_result: dict[str, Any],
    last_known_phase: str,
    top_level_failure: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "stage": "stage2_v1",
        "definition": "Stage 2 canonical-caption-conditioned backbone adaptation with Stage 1 canonical-caption conditioning",
        "backbone_name": config.backbone_name,
        "run_dir": str(run_dir.resolve()),
        "stage2_focus": config.stage2_focus,
        "conditioning_objective": config.conditioning_objective,
        "conditioning_text_field": config.conditioning_text_field,
        "train_transformer_core_only": config.train_transformer_core_only,
        "freeze_text_encoder": config.freeze_text_encoder,
        "freeze_vae": config.freeze_vae,
        "last_known_phase": last_known_phase,
        "component_plan": component_plan,
        "backbone_runtime": backbone_runtime,
        "manifest": manifest_paths.manifest_path if manifest_paths is not None else None,
        "manifest_summary": manifest_paths.summary_path if manifest_paths is not None else None,
        "unmatched_images": manifest_paths.unmatched_images_path if manifest_paths is not None else None,
        "unmatched_render_records": manifest_paths.unmatched_render_records_path if manifest_paths is not None else None,
        "num_pairs": num_pairs,
        "pairing_summary": pairing_summary,
        "training": training_result,
        "top_level_failure": top_level_failure,
        "artifacts": {
            "config": str((run_dir / "stage2_config_snapshot.json").resolve()),
            "trainer_plan": str((run_dir / "trainer_plan.json").resolve()),
            "run_summary": str((run_dir / "stage2_run_summary.json").resolve()),
            "memory_log_pattern": str((run_dir / f"rank*_{config.memory_log_artifact_name}").resolve()),
        },
        "created_at": datetime.utcnow().isoformat() + "Z",
    }



def _safe_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _safe_jsonable(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe_jsonable(item) for item in value]
    return str(value)



def _sample_sequence(values: list[str] | tuple[str, ...] | None, *, limit: int = 20) -> dict[str, Any]:
    sequence = list(values or [])
    return {
        "count": len(sequence),
        "sample": sequence[:limit],
        "truncated": len(sequence) > limit,
    }



def _summarize_tensor_like(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    shape = getattr(value, "shape", None)
    return {
        "shape": [int(dim) for dim in shape] if shape is not None else None,
        "dtype": _torch_dtype_label(getattr(value, "dtype", None)) if getattr(value, "dtype", None) is not None else None,
        "device": str(getattr(value, "device", None)) if getattr(value, "device", None) is not None else None,
    }


def _accelerator_rank_info(accelerator: Any | None, device: Any) -> dict[str, Any]:
    local_rank = getattr(accelerator, "local_process_index", 0) if accelerator is not None else 0
    global_rank = getattr(accelerator, "process_index", 0) if accelerator is not None else 0
    world_size = getattr(accelerator, "num_processes", 1) if accelerator is not None else 1
    distributed_type = str(getattr(getattr(accelerator, "state", None), "distributed_type", "no")) if accelerator is not None else "no"
    return {
        "global_rank": int(global_rank),
        "local_rank": int(local_rank),
        "world_size": int(world_size),
        "distributed_type": distributed_type,
        "device": str(device),
        "device_type": getattr(device, "type", None),
        "device_index": getattr(device, "index", None),
        "pid": os.getpid(),
    }



def _collect_cuda_memory_stats(torch: Any, device: Any) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(device),
    }
    if not torch.cuda.is_available() or getattr(device, "type", None) != "cuda":
        stats["memory_stats_available"] = False
        return stats

    resolved_device = device
    if getattr(device, "index", None) is None:
        resolved_device = torch.device("cuda", torch.cuda.current_device())
    stats.update(
        {
            "memory_stats_available": True,
            "device_index": resolved_device.index,
            "allocated_bytes": int(torch.cuda.memory_allocated(resolved_device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(resolved_device)),
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated(resolved_device)),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved(resolved_device)),
        }
    )
    return stats



def _append_memory_event(
    *,
    artifact_path: Path,
    accelerator: Any | None,
    device: Any,
    phase: str,
    torch_module: Any,
    epoch: int | None = None,
    global_step: int | None = None,
    optimizer_step: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "phase": phase,
        "epoch": epoch,
        "global_step": global_step,
        "optimizer_step": optimizer_step,
    }
    payload.update(_accelerator_rank_info(accelerator, device))
    payload["memory"] = _collect_cuda_memory_stats(torch_module, device)
    if extra:
        payload["extra"] = _safe_jsonable(extra)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return payload



def _append_jsonl_event(artifact_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    safe_payload = _safe_jsonable(payload)
    with artifact_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(safe_payload, ensure_ascii=False) + "\n")
    return safe_payload


QUIET_STAGE2_CONSOLE_PHASES = {
    "training_start",
    "after_backbone_load",
    "after_frozen_component_device_setup",
    "after_gradient_checkpointing_setup",
    "after_dataloader_setup",
    "after_accelerate_prepare_dataloader",
    "after_checkpoint_dir_ready",
    "after_lr_scheduler_setup",
    "epoch_start",
    "optimizer_step_complete",
    "before_checkpoint_save",
    "after_checkpoint_save",
    "before_final_checkpoint_save",
    "after_final_checkpoint_save",
    "training_loop_complete",
}


def _should_emit_stage2_console_phase(phase: str) -> bool:
    lowered = str(phase).strip().lower()
    if lowered.startswith("non_finite_"):
        return True
    if "failed" in lowered or "error" in lowered or "exception" in lowered:
        return True
    return phase in QUIET_STAGE2_CONSOLE_PHASES


def _summarize_console_extra(phase: str, extra: dict[str, Any] | None) -> str:
    if not extra:
        return ""
    safe_extra = _safe_jsonable(extra)
    if phase == "epoch_start":
        epoch = safe_extra.get("epoch") or safe_extra.get("epoch_index")
        planned_epochs = safe_extra.get("planned_epochs")
        parts = []
        if epoch is not None:
            parts.append(f"epoch={epoch}")
        if planned_epochs is not None:
            parts.append(f"planned_epochs={planned_epochs}")
        return ", ".join(parts)
    if phase == "optimizer_step_complete":
        selected_keys = ["epoch", "optimizer_step", "global_step", "loss"]
    elif phase in {"before_checkpoint_save", "after_checkpoint_save", "before_final_checkpoint_save", "after_final_checkpoint_save"}:
        selected_keys = ["epoch", "optimizer_step", "checkpoint_dir"]
    elif phase in {"training_start", "after_backbone_load", "after_frozen_component_device_setup", "after_gradient_checkpointing_setup", "after_dataloader_setup", "after_accelerate_prepare_dataloader", "after_checkpoint_dir_ready", "after_lr_scheduler_setup", "training_loop_complete"}:
        selected_keys = ["backbone_name", "num_pairs", "batch_size", "gradient_accumulation_steps", "dataloader_batches_per_epoch", "learning_rate", "loss_count", "loss_total", "loss_last", "optimizer_steps", "checkpoint_dir", "name", "warmup_steps", "epoch", "optimizer_step", "global_step"]
    else:
        selected_keys = list(safe_extra.keys())[:4]
    compact_items = []
    for key in selected_keys:
        if key in safe_extra and safe_extra[key] is not None:
            compact_items.append(f"{key}={safe_extra[key]}")
    return ", ".join(compact_items[:6])


def _emit_stage2_console_event(
    *,
    accelerator: Any | None,
    device: Any,
    phase: str,
    extra: dict[str, Any] | None = None,
    main_process_only: bool = True,
) -> dict[str, Any] | None:
    rank_info = _accelerator_rank_info(accelerator, device)
    is_main_process = bool(getattr(accelerator, "is_main_process", True)) if accelerator is not None else True
    if main_process_only and not is_main_process:
        return None
    payload: dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "phase": phase,
        "rank": rank_info["global_rank"],
        "world_size": rank_info["world_size"],
        "local_rank": rank_info["local_rank"],
        "device": rank_info["device"],
        "pid": rank_info["pid"],
    }
    if extra:
        payload["extra"] = _safe_jsonable(extra)
    if not _should_emit_stage2_console_phase(phase):
        return payload
    message = f"[Stage2][rank {rank_info['global_rank']}/{max(rank_info['world_size'] - 1, 0)}][{rank_info['device']}] {phase}"
    extra_summary = _summarize_console_extra(phase, extra)
    if extra_summary:
        message += " | " + extra_summary
    print(message, flush=True)
    return payload


def _mark_sync_point(
    *,
    phase: str,
    accelerator: Any | None,
    device: Any,
    torch_module: Any | None,
    memory_log_path: Path | None,
    extra: dict[str, Any] | None = None,
    main_process_only: bool = False,
) -> None:
    _emit_stage2_console_event(accelerator=accelerator, device=device, phase=f"before_{phase}", extra=extra, main_process_only=main_process_only)
    if torch_module is not None and memory_log_path is not None:
        _append_memory_event(artifact_path=memory_log_path, accelerator=accelerator, device=device, phase=f"before_{phase}", torch_module=torch_module, extra=extra)
    if accelerator is not None:
        accelerator.wait_for_everyone()
    if torch_module is not None and memory_log_path is not None:
        _append_memory_event(artifact_path=memory_log_path, accelerator=accelerator, device=device, phase=f"after_{phase}", torch_module=torch_module, extra=extra)
    _emit_stage2_console_event(accelerator=accelerator, device=device, phase=f"after_{phase}", extra=extra, main_process_only=main_process_only)



def _move_diagnostic_target_label(device: Any, dtype: Any | None) -> str:
    if dtype is None:
        return str(device)
    return f"{device} ({dtype})"



def _move_component_with_diagnostics(
    *,
    component: Any,
    component_name: str,
    action: str,
    device: Any,
    dtype: Any | None,
    torch_module: Any,
    accelerator: Any | None = None,
    runtime_device: Any | None = None,
    memory_log_path: Path | None = None,
    component_move_log_path: Path | None = None,
    epoch: int | None = None,
    global_step: int | None = None,
    optimizer_step: int | None = None,
    move_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    diagnostic_device = device if getattr(device, "type", None) == "cuda" else runtime_device
    event: dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "component_name": component_name,
        "action": action,
        "target_device": str(device),
        "target_dtype": str(dtype) if dtype is not None else None,
        "target": _move_diagnostic_target_label(device, dtype),
        "memory_before": _collect_cuda_memory_stats(torch_module, diagnostic_device),
    }
    if move_state is not None:
        move_state["last_component_move_attempt"] = event
    kwargs: dict[str, Any] = {"device": device}
    if dtype is not None and component_name != "image_encoder":
        kwargs["dtype"] = dtype
    try:
        component.to(**kwargs)
        event["status"] = "ok"
        event["memory_after"] = _collect_cuda_memory_stats(torch_module, diagnostic_device)
    except Exception as exc:
        event["status"] = "failed"
        event["memory_after"] = _collect_cuda_memory_stats(torch_module, diagnostic_device)
        event["error"] = {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        if move_state is not None:
            move_state["last_component_move_attempt"] = event
            move_state.setdefault("component_move_failures", []).append(event)
        if memory_log_path is not None and runtime_device is not None:
            _append_memory_event(
                artifact_path=memory_log_path,
                accelerator=accelerator,
                device=runtime_device,
                phase=f"component_move_failed:{component_name}:{action}",
                torch_module=torch_module,
                epoch=epoch,
                global_step=global_step,
                optimizer_step=optimizer_step,
                extra={"component_move": event},
            )
        if component_move_log_path is not None:
            _append_jsonl_event(component_move_log_path, event)
        raise RuntimeError(
            f"Failed during pipeline component move: component={component_name}, action={action}, target={event['target']}. Original error: {exc}"
        ) from exc
    if move_state is not None:
        move_state["last_component_move_attempt"] = event
        move_state.setdefault("component_move_events", []).append(event)
    if memory_log_path is not None and runtime_device is not None:
        _append_memory_event(
            artifact_path=memory_log_path,
            accelerator=accelerator,
            device=runtime_device,
            phase=f"component_move_ok:{component_name}:{action}",
            torch_module=torch_module,
            epoch=epoch,
            global_step=global_step,
            optimizer_step=optimizer_step,
            extra={"component_move": event},
        )
    if component_move_log_path is not None:
        _append_jsonl_event(component_move_log_path, event)
    return event



def _classify_training_failure(exc: Exception) -> str:
    message = str(exc).lower()
    if "non-finite" in message or "nan" in message or "inf" in message:
        return "non_finite_values"
    if "out of memory" in message:
        return "out_of_memory"
    return "runtime_error"


def _raise_on_non_finite_scalar(*, value: Any, name: str, accelerator: Any | None, device: Any, memory_log_path: Path | None, epoch: int, global_step: int, optimizer_step: int, torch_module: Any, extra: dict[str, Any] | None = None) -> float:
    scalar = float(value.detach().float().cpu().item()) if hasattr(value, "detach") else float(value)
    if math.isfinite(scalar):
        return scalar
    payload = {"name": name, "value": scalar, **(extra or {})}
    if memory_log_path is not None:
        _append_memory_event(artifact_path=memory_log_path, accelerator=accelerator, device=device, phase=f"non_finite_{name}", torch_module=torch_module, epoch=epoch, global_step=global_step, optimizer_step=optimizer_step, extra=payload)
    _emit_stage2_console_event(accelerator=accelerator, device=device, phase=f"non_finite_{name}", extra=payload, main_process_only=False)
    raise RuntimeError(f"Detected non-finite {name} during Stage 2 training: {scalar}")


def _collect_gradient_diagnostics(module: Any) -> dict[str, Any]:
    import torch

    grad_norm_sq = 0.0
    grad_parameter_count = 0
    grad_value_count = 0
    non_finite_gradient_parameter_names: list[str] = []
    sample_gradients: list[dict[str, Any]] = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        grad_parameter_count += 1
        grad_value_count += int(grad.numel())
        finite_mask = torch.isfinite(grad)
        if not bool(finite_mask.all().item()):
            non_finite_gradient_parameter_names.append(name)
        grad_norm_sq += float(torch.sum(torch.square(grad.float())).detach().cpu().item())
        if len(sample_gradients) < 8:
            sample_gradients.append({
                "name": name,
                "shape": list(parameter.shape),
                "grad_norm": float(torch.linalg.vector_norm(grad.float()).detach().cpu().item()),
                "grad_abs_max": float(grad.detach().abs().max().float().cpu().item()),
                "grad_is_finite": bool(finite_mask.all().item()),
            })
    return {
        "has_gradients": grad_parameter_count > 0,
        "grad_parameter_count": grad_parameter_count,
        "grad_value_count": grad_value_count,
        "grad_global_norm": math.sqrt(max(grad_norm_sq, 0.0)),
        "all_gradients_finite": not non_finite_gradient_parameter_names,
        "non_finite_gradient_parameter_count": len(non_finite_gradient_parameter_names),
        "non_finite_gradient_parameter_names_sample": non_finite_gradient_parameter_names[:20],
        "gradient_sample": sample_gradients,
    }


def _record_gradient_diagnostics(*, module: Any, accelerator: Any | None, device: Any, memory_log_path: Path | None, epoch: int, global_step: int, optimizer_step: int, torch_module: Any, force_console: bool = False) -> dict[str, Any]:
    diagnostics = _collect_gradient_diagnostics(module)
    if memory_log_path is not None:
        _append_memory_event(artifact_path=memory_log_path, accelerator=accelerator, device=device, phase="gradient_diagnostics", torch_module=torch_module, epoch=epoch, global_step=global_step, optimizer_step=optimizer_step, extra=diagnostics)
    if force_console or optimizer_step <= 3:
        _emit_stage2_console_event(accelerator=accelerator, device=device, phase="gradient_diagnostics", extra={"optimizer_step": optimizer_step, **diagnostics}, main_process_only=False)
    return diagnostics


def _collect_trainable_parameter_diagnostics(module: Any) -> dict[str, Any]:
    import torch

    non_finite_parameter_names: list[str] = []
    sample_parameters: list[dict[str, Any]] = []
    trainable_parameter_count = 0
    trainable_value_count = 0
    max_abs_value = 0.0
    dtype_counts: dict[str, int] = {}
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        values = parameter.detach()
        finite_mask = torch.isfinite(values)
        trainable_parameter_count += 1
        trainable_value_count += int(values.numel())
        dtype_label = _torch_dtype_label(values.dtype)
        dtype_counts[dtype_label] = dtype_counts.get(dtype_label, 0) + 1
        if not bool(finite_mask.all().item()):
            non_finite_parameter_names.append(name)
        if values.numel() > 0:
            max_abs_value = max(max_abs_value, float(values.abs().max().float().cpu().item()))
        if len(sample_parameters) < 8:
            sample_parameters.append({
                "name": name,
                "shape": list(parameter.shape),
                "dtype": dtype_label,
                "abs_max": float(values.abs().max().float().cpu().item()) if values.numel() > 0 else 0.0,
                "is_finite": bool(finite_mask.all().item()),
            })
    return {
        "trainable_parameter_count": trainable_parameter_count,
        "trainable_value_count": trainable_value_count,
        "all_trainable_parameters_finite": not non_finite_parameter_names,
        "non_finite_parameter_count": len(non_finite_parameter_names),
        "non_finite_parameter_names_sample": non_finite_parameter_names[:20],
        "max_abs_trainable_parameter_value": max_abs_value,
        "trainable_parameter_dtype_counts": dtype_counts,
        "trainable_parameter_sample": sample_parameters,
    }


def _assert_trainable_parameters_finite(*, module: Any, accelerator: Any | None, device: Any, memory_log_path: Path | None, epoch: int, global_step: int, optimizer_step: int, torch_module: Any, force_console: bool = False) -> dict[str, Any]:
    diagnostics = _collect_trainable_parameter_diagnostics(module)
    if memory_log_path is not None:
        _append_memory_event(artifact_path=memory_log_path, accelerator=accelerator, device=device, phase="trainable_parameter_diagnostics", torch_module=torch_module, epoch=epoch, global_step=global_step, optimizer_step=optimizer_step, extra=diagnostics)
    if force_console or optimizer_step <= 3 or not diagnostics["all_trainable_parameters_finite"]:
        _emit_stage2_console_event(accelerator=accelerator, device=device, phase="trainable_parameter_diagnostics", extra={"optimizer_step": optimizer_step, **diagnostics}, main_process_only=False)
    if not diagnostics["all_trainable_parameters_finite"]:
        raise RuntimeError(f"Detected non-finite trainable parameters immediately after optimizer step {optimizer_step}: {diagnostics['non_finite_parameter_names_sample']}")
    return diagnostics


def _move_stage2_nontransformer_modules_to_device(pipeline: Any, *, device: Any, train_dtype: Any) -> None:
    for component_name in ["vae", "text_encoder", "text_encoder_2", "image_encoder"]:
        component = getattr(pipeline, component_name, None)
        if component is None or not hasattr(component, "to"):
            continue
        kwargs: dict[str, Any] = {"device": device}
        if component_name != "image_encoder":
            kwargs["dtype"] = train_dtype
        component.to(**kwargs)



def _set_module_mode(module: Any, *, training: bool) -> None:
    if module is None or not hasattr(module, "train"):
        return
    module.train(training)



def _move_named_pipeline_components(
    pipeline: Any,
    *,
    component_names: list[str],
    device: Any,
    dtype: Any | None,
    torch_module: Any | None = None,
    accelerator: Any | None = None,
    runtime_device: Any | None = None,
    memory_log_path: Path | None = None,
    component_move_log_path: Path | None = None,
    epoch: int | None = None,
    global_step: int | None = None,
    optimizer_step: int | None = None,
    move_state: dict[str, Any] | None = None,
) -> None:
    for component_name in component_names:
        component = getattr(pipeline, component_name, None)
        if component is None or not hasattr(component, "to"):
            continue
        if torch_module is None:
            kwargs: dict[str, Any] = {"device": device}
            if dtype is not None and component_name != "image_encoder":
                kwargs["dtype"] = dtype
            component.to(**kwargs)
            continue
        action = "offload_to_cpu" if str(device).startswith("cpu") else "move_to_device"
        _move_component_with_diagnostics(
            component=component,
            component_name=component_name,
            action=action,
            device=device,
            dtype=dtype,
            torch_module=torch_module,
            accelerator=accelerator,
            runtime_device=runtime_device,
            memory_log_path=memory_log_path,
            component_move_log_path=component_move_log_path,
            epoch=epoch,
            global_step=global_step,
            optimizer_step=optimizer_step,
            move_state=move_state,
        )



def _enable_transformer_gradient_checkpointing(transformer: Any) -> dict[str, Any]:
    methods = [
        "enable_gradient_checkpointing",
        "gradient_checkpointing_enable",
    ]
    attempted_methods: list[str] = []
    for method_name in methods:
        method = getattr(transformer, method_name, None)
        if callable(method):
            attempted_methods.append(method_name)
            method()
            return {
                "enabled": True,
                "method": method_name,
                "attempted_methods": attempted_methods,
            }
    if hasattr(transformer, "gradient_checkpointing"):
        attempted_methods.append("gradient_checkpointing_attr")
        try:
            setattr(transformer, "gradient_checkpointing", True)
            return {
                "enabled": True,
                "method": "gradient_checkpointing_attr",
                "attempted_methods": attempted_methods,
            }
        except Exception:
            pass
    return {
        "enabled": False,
        "method": None,
        "attempted_methods": attempted_methods,
        "reason": "transformer_exposes_no_supported_gradient_checkpointing_interface",
    }





def _iter_named_parameters_from_pipeline_or_module(module_or_pipeline: Any):
    if hasattr(module_or_pipeline, "named_parameters"):
        yield from module_or_pipeline.named_parameters()
        return

    seen_parameter_ids: set[int] = set()
    component_names = ["transformer", "text_encoder", "text_encoder_2", "vae", "image_encoder"]
    for component_name in component_names:
        component = getattr(module_or_pipeline, component_name, None)
        if component is None or not hasattr(component, "named_parameters"):
            continue
        for name, parameter in component.named_parameters():
            parameter_id = id(parameter)
            if parameter_id in seen_parameter_ids:
                continue
            seen_parameter_ids.add(parameter_id)
            qualified_name = f"{component_name}.{name}" if name else component_name
            yield qualified_name, parameter


def _summarize_trainable_parameters(module_or_pipeline: Any) -> dict[str, Any]:
    trainable_names: list[str] = []
    frozen_names: list[str] = []
    lora_parameter_names: list[str] = []
    trainable_parameter_count = 0
    frozen_parameter_count = 0
    lora_parameter_count = 0
    for name, parameter in _iter_named_parameters_from_pipeline_or_module(module_or_pipeline):
        count = int(parameter.numel())
        if parameter.requires_grad:
            trainable_names.append(name)
            trainable_parameter_count += count
        else:
            frozen_names.append(name)
            frozen_parameter_count += count
        if ".lora_" in name or name.startswith("lora_"):
            lora_parameter_names.append(name)
            lora_parameter_count += count
    non_lora_trainable_names = [name for name in trainable_names if ".lora_" not in name and not name.startswith("lora_")]
    component_trainable_counts: dict[str, int] = {}
    component_frozen_counts: dict[str, int] = {}
    for name, parameter in _iter_named_parameters_from_pipeline_or_module(module_or_pipeline):
        component_name = name.split(".", 1)[0] if "." in name else name
        if parameter.requires_grad:
            component_trainable_counts[component_name] = component_trainable_counts.get(component_name, 0) + int(parameter.numel())
        else:
            component_frozen_counts[component_name] = component_frozen_counts.get(component_name, 0) + int(parameter.numel())
    return {
        "trainable_parameter_count": trainable_parameter_count,
        "frozen_parameter_count": frozen_parameter_count,
        "trainable_parameter_names_sample": _sample_sequence(trainable_names, limit=40),
        "frozen_parameter_names_sample": _sample_sequence(frozen_names, limit=40),
        "lora_parameter_count": lora_parameter_count,
        "lora_parameter_names_sample": _sample_sequence(lora_parameter_names, limit=40),
        "non_lora_trainable_parameter_names_sample": _sample_sequence(non_lora_trainable_names, limit=40),
        "only_lora_parameters_trainable": bool(trainable_names) and not non_lora_trainable_names,
        "summary_target_type": type(module_or_pipeline).__name__,
        "summary_from_pipeline_components": not hasattr(module_or_pipeline, "named_parameters"),
        "trainable_parameter_count_by_component": component_trainable_counts,
        "frozen_parameter_count_by_component": component_frozen_counts,
    }



def _save_transformer_checkpoint(transformer: Any, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(transformer, "save_pretrained"):
        transformer.save_pretrained(output_dir)
        return
    if importlib.util.find_spec("torch") is None:
        return
    import torch

    torch.save(transformer.state_dict(), output_dir / "pytorch_model.bin")



def _resolve_training_device(config: Stage2TrainConfig) -> Any:
    import torch

    if config.backbone_device:
        return torch.device(config.backbone_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")



def _resolve_training_dtype(config: Stage2TrainConfig, device: Any) -> Any:
    import torch

    normalized = str(config.backbone_torch_dtype).lower().strip()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported torch dtype label: {config.backbone_torch_dtype}")
    if device.type == "cpu" and mapping[normalized] != torch.float32:
        return torch.float32
    return mapping[normalized]



def _torch_dtype_label(dtype: Any) -> str:
    text = str(dtype)
    return text.split(".")[-1]



def _config_to_dict(config: Stage2TrainConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["backbone_assumptions"] = _infer_backbone_assumptions(config.backbone_name)
    payload["resolved_module_selection"] = resolve_effective_module_selection(config)
    return payload



def _build_component_plan(config: Stage2TrainConfig, backbone_runtime: dict[str, Any]) -> dict[str, Any]:
    selection = resolve_effective_module_selection(config)
    trainable_groups = selection["trainable_component_groups"]
    if str(config.training_parameterization).lower() == "lora" and config.trainable_component_groups == list(DEFAULT_TEXT_CONDITIONING_GROUPS):
        trainable_groups = list(DEFAULT_LORA_TARGET_GROUPS)
        selection = resolve_effective_module_selection(Stage2TrainConfig(**{**asdict(config), "trainable_component_groups": trainable_groups}))
    frozen_groups: list[str] = []
    if config.train_transformer_core_only:
        frozen_groups.append("non_transformer_top_level_modules")
    if config.freeze_text_encoder:
        frozen_groups.append("text_encoder")
    if config.freeze_vae:
        frozen_groups.append("vae")

    return {
        "focus": config.stage2_focus,
        "conditioning_objective": config.conditioning_objective,
        "fallback_if_oom": "restrict training to conditioning-related transformer submodules",
        "trainable_component_groups": trainable_groups,
        "unknown_trainable_component_groups": selection["unknown_trainable_component_groups"],
        "frozen_component_groups": frozen_groups,
        "module_selection": {
            "requested_include_patterns": list(config.module_include_patterns),
            "effective_include_patterns": selection["effective_include_patterns"],
            "exclude_patterns": selection["effective_exclude_patterns"],
            "selection_semantics": (
                "resolved_component_groups_plus_patterns_with_real_requires_grad_application"
                if selection["should_apply_real_transformer_selection"] or config.inspect_module_reference
                else "resolved_component_groups_plus_patterns_metadata_only"
            ),
        },
        "adapter_plan": asdict(config.adapter_plan),
        "backbone_assumptions": _infer_backbone_assumptions(config.backbone_name),
        "backbone_runtime": backbone_runtime,
        "implementation_boundary": (
            "Pattern selectors now resolve trainable component groups into real transformer-internal module patterns. "
            "The real training path supports both full real-parameter transformer updates and a conservative LoRA mode "
            "that injects adapters into selected conditioning-related linear sites while freezing base weights."
        ),
        "memory_strategy": {
            "enable_gradient_checkpointing": config.enable_gradient_checkpointing,
            "frozen_components_runtime": "always_on_device",
        },
    }



def _build_backbone_runtime_summary(config: Stage2TrainConfig) -> dict[str, Any]:
    selection = resolve_effective_module_selection(config)
    loader_name = "generic_python_loader" if not config.inspect_module_reference else None
    load_result = load_generative_backbone(
        config.backbone_name,
        loader=loader_name,
        allow_unimplemented=True,
        torch_dtype=config.backbone_torch_dtype,
        device=config.backbone_device,
        device_map=config.backbone_device_map,
        local_files_only=config.backbone_local_files_only,
        component=config.backbone_component,
    )
    summary: dict[str, Any] = {
        "backbone_name": config.backbone_name,
        "family": infer_backbone_family(config.backbone_name),
        "loader": load_result.loader_name,
        "loader_status": load_result.implementation_status,
        "loader_notes": list(load_result.notes or []),
        "resolved_module_name": load_result.resolved_module_name,
        "resolved_module_type": load_result.resolved_module_type,
        "module_reference": config.inspect_module_reference,
        "module_selection_applied": False,
        "adapter_injection_applied": False,
        "resolved_module_selection": selection,
        "module_targeting": None,
        "adapter_injection": None,
        "requested_component": config.backbone_component,
        "requested_torch_dtype": config.backbone_torch_dtype,
        "requested_device": config.backbone_device,
        "requested_device_map": config.backbone_device_map,
        "local_files_only": config.backbone_local_files_only,
    }

    if not config.inspect_module_reference and load_result.implementation_status != "loaded":
        summary["inspection_status"] = "not_requested"
        return summary

    try:
        if config.inspect_module_reference:
            module = load_module_from_reference(config.inspect_module_reference)
        else:
            real_load = load_real_backbone_module(
                config.backbone_name,
                torch_dtype=config.backbone_torch_dtype,
                device=config.backbone_device,
                device_map=config.backbone_device_map,
                local_files_only=config.backbone_local_files_only,
                component=config.backbone_component,
                allow_unimplemented=False,
            )
            module = real_load.module
            summary["loader"] = real_load.loader_name
            summary["loader_status"] = real_load.implementation_status
            summary["loader_notes"] = list(real_load.notes or [])
            summary["resolved_module_name"] = real_load.resolved_module_name
            summary["resolved_module_type"] = real_load.resolved_module_type
        if selection["should_apply_real_transformer_selection"]:
            targeting = apply_trainable_parameter_selection(
                module,
                include_patterns=selection["effective_include_patterns"],
                exclude_patterns=selection["effective_exclude_patterns"],
            )
            summary["module_selection_applied"] = True
        else:
            targeting = inspect_target_modules(
                module,
                include_patterns=selection["effective_include_patterns"],
                exclude_patterns=selection["effective_exclude_patterns"],
                limit=config.inspect_limit,
            )
        summary["inspection_status"] = "ok"
        summary["module_targeting"] = targeting.to_dict()

        if config.inject_adapters_on_real_module:
            adapter_plan = config.adapter_plan
            target_patterns = adapter_plan.target_module_patterns or selection["effective_include_patterns"]
            injection = inject_lora_adapters(
                module,
                include_patterns=target_patterns,
                exclude_patterns=adapter_plan.exclude_module_patterns,
                rank=adapter_plan.rank,
                alpha=adapter_plan.alpha,
                dropout=adapter_plan.dropout,
                adapter_dtype=adapter_plan.master_weight_dtype,
            )
            summary["adapter_injection_applied"] = True
            summary["adapter_injection"] = injection.to_dict()
            summary["module_targeting_after_adapter_injection"] = inspect_target_modules(
                module,
                include_patterns=selection["effective_include_patterns"],
                exclude_patterns=selection["effective_exclude_patterns"],
                limit=config.inspect_limit,
            ).to_dict()
    except Exception as exc:  # noqa: BLE001
        summary["inspection_status"] = "failed"
        summary["inspection_error"] = str(exc)

    return summary



def inspect_stage2_backbone_targets(
    *,
    backbone_name: str,
    module_reference: str | None,
    include_patterns: list[str],
    exclude_patterns: list[str] | None = None,
    limit: int = 200,
    apply_selection: bool = False,
    inject_adapters: bool = False,
    adapter_plan: AdapterPlan | None = None,
    load_backbone: bool = False,
    torch_dtype: str = "bfloat16",
    device: str | None = None,
    device_map: str | None = None,
    local_files_only: bool = False,
    component: str | None = None,
) -> dict[str, Any]:
    if load_backbone:
        try:
            real_load = load_real_backbone_module(
                backbone_name,
                torch_dtype=torch_dtype,
                device=device,
                device_map=device_map,
                local_files_only=local_files_only,
                component=component,
                allow_unimplemented=False,
            )
        except Exception as exc:  # noqa: BLE001
            return {
                "backbone_name": backbone_name,
                "backbone_family": infer_backbone_family(backbone_name),
                "module_reference": None,
                "loaded_backbone": True,
                "load_summary": {
                    "loader": "real_backbone_loader",
                    "loader_status": "failed",
                    "loader_error": str(exc),
                    "local_files_only": local_files_only,
                    "torch_dtype": torch_dtype,
                    "device": device,
                    "device_map": device_map,
                    "component": component,
                },
                "apply_selection": apply_selection,
                "targeting": None,
                "inject_adapters": False,
                "notes": [
                    "Real backbone load was attempted but did not complete.",
                    "This is a real runtime failure report, not a fake success.",
                ],
            }
        module = real_load.module
        module_reference_value = None
        load_summary = {
            "loader": real_load.loader_name,
            "loader_status": real_load.implementation_status,
            "loader_notes": list(real_load.notes or []),
            "resolved_module_name": real_load.resolved_module_name,
            "resolved_module_type": real_load.resolved_module_type,
            "local_files_only": local_files_only,
            "torch_dtype": torch_dtype,
            "device": device,
            "device_map": device_map,
            "component": component,
        }
    else:
        if not module_reference:
            raise ValueError("module_reference is required unless load_backbone=True")
        module = load_module_from_reference(module_reference)
        module_reference_value = module_reference
        load_summary = None
    if apply_selection:
        targeting = apply_trainable_parameter_selection(
            module,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )
    else:
        targeting = inspect_target_modules(
            module,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            limit=limit,
        )

    summary: dict[str, Any] = {
        "backbone_name": backbone_name,
        "backbone_family": infer_backbone_family(backbone_name),
        "module_reference": module_reference_value,
        "loaded_backbone": load_backbone,
        "load_summary": load_summary,
        "apply_selection": apply_selection,
        "targeting": targeting.to_dict(),
        "notes": [
            "This utility inspects either an explicitly provided torch module tree or a real loaded backbone component.",
            "It does not imply that full backbone loading or full Stage 2 training is wired for every backbone family.",
        ],
    }

    if inject_adapters:
        active_plan = adapter_plan or AdapterPlan(
            target_module_patterns=list(include_patterns),
            exclude_module_patterns=list(exclude_patterns or []),
        )
        injection = inject_lora_adapters(
            module,
            include_patterns=active_plan.target_module_patterns,
            exclude_patterns=active_plan.exclude_module_patterns,
            rank=active_plan.rank,
            alpha=active_plan.alpha,
            dropout=active_plan.dropout,
            adapter_dtype=active_plan.master_weight_dtype,
        )
        summary["inject_adapters"] = True
        summary["adapter_plan"] = asdict(active_plan)
        summary["adapter_injection"] = injection.to_dict()
        summary["targeting_after_adapter_injection"] = inspect_target_modules(
            module,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            limit=limit,
        ).to_dict()
    else:
        summary["inject_adapters"] = False

    return summary



def _build_trainer_plan(
    config: Stage2TrainConfig,
    manifest_path: str,
    num_pairs: int,
    component_plan: dict[str, Any],
) -> dict[str, Any]:
    return {
        "stage": "stage2_v1",
        "objective": "full-transformer fine-tuning of the selected generative backbone using real-image + Stage-1-canonical-caption pairs",
        "backbone_name": config.backbone_name,
        "manifest_path": str(Path(manifest_path).resolve()),
        "num_pairs": num_pairs,
        "optimizer_name": config.optimizer_name,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "max_steps": config.max_steps,
        "resolution": config.resolution,
        "weight_dtype": config.weight_dtype,
        "conditioning_batch_contract": {
            "image_field": "image",
            "text_field": "conditioning_text",
            "text_source": config.conditioning_text_field,
            "paired_from": "stage1_render.records.jsonl",
        },
        "component_plan": component_plan,
        "freeze_plan": {
            "train_transformer_core_only": config.train_transformer_core_only,
            "freeze_text_encoder": config.freeze_text_encoder,
            "freeze_vae": config.freeze_vae,
        },
        "implementation_status": {
            "pairing_manifest": "implemented",
            "text_conditioning_manifest_fields": "implemented",
            "run_directory_setup": "implemented",
            "config_snapshot": "implemented",
            "sdxl_lora_training": "delegated_to_official_diffusers_trainer",
        },
        "notes": [
            "Stage 2 is SDXL LoRA fine-tuning only (as of 2026-04-18).",
            "Training is delegated to the official diffusers train_text_to_image_lora_sdxl.py via a thin wrapper.",
        ],
    }



def _infer_backbone_assumptions(backbone_name: str) -> dict[str, Any]:
    family = infer_backbone_family(backbone_name)
    if family == "sdxl":
        return {
            "family": "sdxl",
            "notes": [
                "SDXL LoRA fine-tuning via the official diffusers trainer.",
            ],
        }
    return {
        "family": "generic_diffusion_backbone",
        "notes": [
            "Non-SDXL backbones are not supported; cspd-stage2 train will not attempt training.",
        ],
    }
