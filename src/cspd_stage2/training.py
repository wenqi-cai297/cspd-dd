from __future__ import annotations

"""Training utilities for CSPD Stage 2.

This module is deliberately honest about scope:
- it prepares run directories and paired manifests,
- records text-conditioning-focused adaptation intent,
- separates trainable and frozen component plans,
- implements a minimal accelerate-based real FLUX training path over (image, canonical_caption) pairs when the runtime supports it,
- keeps the older tiny placeholder loop as an explicit plumbing fallback,
- does not pretend every environment can actually load or fine-tune gated FLUX checkpoints.
"""

import importlib.util
import json
import os
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
import math
import re
from contextlib import nullcontext

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


DEFAULT_TEXT_CONDITIONING_GROUPS = [
    "full_transformer",
]

DEFAULT_LORA_TARGET_GROUPS = [
    "conditioning_transformer",
]

DEFAULT_EXCLUDE_PATTERNS = [
    "vae",
    "autoencoder",
    "decoder",
    "image_encoder",
]

CONDITIONING_RELATED_GROUP_PATTERNS = {
    "full_transformer": ["*"],
    "conditioning_context_embedder": [
        "context_embedder",
        "context_embedder.*",
    ],
    "conditioning_time_text_embed": [
        "time_text_embed*",
        "time_text_embed*.*",
    ],
    "conditioning_norm1_context": [
        "transformer_blocks.*.norm1_context*",
        "transformer_blocks.*.norm1_context*.*",
    ],
    "conditioning_added_kv_attention": [
        "transformer_blocks.*.attn.add_q_proj",
        "transformer_blocks.*.attn.add_k_proj",
        "transformer_blocks.*.attn.add_v_proj",
        "transformer_blocks.*.attn.to_add_out",
        "transformer_blocks.*.attn.to_add_out.*",
    ],
    "conditioning_ff_context": [
        "transformer_blocks.*.ff_context*",
        "transformer_blocks.*.ff_context*.*",
    ],
}
CONDITIONING_RELATED_GROUP_PATTERNS["conditioning_transformer"] = [
    pattern
    for group_name in [
        "conditioning_context_embedder",
        "conditioning_time_text_embed",
        "conditioning_norm1_context",
        "conditioning_added_kv_attention",
        "conditioning_ff_context",
    ]
    for pattern in CONDITIONING_RELATED_GROUP_PATTERNS[group_name]
]


SPLIT_ONLY_DATASET_ROOT_NAMES = {
    "train",
    "val",
    "valid",
    "validation",
    "test",
    "testing",
}


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
    dataset_label = derive_stage2_dataset_label(dataset_root)
    backbone_slug = sanitize_stage2_backbone_slug(backbone_name)
    return str(Path("runs") / "stage2" / "train" / dataset_label / backbone_slug / resolved_timestamp)


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
    task_note: str = "placeholder adapter config until a concrete backbone-specific training stack is integrated"


@dataclass(slots=True)
class Stage2TrainConfig:
    dataset_root: str
    render_input: str
    output_dir: str | None = None
    backbone_name: str = "black-forest-labs/FLUX.1-Kontext-dev"
    memory_log_artifact_name: str = "memory_diagnostics.jsonl"
    batch_size: int = 4
    learning_rate: float = 1e-4
    epochs: int = 1
    max_steps: int | None = None
    num_workers: int = 0
    resolution: int = 512
    seed: int = 42
    weight_dtype: str = "float16"
    optimizer_name: str = "adamw"
    log_every: int = 10
    save_every: int = 200
    max_train_samples: int | None = None
    class_name_map: str | None = None
    class_archetype_map: str | None = None
    verify_images: bool = False
    strict_pairing: bool = False
    dry_run: bool = False
    generate_manifest_only: bool = False
    allow_placeholder_loop: bool = False
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
    keep_frozen_modules_on_cpu_until_needed: bool = True
    offload_frozen_modules_after_step: bool = True


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

        if not config.generate_manifest_only and not config.dry_run:
            try:
                last_known_phase = "before_real_training"
                training_result = run_real_stage2_backbone_training(
                    config=config,
                    pairs=pairing.pairs,
                    run_dir=run_dir,
                    manifest_path=manifest_paths.manifest_path,
                )
                last_known_phase = training_result.get("last_known_phase", "after_real_training")
            except Exception as exc:  # noqa: BLE001
                last_known_phase = "real_training_exception"
                if config.allow_placeholder_loop:
                    training_result = run_placeholder_transformer_core_loop(config, manifest_paths.manifest_path)
                    training_result["real_training_error"] = str(exc)
                    training_result["real_training_traceback"] = traceback.format_exc()
                    training_result["last_known_phase"] = training_result.get("last_known_phase") or last_known_phase
                    training_result["message"] = (
                        "Real Stage 2 backbone training could not run in this environment; fell back to the explicit placeholder loop."
                    )
                else:
                    training_result = {
                        "status": "failed_before_training",
                        "implemented_training": False,
                        "placeholder_training": False,
                        "message": (
                            "Real Stage 2 backbone training was attempted but could not start or complete in this environment. "
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



def _effective_prompt_max_sequence_length(config: Stage2TrainConfig) -> int:
    family = infer_backbone_family(config.backbone_name)
    if family in {"pixart", "pixart_sigma"}:
        return 300
    return 512

def _safe_write_json(path: Path, payload: dict[str, Any]) -> None:
    try:
        write_json(path, payload)
    except Exception:
        fallback = _safe_jsonable(payload)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(fallback, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")



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
        "definition": "full-transformer fine-tuning of the selected generative backbone with Stage 1 canonical-caption conditioning",
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
            "component_move_diagnostics_pattern": str((run_dir / "rank*_component_move_diagnostics.jsonl").resolve()),
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
    message = f"[Stage2][rank {rank_info['global_rank']}/{max(rank_info['world_size'] - 1, 0)}][{rank_info['device']}] {phase}"
    if extra:
        compact_items = []
        for key, value in list(_safe_jsonable(extra).items())[:6]:
            compact_items.append(f"{key}={value}")
        if compact_items:
            message += " | " + ", ".join(compact_items)
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



def run_real_stage2_backbone_training(
    *,
    config: Stage2TrainConfig,
    pairs: list[Any],
    run_dir: Path,
    manifest_path: str,
) -> dict[str, Any]:
    family = infer_backbone_family(config.backbone_name)
    if family in {"pixart_sigma", "pixart"}:
        return run_real_stage2_pixart_training(
            config=config,
            pairs=pairs,
            run_dir=run_dir,
            manifest_path=manifest_path,
        )
    return run_real_stage2_flux_training(
        config=config,
        pairs=pairs,
        run_dir=run_dir,
        manifest_path=manifest_path,
    )


def run_real_stage2_flux_training(
    *,
    config: Stage2TrainConfig,
    pairs: list[Any],
    run_dir: Path,
    manifest_path: str,
) -> dict[str, Any]:
    accelerator = None
    device = "unknown"
    memory_log_path: Path | None = None
    torch = None
    transformer = None
    checkpoint_dir = run_dir / "checkpoints"
    final_checkpoint_dir = checkpoint_dir / "final_transformer"
    is_main_process = True
    load_dtype_label: str | None = None
    train_dtype_label: str | None = None
    selection_result: dict[str, Any] | None = None
    gradient_checkpointing: dict[str, Any] = {
        "enabled": False,
        "method": None,
        "attempted_methods": [],
        "reason": "not_initialized",
    }
    logs: list[dict[str, Any]] = []
    losses: list[float] = []
    global_step = 0
    optimizer_step_count = 0
    steps_per_epoch = None
    total_optimizer_steps = None
    last_known_phase = "preflight"
    phase_history: list[str] = []
    component_move_state: dict[str, Any] = {"component_move_events": [], "component_move_failures": [], "last_component_move_attempt": None}

    def mark_phase(phase: str, *, extra: dict[str, Any] | None = None, epoch: int | None = None, global_step_value: int | None = None, optimizer_step_value: int | None = None, main_process_only: bool = True) -> None:
        nonlocal last_known_phase
        last_known_phase = phase
        phase_history.append(phase)
        if torch is not None and memory_log_path is not None:
            _append_memory_event(
                artifact_path=memory_log_path,
                accelerator=accelerator,
                device=device,
                phase=phase,
                torch_module=torch,
                epoch=epoch,
                global_step=global_step_value,
                optimizer_step=optimizer_step_value,
                extra=extra,
            )
        _emit_stage2_console_event(
            accelerator=accelerator,
            device=device,
            phase=phase,
            extra={"epoch": epoch, "global_step": global_step_value, "optimizer_step": optimizer_step_value, **(extra or {})},
            main_process_only=main_process_only,
        )

    try:
        if importlib.util.find_spec("torch") is None:
            raise RuntimeError("PyTorch is not installed")
        if importlib.util.find_spec("diffusers") is None:
            raise RuntimeError("diffusers is not installed")
        if config.use_accelerate and importlib.util.find_spec("accelerate") is None:
            raise RuntimeError("accelerate is not installed")

        import torch as torch_module
        torch = torch_module
        if config.use_accelerate:
            from accelerate import Accelerator
            from accelerate.utils import set_seed

        if not pairs:
            raise ValueError("No paired training samples were available after manifest generation")

        if config.use_accelerate:
            accelerator = Accelerator(
                gradient_accumulation_steps=max(config.gradient_accumulation_steps, 1),
            )
            set_seed(config.seed)
            device = accelerator.device
        else:
            torch.manual_seed(config.seed)
            device = _resolve_training_device(config)
        load_dtype = _resolve_training_dtype(config, device)
        train_dtype = torch.float32 if device.type == "cpu" else load_dtype
        load_dtype_label = _torch_dtype_label(load_dtype)
        train_dtype_label = _torch_dtype_label(train_dtype)
        rank_info = _accelerator_rank_info(accelerator, device)
        memory_log_path = run_dir / f"rank{rank_info['global_rank']:02d}_{config.memory_log_artifact_name}"
        component_move_log_path = run_dir / f"rank{rank_info['global_rank']:02d}_component_move_diagnostics.jsonl"
        _mark_sync_point(phase="accelerator_startup_barrier", accelerator=accelerator, device=device, torch_module=torch, memory_log_path=memory_log_path, extra={"backbone_name": config.backbone_name}, main_process_only=False)
        mark_phase(
            "training_start",
            extra={
                "backbone_name": config.backbone_name,
                "manifest_path": str(Path(manifest_path).resolve()),
                "num_pairs": len(pairs),
                "load_dtype": load_dtype_label,
                "train_dtype": train_dtype_label,
            },
        )

        requested_device_for_load = None if config.use_accelerate else str(device)
        requested_device_map = None if config.use_accelerate else config.backbone_device_map

        mark_phase("before_backbone_load")
        backbone = load_real_backbone_module(
            config.backbone_name,
            torch_dtype=load_dtype_label,
            device=requested_device_for_load,
            device_map=requested_device_map,
            local_files_only=config.backbone_local_files_only,
            component=None,
            allow_unimplemented=False,
        )
        pipeline = backbone.root_module
        if pipeline is None:
            raise RuntimeError("Real backbone load did not return a pipeline root module")
        mark_phase(
            "after_backbone_load",
            extra={
                "resolved_module_name": backbone.resolved_module_name,
                "resolved_module_type": backbone.resolved_module_type,
                "loader": backbone.loader_name,
                "loader_status": backbone.implementation_status,
            },
        )

        mark_phase("before_freeze_selection")
        selection_result = _freeze_stage2_modules(pipeline, config)
        mark_phase(
            "after_freeze_selection",
            extra={
                "applied_transformer_module_selection": selection_result["selection"].to_dict(),
                "training_parameterization": config.training_parameterization,
                "adapter_injection": selection_result["adapter_injection"].to_dict() if selection_result["adapter_injection"] is not None else None,
                "trainable_parameter_summary": selection_result["trainable_parameter_summary"],
            },
        )
        transformer = pipeline.transformer
        transformer.train()
        gradient_checkpointing = {
            "enabled": False,
            "method": None,
            "attempted_methods": [],
            "reason": "disabled_by_config",
        }
        if config.enable_gradient_checkpointing:
            gradient_checkpointing = _enable_transformer_gradient_checkpointing(transformer)
        _set_module_mode(getattr(pipeline, "vae", None), training=False)
        _set_module_mode(getattr(pipeline, "text_encoder", None), training=False)
        _set_module_mode(getattr(pipeline, "text_encoder_2", None), training=False)
        _set_module_mode(getattr(pipeline, "image_encoder", None), training=False)
        if config.keep_frozen_modules_on_cpu_until_needed:
            _move_named_pipeline_components(
                pipeline,
                component_names=["vae", "text_encoder", "text_encoder_2", "image_encoder"],
                device=torch.device("cpu"),
                dtype=torch.float32,
                torch_module=torch,
                accelerator=accelerator,
                runtime_device=device,
                memory_log_path=memory_log_path,
                component_move_log_path=component_move_log_path,
                move_state=component_move_state,
            )
        mark_phase(
            "after_gradient_checkpointing_setup",
            extra={
                "gradient_checkpointing": gradient_checkpointing,
                "keep_frozen_modules_on_cpu_until_needed": config.keep_frozen_modules_on_cpu_until_needed,
                "offload_frozen_modules_after_step": config.offload_frozen_modules_after_step,
            },
        )

        mark_phase("before_optimizer_setup")
        optimizer = torch.optim.AdamW(
            (parameter for parameter in transformer.parameters() if parameter.requires_grad),
            lr=config.learning_rate,
        )
        mark_phase("after_optimizer_setup")

        mark_phase("before_dataloader_setup")
        dataloader = make_stage2_dataloader(
            pairs,
            resolution=config.resolution,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=True,
            drop_last=config.dataloader_drop_last,
        )
        mark_phase("after_dataloader_setup")

        if accelerator is not None:
            mark_phase("before_accelerate_prepare_transformer")
            transformer = accelerator.prepare(transformer)
            pipeline.transformer = transformer
            mark_phase("after_accelerate_prepare_transformer")

            mark_phase("before_accelerate_prepare_optimizer")
            optimizer = accelerator.prepare(optimizer)
            mark_phase("after_accelerate_prepare_optimizer")

            mark_phase("before_accelerate_prepare_dataloader")
            dataloader = accelerator.prepare(dataloader)
            mark_phase(
                "after_accelerate_prepare_dataloader",
                extra={
                    "batch_size": config.batch_size,
                    "num_workers": config.num_workers,
                    "gradient_accumulation_steps": max(config.gradient_accumulation_steps, 1),
                    "dataloader_batches_per_epoch": len(dataloader) if hasattr(dataloader, "__len__") else None,
                    "gradient_checkpointing": gradient_checkpointing,
                    "keep_frozen_modules_on_cpu_until_needed": config.keep_frozen_modules_on_cpu_until_needed,
                },
            )
        else:
            mark_phase(
                "after_accelerate_prepare_skipped",
                extra={
                    "reason": "accelerate_disabled",
                    "batch_size": config.batch_size,
                    "num_workers": config.num_workers,
                },
            )

        is_main_process = accelerator.is_main_process if accelerator is not None else True
        if is_main_process:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        _mark_sync_point(phase="checkpoint_dir_ready_barrier", accelerator=accelerator, device=device, torch_module=torch, memory_log_path=memory_log_path, extra={"checkpoint_dir": str(checkpoint_dir.resolve())}, main_process_only=False)
        mark_phase("after_checkpoint_dir_ready", extra={"checkpoint_dir": str(checkpoint_dir.resolve())})

        stop_after = config.max_steps if config.max_steps is not None else None
        steps_per_epoch = len(dataloader) if hasattr(dataloader, "__len__") else None
        if steps_per_epoch not in (0, None):
            optimizer_updates_per_epoch = max(
                math.ceil(steps_per_epoch / max(config.gradient_accumulation_steps, 1)),
                1,
            )
            total_optimizer_steps = optimizer_updates_per_epoch * max(config.epochs, 1)
            if stop_after is not None:
                total_optimizer_steps = min(total_optimizer_steps, stop_after)

        for epoch in range(max(config.epochs, 1)):
            mark_phase("epoch_start", epoch=epoch + 1, extra={"epoch_index": epoch + 1, "planned_epochs": max(config.epochs, 1)})
            for batch_index, batch in enumerate(dataloader, start=1):
                if epoch == 0 and batch_index == 1:
                    mark_phase("first_batch_fetched", epoch=epoch + 1, global_step_value=global_step + 1, optimizer_step_value=optimizer_step_count + 1, extra={"batch_index": batch_index, "batch_size": len(batch.get("conditioning_text", [])) if isinstance(batch, dict) else None}, main_process_only=False)
                if stop_after is not None and optimizer_step_count >= stop_after:
                    break
                loss = _run_real_flux_train_step(
                    pipeline=pipeline,
                    transformer=transformer,
                    batch=batch,
                    optimizer=optimizer,
                    accelerator=accelerator,
                    device=device,
                    train_dtype=train_dtype,
                    resolution=config.resolution,
                    memory_log_path=memory_log_path,
                    component_move_log_path=component_move_log_path,
                    epoch=epoch + 1,
                    global_step=global_step + 1,
                    optimizer_step=optimizer_step_count + 1,
                    keep_frozen_modules_on_cpu_until_needed=config.keep_frozen_modules_on_cpu_until_needed,
                    offload_frozen_modules_after_step=config.offload_frozen_modules_after_step,
                    move_state=component_move_state,
                )
                global_step += 1
                sync_gradients = accelerator.sync_gradients if accelerator is not None else True
                if sync_gradients:
                    optimizer_step_count += 1
                    if accelerator is not None:
                        loss_value = float(accelerator.gather_for_metrics(loss.detach().reshape(1)).mean().item())
                    else:
                        loss_value = float(loss.detach().cpu().item())
                    losses.append(loss_value)
                    if optimizer_step_count == 1 or optimizer_step_count % max(config.log_every, 1) == 0:
                        logs.append({"step": optimizer_step_count, "epoch": epoch + 1, "loss": loss_value})
                        mark_phase("optimizer_step_complete", epoch=epoch + 1, global_step_value=global_step, optimizer_step_value=optimizer_step_count, extra={"loss": loss_value}, main_process_only=True)
                    if is_main_process and optimizer_step_count % max(config.save_every, 1) == 0:
                        checkpoint_step_dir = checkpoint_dir / f"step_{optimizer_step_count:06d}"
                        mark_phase("before_checkpoint_save", epoch=epoch + 1, global_step_value=global_step, optimizer_step_value=optimizer_step_count, extra={"checkpoint_dir": str(checkpoint_step_dir.resolve())})
                        checkpoint_model = accelerator.unwrap_model(transformer) if accelerator is not None else transformer
                        _save_transformer_checkpoint(
                            checkpoint_model,
                            checkpoint_step_dir,
                        )
                        mark_phase("after_checkpoint_save", epoch=epoch + 1, global_step_value=global_step, optimizer_step_value=optimizer_step_count, extra={"checkpoint_dir": str(checkpoint_step_dir.resolve())})
            if stop_after is not None and optimizer_step_count >= stop_after:
                break

        _mark_sync_point(phase="training_complete_barrier", accelerator=accelerator, device=device, torch_module=torch, memory_log_path=memory_log_path, extra={"global_step": global_step, "optimizer_steps": optimizer_step_count}, main_process_only=False)
        mark_phase(
            "training_loop_complete",
            epoch=max(config.epochs, 1),
            global_step_value=global_step,
            optimizer_step_value=optimizer_step_count,
            extra={"loss_count": len(losses)},
        )
        if is_main_process:
            mark_phase("before_final_checkpoint_save", epoch=max(config.epochs, 1), global_step_value=global_step, optimizer_step_value=optimizer_step_count, extra={"checkpoint_dir": str(final_checkpoint_dir.resolve())})
            checkpoint_model = accelerator.unwrap_model(transformer) if accelerator is not None else transformer
            _save_transformer_checkpoint(checkpoint_model, final_checkpoint_dir)
            mark_phase("after_final_checkpoint_save", epoch=max(config.epochs, 1), global_step_value=global_step, optimizer_step_value=optimizer_step_count, extra={"checkpoint_dir": str(final_checkpoint_dir.resolve())})

        world_size = accelerator.num_processes if accelerator is not None else 1
        launch_notes = [
            "Uses Hugging Face Accelerate for process setup, dataloader sharding, backward, and main-process-only checkpoint writes.",
            "Recommended launch is via: accelerate launch ... cspd-stage2 train ...",
        ]
        if config.backbone_device_map:
            launch_notes.append(
                "backbone_device_map is ignored during accelerate-managed training to avoid conflicting with multi-process device placement."
            )

        summary = {
            "status": "completed",
            "implemented_training": True,
            "placeholder_training": False,
            "message": "Completed a minimal accelerate-based real FLUX Stage 2 training run on (image, canonical_caption) pairs.",
            "component_plan_status": "real_training_ran",
            "manifest_path": str(Path(manifest_path).resolve()),
            "device": str(device),
            "load_dtype": load_dtype_label,
            "train_dtype": train_dtype_label,
            "training_parameterization": config.training_parameterization,
            "applied_transformer_module_selection": selection_result["selection"].to_dict() if selection_result is not None else None,
            "adapter_injection": selection_result["adapter_injection"].to_dict() if selection_result is not None and selection_result["adapter_injection"] is not None else None,
            "trainable_parameter_summary": selection_result["trainable_parameter_summary"] if selection_result is not None else None,
            "accelerate": {
                "enabled": bool(accelerator is not None),
                "num_processes": world_size,
                "gradient_accumulation_steps": max(config.gradient_accumulation_steps, 1),
                "distributed_type": str(getattr(getattr(accelerator, "state", None), "distributed_type", "no")) if accelerator is not None else "no",
                "requested_device_map_ignored": bool(config.backbone_device_map),
            },
            "gradient_checkpointing": gradient_checkpointing,
            "memory_strategy": {
                "keep_frozen_modules_on_cpu_until_needed": config.keep_frozen_modules_on_cpu_until_needed,
                "offload_frozen_modules_after_step": config.offload_frozen_modules_after_step,
            },
            "dataloader_batches_per_epoch": steps_per_epoch,
            "forward_steps": global_step,
            "optimizer_steps": optimizer_step_count,
            "steps": optimizer_step_count,
            "epochs": max(config.epochs, 1),
            "num_pairs": len(pairs),
            "losses": losses,
            "logs": logs,
            "estimated_total_optimizer_steps": total_optimizer_steps,
            "final_checkpoint_dir": str(final_checkpoint_dir.resolve()),
            "memory_log_path": str(memory_log_path.resolve()) if memory_log_path is not None else None,
            "component_move_log_path": str(component_move_log_path.resolve()) if component_move_log_path is not None else None,
            "last_component_move_attempt": component_move_state.get("last_component_move_attempt"),
            "component_move_failures": component_move_state.get("component_move_failures", []),
            "last_known_phase": last_known_phase,
            "phase_history": phase_history,
            "launch_notes": launch_notes,
        }
        if is_main_process:
            _safe_write_json(run_dir / "training_metrics.json", summary)
        if accelerator is not None:
            accelerator.wait_for_everyone()
        return summary
    except Exception as exc:  # noqa: BLE001
        failure_summary = {
            "status": "failed_during_real_training",
            "implemented_training": False,
            "placeholder_training": False,
            "message": "Real Stage 2 FLUX training failed after entering the real training path.",
            "component_plan_status": "real_training_attempted",
            "manifest_path": str(Path(manifest_path).resolve()),
            "device": str(device),
            "load_dtype": load_dtype_label,
            "train_dtype": train_dtype_label,
            "training_parameterization": config.training_parameterization,
            "applied_transformer_module_selection": selection_result["selection"].to_dict() if selection_result is not None else None,
            "adapter_injection": selection_result["adapter_injection"].to_dict() if selection_result is not None and selection_result["adapter_injection"] is not None else None,
            "trainable_parameter_summary": selection_result["trainable_parameter_summary"] if selection_result is not None else None,
            "gradient_checkpointing": gradient_checkpointing,
            "memory_strategy": {
                "keep_frozen_modules_on_cpu_until_needed": config.keep_frozen_modules_on_cpu_until_needed,
                "offload_frozen_modules_after_step": config.offload_frozen_modules_after_step,
            },
            "dataloader_batches_per_epoch": steps_per_epoch,
            "forward_steps": global_step,
            "optimizer_steps": optimizer_step_count,
            "steps": optimizer_step_count,
            "epochs": max(config.epochs, 1),
            "num_pairs": len(pairs),
            "losses": losses,
            "logs": logs,
            "estimated_total_optimizer_steps": total_optimizer_steps,
            "final_checkpoint_dir": str(final_checkpoint_dir.resolve()),
            "memory_log_path": str(memory_log_path.resolve()) if memory_log_path is not None else None,
            "component_move_log_path": str(component_move_log_path.resolve()) if component_move_log_path is not None else None,
            "last_component_move_attempt": component_move_state.get("last_component_move_attempt"),
            "component_move_failures": component_move_state.get("component_move_failures", []),
            "last_known_phase": last_known_phase,
            "phase_history": phase_history,
            "failure": {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "rank_info": _accelerator_rank_info(accelerator, device) if torch is not None else None,
            },
        }
        if is_main_process:
            _safe_write_json(run_dir / "training_metrics.json", failure_summary)
        return failure_summary


def run_real_stage2_pixart_training(
    *,
    config: Stage2TrainConfig,
    pairs: list[Any],
    run_dir: Path,
    manifest_path: str,
) -> dict[str, Any]:
    accelerator = None
    device = "unknown"
    memory_log_path: Path | None = None
    torch = None
    transformer = None
    checkpoint_dir = run_dir / "checkpoints"
    final_checkpoint_dir = checkpoint_dir / "final_transformer"
    is_main_process = True
    load_dtype_label: str | None = None
    train_dtype_label: str | None = None
    selection_result: dict[str, Any] | None = None
    gradient_checkpointing: dict[str, Any] = {"enabled": False, "method": None, "attempted_methods": [], "reason": "not_initialized"}
    logs: list[dict[str, Any]] = []
    losses: list[float] = []
    global_step = 0
    optimizer_step_count = 0
    steps_per_epoch = None
    total_optimizer_steps = None
    last_known_phase = "preflight"
    phase_history: list[str] = []

    def mark_phase(phase: str, *, extra: dict[str, Any] | None = None, epoch: int | None = None, global_step_value: int | None = None, optimizer_step_value: int | None = None, main_process_only: bool = True) -> None:
        nonlocal last_known_phase
        last_known_phase = phase
        phase_history.append(phase)
        if torch is not None and memory_log_path is not None:
            _append_memory_event(artifact_path=memory_log_path, accelerator=accelerator, device=device, phase=phase, torch_module=torch, epoch=epoch, global_step=global_step_value, optimizer_step=optimizer_step_value, extra=extra)
        _emit_stage2_console_event(accelerator=accelerator, device=device, phase=phase, extra={"epoch": epoch, "global_step": global_step_value, "optimizer_step": optimizer_step_value, **(extra or {})}, main_process_only=main_process_only)

    try:
        if importlib.util.find_spec("torch") is None:
            raise RuntimeError("PyTorch is not installed")
        if importlib.util.find_spec("diffusers") is None:
            raise RuntimeError("diffusers is not installed")
        if config.use_accelerate and importlib.util.find_spec("accelerate") is None:
            raise RuntimeError("accelerate is not installed")

        import torch as torch_module
        torch = torch_module
        if config.use_accelerate:
            from accelerate import Accelerator
            from accelerate.utils import set_seed

        if not pairs:
            raise ValueError("No paired training samples were available after manifest generation")

        if config.use_accelerate:
            accelerator = Accelerator(gradient_accumulation_steps=max(config.gradient_accumulation_steps, 1))
            set_seed(config.seed)
            device = accelerator.device
        else:
            torch.manual_seed(config.seed)
            device = _resolve_training_device(config)
        load_dtype = _resolve_training_dtype(config, device)
        train_dtype = torch.float32 if device.type == "cpu" else load_dtype
        load_dtype_label = _torch_dtype_label(load_dtype)
        train_dtype_label = _torch_dtype_label(train_dtype)
        rank_info = _accelerator_rank_info(accelerator, device)
        memory_log_path = run_dir / f"rank{rank_info['global_rank']:02d}_{config.memory_log_artifact_name}"
        _mark_sync_point(phase="accelerator_startup_barrier", accelerator=accelerator, device=device, torch_module=torch, memory_log_path=memory_log_path, extra={"backbone_name": config.backbone_name}, main_process_only=False)
        mark_phase("training_start", extra={"backbone_name": config.backbone_name, "manifest_path": str(Path(manifest_path).resolve()), "num_pairs": len(pairs), "load_dtype": load_dtype_label, "train_dtype": train_dtype_label})

        requested_device_for_load = None if config.use_accelerate else str(device)
        requested_device_map = None if config.use_accelerate else config.backbone_device_map
        mark_phase("before_backbone_load")
        backbone = load_real_backbone_module(config.backbone_name, torch_dtype=load_dtype_label, device=requested_device_for_load, device_map=requested_device_map, local_files_only=config.backbone_local_files_only, component=None, allow_unimplemented=False)
        pipeline = backbone.root_module
        if pipeline is None:
            raise RuntimeError("Real backbone load did not return a pipeline root module")
        mark_phase("after_backbone_load", extra={"resolved_module_name": backbone.resolved_module_name, "resolved_module_type": backbone.resolved_module_type, "loader": backbone.loader_name, "loader_status": backbone.implementation_status})

        selection_result = _freeze_stage2_modules(pipeline, config)
        transformer = pipeline.transformer
        transformer.train()
        if config.enable_gradient_checkpointing:
            gradient_checkpointing = _enable_transformer_gradient_checkpointing(transformer)
        else:
            gradient_checkpointing = {"enabled": False, "method": None, "attempted_methods": [], "reason": "disabled_by_config"}
        _set_module_mode(getattr(pipeline, "vae", None), training=False)
        _set_module_mode(getattr(pipeline, "text_encoder", None), training=False)

        mark_phase("before_optimizer_setup")
        optimizer = torch.optim.AdamW((parameter for parameter in transformer.parameters() if parameter.requires_grad), lr=config.learning_rate)
        mark_phase("after_optimizer_setup")
        mark_phase("before_dataloader_setup")
        dataloader = make_stage2_dataloader(pairs, resolution=config.resolution, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=True, drop_last=config.dataloader_drop_last)
        mark_phase("after_dataloader_setup")
        if accelerator is not None:
            mark_phase("before_accelerate_prepare_transformer")
            transformer = accelerator.prepare(transformer)
            pipeline.transformer = transformer
            mark_phase("after_accelerate_prepare_transformer")
            mark_phase("before_accelerate_prepare_optimizer")
            optimizer = accelerator.prepare(optimizer)
            mark_phase("after_accelerate_prepare_optimizer")
            mark_phase("before_accelerate_prepare_dataloader")
            dataloader = accelerator.prepare(dataloader)
            mark_phase("after_accelerate_prepare_dataloader", extra={"dataloader_batches_per_epoch": len(dataloader) if hasattr(dataloader, "__len__") else None})
        is_main_process = accelerator.is_main_process if accelerator is not None else True
        if is_main_process:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        _mark_sync_point(phase="checkpoint_dir_ready_barrier", accelerator=accelerator, device=device, torch_module=torch, memory_log_path=memory_log_path, extra={"checkpoint_dir": str(checkpoint_dir.resolve())}, main_process_only=False)
        mark_phase("after_checkpoint_dir_ready", extra={"checkpoint_dir": str(checkpoint_dir.resolve())})

        stop_after = config.max_steps if config.max_steps is not None else None
        steps_per_epoch = len(dataloader) if hasattr(dataloader, "__len__") else None
        if steps_per_epoch not in (0, None):
            optimizer_updates_per_epoch = max(math.ceil(steps_per_epoch / max(config.gradient_accumulation_steps, 1)), 1)
            total_optimizer_steps = optimizer_updates_per_epoch * max(config.epochs, 1)
            if stop_after is not None:
                total_optimizer_steps = min(total_optimizer_steps, stop_after)

        for epoch in range(max(config.epochs, 1)):
            mark_phase("epoch_start", epoch=epoch + 1, extra={"epoch_index": epoch + 1, "planned_epochs": max(config.epochs, 1)})
            for batch_index, batch in enumerate(dataloader, start=1):
                if epoch == 0 and batch_index == 1:
                    mark_phase("first_batch_fetched", epoch=epoch + 1, global_step_value=global_step + 1, optimizer_step_value=optimizer_step_count + 1, extra={"batch_index": batch_index, "batch_size": len(batch.get("conditioning_text", [])) if isinstance(batch, dict) else None}, main_process_only=False)
                if stop_after is not None and optimizer_step_count >= stop_after:
                    break
                loss = _run_real_pixart_train_step(pipeline=pipeline, transformer=transformer, batch=batch, optimizer=optimizer, accelerator=accelerator, device=device, train_dtype=train_dtype, memory_log_path=memory_log_path, epoch=epoch + 1, global_step=global_step + 1, optimizer_step=optimizer_step_count + 1, config=config)
                global_step += 1
                sync_gradients = accelerator.sync_gradients if accelerator is not None else True
                if sync_gradients:
                    optimizer_step_count += 1
                    loss_value = float(accelerator.gather_for_metrics(loss.detach().reshape(1)).mean().item()) if accelerator is not None else float(loss.detach().cpu().item())
                    losses.append(loss_value)
                    if optimizer_step_count == 1 or optimizer_step_count % max(config.log_every, 1) == 0:
                        logs.append({"step": optimizer_step_count, "epoch": epoch + 1, "loss": loss_value})
                        mark_phase("optimizer_step_complete", epoch=epoch + 1, global_step_value=global_step, optimizer_step_value=optimizer_step_count, extra={"loss": loss_value}, main_process_only=True)
                    if is_main_process and optimizer_step_count % max(config.save_every, 1) == 0:
                        checkpoint_step_dir = checkpoint_dir / f"step_{optimizer_step_count:06d}"
                        mark_phase("before_checkpoint_save", epoch=epoch + 1, global_step_value=global_step, optimizer_step_value=optimizer_step_count, extra={"checkpoint_dir": str(checkpoint_step_dir.resolve())})
                        checkpoint_model = accelerator.unwrap_model(transformer) if accelerator is not None else transformer
                        _save_transformer_checkpoint(checkpoint_model, checkpoint_step_dir)
                        mark_phase("after_checkpoint_save", epoch=epoch + 1, global_step_value=global_step, optimizer_step_value=optimizer_step_count, extra={"checkpoint_dir": str(checkpoint_step_dir.resolve())})
            if stop_after is not None and optimizer_step_count >= stop_after:
                break

        _mark_sync_point(phase="training_complete_barrier", accelerator=accelerator, device=device, torch_module=torch, memory_log_path=memory_log_path, extra={"global_step": global_step, "optimizer_steps": optimizer_step_count}, main_process_only=False)
        if is_main_process:
            mark_phase("before_final_checkpoint_save", epoch=max(config.epochs, 1), global_step_value=global_step, optimizer_step_value=optimizer_step_count, extra={"checkpoint_dir": str(final_checkpoint_dir.resolve())})
            checkpoint_model = accelerator.unwrap_model(transformer) if accelerator is not None else transformer
            _save_transformer_checkpoint(checkpoint_model, final_checkpoint_dir)
            mark_phase("after_final_checkpoint_save", epoch=max(config.epochs, 1), global_step_value=global_step, optimizer_step_value=optimizer_step_count, extra={"checkpoint_dir": str(final_checkpoint_dir.resolve())})
        world_size = accelerator.num_processes if accelerator is not None else 1
        summary = {
            "status": "completed",
            "implemented_training": True,
            "placeholder_training": False,
            "message": "Completed a minimal accelerate-based real PixArt Stage 2 training run on (image, canonical_caption) pairs.",
            "component_plan_status": "real_training_ran",
            "manifest_path": str(Path(manifest_path).resolve()),
            "device": str(device),
            "load_dtype": load_dtype_label,
            "train_dtype": train_dtype_label,
            "training_parameterization": config.training_parameterization,
            "applied_transformer_module_selection": selection_result["selection"].to_dict() if selection_result is not None else None,
            "adapter_injection": selection_result["adapter_injection"].to_dict() if selection_result is not None and selection_result["adapter_injection"] is not None else None,
            "trainable_parameter_summary": selection_result["trainable_parameter_summary"] if selection_result is not None else None,
            "accelerate": {"enabled": bool(accelerator is not None), "num_processes": world_size, "gradient_accumulation_steps": max(config.gradient_accumulation_steps, 1), "distributed_type": str(getattr(getattr(accelerator, "state", None), "distributed_type", "no")) if accelerator is not None else "no", "requested_device_map_ignored": bool(config.backbone_device_map)},
            "gradient_checkpointing": gradient_checkpointing,
            "dataloader_batches_per_epoch": steps_per_epoch,
            "forward_steps": global_step,
            "optimizer_steps": optimizer_step_count,
            "steps": optimizer_step_count,
            "epochs": max(config.epochs, 1),
            "num_pairs": len(pairs),
            "losses": losses,
            "logs": logs,
            "estimated_total_optimizer_steps": total_optimizer_steps,
            "final_checkpoint_dir": str(final_checkpoint_dir.resolve()),
            "memory_log_path": str(memory_log_path.resolve()) if memory_log_path is not None else None,
            "last_known_phase": last_known_phase,
            "phase_history": phase_history,
            "launch_notes": ["Uses Hugging Face Accelerate for process setup, dataloader sharding, backward, and main-process-only checkpoint writes.", "PixArt-Σ uses the diffusers PixArtSigmaPipeline contract: prompt_embeds + prompt_attention_mask conditioning, VAE latents, and scheduler.add_noise training timesteps."],
        }
        if is_main_process:
            _safe_write_json(run_dir / "training_metrics.json", summary)
        if accelerator is not None:
            accelerator.wait_for_everyone()
        return summary
    except Exception as exc:  # noqa: BLE001
        failure_summary = {
            "status": "failed_during_real_training",
            "implemented_training": False,
            "placeholder_training": False,
            "message": "Real Stage 2 PixArt training failed after entering the real training path.",
            "component_plan_status": "real_training_attempted",
            "manifest_path": str(Path(manifest_path).resolve()),
            "device": str(device),
            "load_dtype": load_dtype_label,
            "train_dtype": train_dtype_label,
            "training_parameterization": config.training_parameterization,
            "applied_transformer_module_selection": selection_result["selection"].to_dict() if selection_result is not None else None,
            "adapter_injection": selection_result["adapter_injection"].to_dict() if selection_result is not None and selection_result["adapter_injection"] is not None else None,
            "trainable_parameter_summary": selection_result["trainable_parameter_summary"] if selection_result is not None else None,
            "gradient_checkpointing": gradient_checkpointing,
            "dataloader_batches_per_epoch": steps_per_epoch,
            "forward_steps": global_step,
            "optimizer_steps": optimizer_step_count,
            "steps": optimizer_step_count,
            "epochs": max(config.epochs, 1),
            "num_pairs": len(pairs),
            "losses": losses,
            "logs": logs,
            "estimated_total_optimizer_steps": total_optimizer_steps,
            "final_checkpoint_dir": str(final_checkpoint_dir.resolve()),
            "memory_log_path": str(memory_log_path.resolve()) if memory_log_path is not None else None,
            "last_known_phase": last_known_phase,
            "phase_history": phase_history,
            "failure": {"error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(), "rank_info": _accelerator_rank_info(accelerator, device) if torch is not None else None},
        }
        if is_main_process:
            _safe_write_json(run_dir / "training_metrics.json", failure_summary)
        return failure_summary


def _run_real_pixart_train_step(
    *,
    pipeline: Any,
    transformer: Any,
    batch: dict[str, Any],
    optimizer: Any,
    accelerator: Any | None,
    device: Any,
    train_dtype: Any,
    memory_log_path: Path,
    epoch: int,
    global_step: int,
    optimizer_step: int,
    config: Stage2TrainConfig,
) -> Any:
    import torch

    accumulation_context = accelerator.accumulate(transformer) if accelerator is not None else nullcontext()
    with accumulation_context:
        if global_step == 1:
            _emit_stage2_console_event(accelerator=accelerator, device=device, phase="before_first_vae_encode", extra={"optimizer_step": optimizer_step}, main_process_only=False)
        pixel_values = batch["pixel_values"].to(device=device, dtype=train_dtype)
        vae_dtype = next(pipeline.vae.parameters()).dtype
        vae_device = next(pipeline.vae.parameters()).device
        with torch.no_grad():
            latents = pipeline.vae.encode(pixel_values.to(device=vae_device, dtype=vae_dtype)).latent_dist.sample()
            if global_step == 1:
                _emit_stage2_console_event(accelerator=accelerator, device=device, phase="after_first_vae_encode", extra={"optimizer_step": optimizer_step, "latents_shape": list(latents.shape)}, main_process_only=False)
            scaling_factor = float(getattr(getattr(pipeline.vae, "config", None), "scaling_factor", 0.13025))
            shift_factor = float(getattr(getattr(pipeline.vae, "config", None), "shift_factor", 0.0) or 0.0)
            latents = (latents - shift_factor) * scaling_factor
            latents = latents.to(device=device, dtype=train_dtype)
            if global_step == 1:
                _emit_stage2_console_event(accelerator=accelerator, device=device, phase="before_first_text_encode", extra={"optimizer_step": optimizer_step}, main_process_only=False)
            prompt_embeds, prompt_attention_mask, _, _ = pipeline.encode_prompt(
                prompt=batch["conditioning_text"],
                do_classifier_free_guidance=False,
                device=device,
                num_images_per_prompt=1,
                max_sequence_length=_effective_prompt_max_sequence_length(config),
            )
            prompt_embeds = prompt_embeds.to(device=device, dtype=train_dtype)
            prompt_attention_mask = prompt_attention_mask.to(device=device)
            if global_step == 1:
                _emit_stage2_console_event(accelerator=accelerator, device=device, phase="after_first_text_encode", extra={"optimizer_step": optimizer_step, "prompt_embeds_shape": list(prompt_embeds.shape)}, main_process_only=False)

        noise = torch.randn_like(latents)
        batch_size = latents.shape[0]
        if not hasattr(pipeline.scheduler, "config") or not hasattr(pipeline.scheduler, "add_noise"):
            raise RuntimeError("Loaded PixArt scheduler does not expose add_noise for training")
        num_train_timesteps = int(getattr(pipeline.scheduler.config, "num_train_timesteps", 1000))
        timesteps = torch.randint(0, num_train_timesteps, (batch_size,), device=device, dtype=torch.long)
        noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps)
        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
        if global_step == 1:
            _emit_stage2_console_event(accelerator=accelerator, device=device, phase="before_first_forward", extra={"optimizer_step": optimizer_step}, main_process_only=False)
        model_output = transformer(hidden_states=noisy_latents, encoder_hidden_states=prompt_embeds, encoder_attention_mask=prompt_attention_mask, timestep=timesteps, added_cond_kwargs=added_cond_kwargs, return_dict=True)
        prediction = model_output.sample if hasattr(model_output, "sample") else model_output[0]
        out_channels = int(getattr(getattr(transformer, "config", None), "out_channels", prediction.shape[1]))
        latent_channels = int(getattr(getattr(transformer, "config", None), "in_channels", latents.shape[1]))
        if out_channels // 2 == latent_channels:
            prediction = prediction.chunk(2, dim=1)[0]
        loss = torch.nn.functional.mse_loss(prediction.float(), noise.float())
        if global_step == 1:
            _emit_stage2_console_event(accelerator=accelerator, device=device, phase="after_first_forward", extra={"optimizer_step": optimizer_step, "loss": float(loss.detach().float().cpu().item())}, main_process_only=False)
        optimizer.zero_grad(set_to_none=True)
        if accelerator is not None:
            accelerator.backward(loss)
            if global_step == 1:
                _emit_stage2_console_event(accelerator=accelerator, device=device, phase="after_first_backward", extra={"optimizer_step": optimizer_step}, main_process_only=False)
            if accelerator.sync_gradients:
                optimizer.step()
                if global_step == 1:
                    _emit_stage2_console_event(accelerator=accelerator, device=device, phase="after_first_optimizer_step", extra={"optimizer_step": optimizer_step}, main_process_only=False)
        else:
            loss.backward()
            if global_step == 1:
                _emit_stage2_console_event(accelerator=accelerator, device=device, phase="after_first_backward", extra={"optimizer_step": optimizer_step}, main_process_only=False)
            optimizer.step()
            if global_step == 1:
                _emit_stage2_console_event(accelerator=accelerator, device=device, phase="after_first_optimizer_step", extra={"optimizer_step": optimizer_step}, main_process_only=False)
        _append_memory_event(artifact_path=memory_log_path, accelerator=accelerator, device=device, phase="after_pixart_step", torch_module=torch, epoch=epoch, global_step=global_step, optimizer_step=optimizer_step, extra={"pixel_values_shape": list(batch['pixel_values'].shape), "latents_shape": list(latents.shape), "timesteps_shape": list(timesteps.shape), "prompt_embeds_shape": list(prompt_embeds.shape), "prompt_attention_mask_shape": list(prompt_attention_mask.shape)})
        return loss.detach()


def _run_real_flux_train_step(
    *,
    pipeline: Any,
    transformer: Any,
    batch: dict[str, Any],
    optimizer: Any,
    accelerator: Any | None,
    device: Any,
    train_dtype: Any,
    resolution: int,
    memory_log_path: Path,
    component_move_log_path: Path | None,
    epoch: int,
    global_step: int,
    optimizer_step: int,
    keep_frozen_modules_on_cpu_until_needed: bool,
    offload_frozen_modules_after_step: bool,
    move_state: dict[str, Any] | None,
) -> Any:
    import torch

    del resolution  # training uses the dataloader-prepared image tensor shape directly

    accumulation_context = accelerator.accumulate(transformer) if accelerator is not None else nullcontext()
    with accumulation_context:
        pixel_values = batch["pixel_values"].to(device=device, dtype=train_dtype)
        if global_step == 1:
            _emit_stage2_console_event(accelerator=accelerator, device=device, phase="before_first_vae_encode", extra={"optimizer_step": optimizer_step}, main_process_only=False)
        _append_memory_event(
            artifact_path=memory_log_path,
            accelerator=accelerator,
            device=device,
            phase="before_vae_encode",
            torch_module=torch,
            epoch=epoch,
            global_step=global_step,
            optimizer_step=optimizer_step,
            extra={
                "pixel_values_shape": list(pixel_values.shape),
                "conditioning_batch_size": len(batch.get("conditioning_text", [])),
                "keep_frozen_modules_on_cpu_until_needed": keep_frozen_modules_on_cpu_until_needed,
            },
        )
        with torch.no_grad():
            if keep_frozen_modules_on_cpu_until_needed:
                _move_named_pipeline_components(
                    pipeline,
                    component_names=["vae"],
                    device=device,
                    dtype=train_dtype,
                    torch_module=torch,
                    accelerator=accelerator,
                    runtime_device=device,
                    memory_log_path=memory_log_path,
                    component_move_log_path=component_move_log_path,
                    epoch=epoch,
                    global_step=global_step,
                    optimizer_step=optimizer_step,
                    move_state=move_state,
                )
                _append_memory_event(
                    artifact_path=memory_log_path,
                    accelerator=accelerator,
                    device=device,
                    phase="after_vae_move_to_device",
                    torch_module=torch,
                    epoch=epoch,
                    global_step=global_step,
                    optimizer_step=optimizer_step,
                )
            vae_dtype = next(pipeline.vae.parameters()).dtype
            vae_device = next(pipeline.vae.parameters()).device
            latents = pipeline.vae.encode(pixel_values.to(device=vae_device, dtype=vae_dtype)).latent_dist.sample()
            latents = (latents - pipeline.vae.config.shift_factor) * pipeline.vae.config.scaling_factor
            latents = latents.to(device=device, dtype=train_dtype)
            del pixel_values
            if global_step == 1:
                _emit_stage2_console_event(accelerator=accelerator, device=device, phase="after_first_vae_encode", extra={"optimizer_step": optimizer_step, "latents_shape": list(latents.shape)}, main_process_only=False)
            _append_memory_event(
                artifact_path=memory_log_path,
                accelerator=accelerator,
                device=device,
                phase="after_vae_encode",
                torch_module=torch,
                epoch=epoch,
                global_step=global_step,
                optimizer_step=optimizer_step,
                extra={"latents_shape": list(latents.shape)},
            )
            if keep_frozen_modules_on_cpu_until_needed and offload_frozen_modules_after_step:
                _move_named_pipeline_components(
                    pipeline,
                    component_names=["vae"],
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                    torch_module=torch,
                    accelerator=accelerator,
                    runtime_device=device,
                    memory_log_path=memory_log_path,
                    component_move_log_path=component_move_log_path,
                    epoch=epoch,
                    global_step=global_step,
                    optimizer_step=optimizer_step,
                    move_state=move_state,
                )
                _append_memory_event(
                    artifact_path=memory_log_path,
                    accelerator=accelerator,
                    device=device,
                    phase="after_vae_offload_to_cpu",
                    torch_module=torch,
                    epoch=epoch,
                    global_step=global_step,
                    optimizer_step=optimizer_step,
                )
            if keep_frozen_modules_on_cpu_until_needed:
                _move_named_pipeline_components(
                    pipeline,
                    component_names=["text_encoder", "text_encoder_2", "image_encoder"],
                    device=device,
                    dtype=train_dtype,
                    torch_module=torch,
                    accelerator=accelerator,
                    runtime_device=device,
                    memory_log_path=memory_log_path,
                    component_move_log_path=component_move_log_path,
                    epoch=epoch,
                    global_step=global_step,
                    optimizer_step=optimizer_step,
                    move_state=move_state,
                )
                _append_memory_event(
                    artifact_path=memory_log_path,
                    accelerator=accelerator,
                    device=device,
                    phase="after_prompt_modules_move_to_device",
                    torch_module=torch,
                    epoch=epoch,
                    global_step=global_step,
                    optimizer_step=optimizer_step,
                )
            if global_step == 1:
                _emit_stage2_console_event(accelerator=accelerator, device=device, phase="before_first_text_encode", extra={"optimizer_step": optimizer_step}, main_process_only=False)
            _append_memory_event(
                artifact_path=memory_log_path,
                accelerator=accelerator,
                device=device,
                phase="before_prompt_encode",
                torch_module=torch,
                epoch=epoch,
                global_step=global_step,
                optimizer_step=optimizer_step,
                extra={"prompt_sample": batch["conditioning_text"][0] if batch.get("conditioning_text") else None},
            )
            prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
                prompt=batch["conditioning_text"],
                prompt_2=batch["conditioning_text"],
                device=device,
                num_images_per_prompt=1,
                max_sequence_length=512,
            )
            prompt_embeds = prompt_embeds.to(device=device, dtype=train_dtype)
            pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=train_dtype)
            text_ids = text_ids.to(device=device, dtype=train_dtype)
            if global_step == 1:
                _emit_stage2_console_event(accelerator=accelerator, device=device, phase="after_first_text_encode", extra={"optimizer_step": optimizer_step, "prompt_embeds_shape": list(prompt_embeds.shape)}, main_process_only=False)
            _append_memory_event(
                artifact_path=memory_log_path,
                accelerator=accelerator,
                device=device,
                phase="after_prompt_encode",
                torch_module=torch,
                epoch=epoch,
                global_step=global_step,
                optimizer_step=optimizer_step,
                extra={
                    "prompt_embeds_shape": list(prompt_embeds.shape),
                    "pooled_prompt_embeds_shape": list(pooled_prompt_embeds.shape),
                    "text_ids_shape": list(text_ids.shape),
                },
            )
            if keep_frozen_modules_on_cpu_until_needed and offload_frozen_modules_after_step:
                _move_named_pipeline_components(
                    pipeline,
                    component_names=["text_encoder", "text_encoder_2", "image_encoder"],
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                    torch_module=torch,
                    accelerator=accelerator,
                    runtime_device=device,
                    memory_log_path=memory_log_path,
                    component_move_log_path=component_move_log_path,
                    epoch=epoch,
                    global_step=global_step,
                    optimizer_step=optimizer_step,
                    move_state=move_state,
                )
                _append_memory_event(
                    artifact_path=memory_log_path,
                    accelerator=accelerator,
                    device=device,
                    phase="after_prompt_modules_offload_to_cpu",
                    torch_module=torch,
                    epoch=epoch,
                    global_step=global_step,
                    optimizer_step=optimizer_step,
                )

        packed_latents = pipeline._pack_latents(
            latents,
            latents.shape[0],
            latents.shape[1],
            latents.shape[2],
            latents.shape[3],
        ).to(device=device, dtype=train_dtype)
        latent_image_ids = pipeline._prepare_latent_image_ids(
            latents.shape[0],
            latents.shape[2] // 2,
            latents.shape[3] // 2,
            device,
            train_dtype,
        )
        del latents

        noise = torch.randn_like(packed_latents)
        timesteps, sigmas = _sample_flux_flow_matching_timesteps(
            batch_size=packed_latents.shape[0],
            device=device,
            dtype=train_dtype,
        )
        noisy_latents = ((1.0 - sigmas) * packed_latents) + (sigmas * noise)
        target = noise - packed_latents

        guidance = None
        unwrapped_transformer = accelerator.unwrap_model(transformer) if accelerator is not None else transformer
        if getattr(getattr(unwrapped_transformer, "config", None), "guidance_embeds", False):
            guidance = torch.ones((packed_latents.shape[0],), device=device, dtype=torch.float32)

        if global_step == 1:
            _emit_stage2_console_event(accelerator=accelerator, device=device, phase="before_first_forward", extra={"optimizer_step": optimizer_step}, main_process_only=False)
        _append_memory_event(
            artifact_path=memory_log_path,
            accelerator=accelerator,
            device=device,
            phase="before_transformer_forward",
            torch_module=torch,
            epoch=epoch,
            global_step=global_step,
            optimizer_step=optimizer_step,
            extra={
                "packed_latents_shape": list(packed_latents.shape),
                "latent_image_ids_shape": list(latent_image_ids.shape),
                "timesteps_shape": list(timesteps.shape),
                "guidance_enabled": guidance is not None,
            },
        )
        model_output = transformer(
            hidden_states=noisy_latents,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            timestep=timesteps,
            img_ids=latent_image_ids,
            txt_ids=text_ids,
            guidance=guidance,
            return_dict=True,
        )
        prediction = model_output.sample if hasattr(model_output, "sample") else model_output[0]
        del model_output, prompt_embeds, pooled_prompt_embeds, text_ids
        loss = torch.nn.functional.mse_loss(prediction.float(), target.float())
        del prediction, target, noisy_latents, noise, packed_latents, latent_image_ids, timesteps, sigmas, guidance
        if global_step == 1:
            _emit_stage2_console_event(accelerator=accelerator, device=device, phase="after_first_forward", extra={"optimizer_step": optimizer_step, "loss": float(loss.detach().float().cpu().item())}, main_process_only=False)
        _append_memory_event(
            artifact_path=memory_log_path,
            accelerator=accelerator,
            device=device,
            phase="after_loss",
            torch_module=torch,
            epoch=epoch,
            global_step=global_step,
            optimizer_step=optimizer_step,
            extra={"loss": float(loss.detach().float().cpu().item())},
        )

        optimizer.zero_grad(set_to_none=True)
        _append_memory_event(
            artifact_path=memory_log_path,
            accelerator=accelerator,
            device=device,
            phase="before_backward",
            torch_module=torch,
            epoch=epoch,
            global_step=global_step,
            optimizer_step=optimizer_step,
        )
        if accelerator is not None:
            accelerator.backward(loss)
        else:
            loss.backward()
        if global_step == 1:
            _emit_stage2_console_event(accelerator=accelerator, device=device, phase="after_first_backward", extra={"optimizer_step": optimizer_step}, main_process_only=False)
        _append_memory_event(
            artifact_path=memory_log_path,
            accelerator=accelerator,
            device=device,
            phase="after_backward",
            torch_module=torch,
            epoch=epoch,
            global_step=global_step,
            optimizer_step=optimizer_step,
        )
        optimizer.step()
        if global_step == 1:
            _emit_stage2_console_event(accelerator=accelerator, device=device, phase="after_first_optimizer_step", extra={"optimizer_step": optimizer_step}, main_process_only=False)
    return loss.detach()



def _sample_flux_flow_matching_timesteps(*, batch_size: int, device: Any, dtype: Any) -> tuple[Any, Any]:
    import torch

    timesteps = torch.rand((batch_size,), device=device, dtype=torch.float32)
    sigmas = timesteps.to(device=device, dtype=dtype)
    while sigmas.ndim < 3:
        sigmas = sigmas.unsqueeze(-1)
    return timesteps, sigmas



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



def _resolve_trainable_component_groups(config: Stage2TrainConfig) -> list[str]:
    groups = list(dict.fromkeys(config.trainable_component_groups or DEFAULT_TEXT_CONDITIONING_GROUPS))
    return groups or list(DEFAULT_TEXT_CONDITIONING_GROUPS)



def _expand_component_group_patterns(groups: list[str]) -> tuple[list[str], list[str]]:
    include_patterns: list[str] = []
    unknown_groups: list[str] = []
    for group_name in groups:
        patterns = CONDITIONING_RELATED_GROUP_PATTERNS.get(group_name)
        if patterns is None:
            unknown_groups.append(group_name)
            continue
        include_patterns.extend(patterns)
    return list(dict.fromkeys(include_patterns)), unknown_groups



def resolve_effective_module_selection(config: Stage2TrainConfig) -> dict[str, Any]:
    groups = _resolve_trainable_component_groups(config)
    group_patterns, unknown_groups = _expand_component_group_patterns(groups)
    manual_patterns = list(dict.fromkeys(config.module_include_patterns or []))
    effective_include_patterns = list(dict.fromkeys(group_patterns + manual_patterns))
    family = infer_backbone_family(config.backbone_name)
    if family in {"pixart_sigma", "pixart"} and not manual_patterns and groups == ["conditioning_transformer"]:
        effective_include_patterns = [
            "caption_projection",
            "caption_projection.*",
            "adaln_single",
            "adaln_single.*",
            "transformer_blocks.*.attn1.*",
            "transformer_blocks.*.attn2.*",
            "transformer_blocks.*.ff.*",
        ]
    if not effective_include_patterns:
        effective_include_patterns = ["*"]
    effective_exclude_patterns = list(dict.fromkeys(config.module_exclude_patterns or []))
    selection_is_full_transformer = effective_include_patterns == ["*"]
    should_apply = bool(config.apply_real_module_selection or not selection_is_full_transformer)
    return {
        "trainable_component_groups": groups,
        "unknown_trainable_component_groups": unknown_groups,
        "group_resolved_include_patterns": group_patterns,
        "manual_include_patterns": manual_patterns,
        "effective_include_patterns": effective_include_patterns,
        "effective_exclude_patterns": effective_exclude_patterns,
        "selection_is_full_transformer": selection_is_full_transformer,
        "should_apply_real_transformer_selection": should_apply,
    }



def _freeze_stage2_modules(pipeline: Any, config: Stage2TrainConfig) -> dict[str, Any]:
    for component_name in ["transformer", "text_encoder", "text_encoder_2", "vae", "image_encoder"]:
        component = getattr(pipeline, component_name, None)
        if component is None or not hasattr(component, "parameters"):
            continue
        for parameter in component.parameters():
            parameter.requires_grad = False

    if not hasattr(pipeline, "transformer") or pipeline.transformer is None:
        raise RuntimeError("Loaded pipeline does not expose a transformer component")

    transformer = pipeline.transformer
    selection = resolve_effective_module_selection(config)
    parameterization = str(getattr(config, "training_parameterization", "full")).strip().lower()
    adapter_injection = None

    if parameterization not in {"full", "lora"}:
        raise ValueError(f"Unsupported training_parameterization: {config.training_parameterization}")

    if parameterization == "lora":
        if str(config.adapter_plan.adapter_type).lower() != "lora":
            raise ValueError("LoRA training_parameterization currently requires adapter_plan.adapter_type='lora'")
        adapter_plan = config.adapter_plan
        target_patterns = adapter_plan.target_module_patterns or selection["effective_include_patterns"]
        adapter_injection = inject_lora_adapters(
            transformer,
            include_patterns=target_patterns,
            exclude_patterns=adapter_plan.exclude_module_patterns,
            rank=adapter_plan.rank,
            alpha=adapter_plan.alpha,
            dropout=adapter_plan.dropout,
        )
        targeting = inspect_target_modules(
            transformer,
            include_patterns=target_patterns,
            exclude_patterns=adapter_plan.exclude_module_patterns,
            limit=None,
        )
    else:
        if config.train_transformer_core_only:
            for parameter in transformer.parameters():
                parameter.requires_grad = True
        if selection["should_apply_real_transformer_selection"]:
            targeting = apply_trainable_parameter_selection(
                transformer,
                include_patterns=selection["effective_include_patterns"],
                exclude_patterns=selection["effective_exclude_patterns"],
            )
        else:
            targeting = inspect_target_modules(
                transformer,
                include_patterns=selection["effective_include_patterns"],
                exclude_patterns=selection["effective_exclude_patterns"],
                limit=None,
            )

        if not config.freeze_text_encoder:
            for component_name in ["text_encoder", "text_encoder_2"]:
                component = getattr(pipeline, component_name, None)
                if component is not None and hasattr(component, "parameters"):
                    for parameter in component.parameters():
                        parameter.requires_grad = True
        if not config.freeze_vae and getattr(pipeline, "vae", None) is not None:
            for parameter in pipeline.vae.parameters():
                parameter.requires_grad = True

    return {
        "parameterization": parameterization,
        "selection": targeting,
        "adapter_injection": adapter_injection,
        "trainable_parameter_summary": _summarize_trainable_parameters(pipeline),
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
    return {
        "trainable_parameter_count": trainable_parameter_count,
        "frozen_parameter_count": frozen_parameter_count,
        "trainable_parameter_names": trainable_names,
        "frozen_parameter_names": frozen_names,
        "lora_parameter_count": lora_parameter_count,
        "lora_parameter_names": lora_parameter_names,
        "non_lora_trainable_parameter_names": non_lora_trainable_names,
        "only_lora_parameters_trainable": bool(trainable_names) and not non_lora_trainable_names,
        "summary_target_type": type(module_or_pipeline).__name__,
        "summary_from_pipeline_components": not hasattr(module_or_pipeline, "named_parameters"),
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



def run_placeholder_transformer_core_loop(config: Stage2TrainConfig, manifest_path: str) -> dict[str, Any]:
    """Optional tiny placeholder loop.

    This keeps the training surface honest: if torch is present, we can verify
    argument plumbing and a few optimizer steps, but we still do not pretend to
    be training FLUX Kontext itself.
    """
    if importlib.util.find_spec("torch") is None:
        return {
            "status": "placeholder_skipped",
            "implemented_training": False,
            "placeholder_training": False,
            "message": "PyTorch is not installed in this environment, so the optional placeholder loop was skipped.",
            "component_plan_status": "implemented_metadata_only",
        }

    import torch

    torch.manual_seed(config.seed)
    model = torch.nn.Linear(8, 8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    max_steps = config.max_steps or min(5, max(config.epochs, 1) * 2)
    losses: list[float] = []

    for _step in range(max_steps):
        inputs = torch.randn(config.batch_size, 8)
        targets = torch.randn(config.batch_size, 8)
        outputs = model(inputs)
        loss = torch.nn.functional.mse_loss(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))

    return {
        "status": "placeholder_complete",
        "implemented_training": False,
        "placeholder_training": True,
        "message": (
            "Ran a tiny PyTorch placeholder optimization loop to validate Stage 2 training plumbing. "
            "This is not FLUX Kontext fine-tuning."
        ),
        "manifest_path": str(Path(manifest_path).resolve()),
        "steps": max_steps,
        "losses": losses,
        "component_plan_status": "implemented_metadata_only",
    }



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
            "keep_frozen_modules_on_cpu_until_needed": config.keep_frozen_modules_on_cpu_until_needed,
            "offload_frozen_modules_after_step": config.offload_frozen_modules_after_step,
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
            "component_target_plan": "implemented_with_optional_real_module_inspection",
            "adapter_target_plan": "implemented_with_optional_real_module_lora_injection",
            "real_module_target_selection": "implemented_for_real_training_when_conditioning_submodule_path_is_selected",
            "real_module_adapter_injection": "optional_when_explicit_module_reference_is_provided",
            "placeholder_loop": "optional",
            "full_flux_kontext_finetuning": "minimally_implemented_when_runtime_supports_real_backbone_loading",
        },
        "notes": [
            "This scaffold is intentionally conservative.",
            "Stage 2 is canonical-caption-conditioned generative adaptation, not image-editing fine-tuning.",
            "Stage 2 no longer means render; render belongs to Stage 1.",
            "Current code records a default policy of freezing non-transformer top-level modules and fine-tuning the full transformer.",
            "When a conditioning-focused transformer submodule group is selected, the real training path now applies that selection to requires_grad before optimization.",
            "The real training path now attempts transformer gradient checkpointing when the loaded FLUX transformer exposes a supported interface.",
            "Frozen VAE/text components are kept on CPU until first use by default, then optionally offloaded back to CPU after encode so accelerate.prepare does not inherit their device residency up front.",
            "Real accelerate-based diffusers-backed FLUX-family training is wired conservatively around packed VAE latents and canonical-caption prompt encoding, but successful execution still depends on the local runtime actually loading the requested backbone.",
            "Heavier optimizer/state sharding or FSDP-style offload is still not implemented here.",
        ],
    }



def _infer_backbone_assumptions(backbone_name: str) -> dict[str, Any]:
    family = infer_backbone_family(backbone_name)
    if family == "flux_kontext":
        return {
            "family": "flux_kontext",
            "notes": [
                "Current target family is experimental FLUX.1 Kontext [dev].",
                "Default Stage 2 policy is to freeze non-transformer top-level modules and fine-tune the full FluxTransformer2DModel.",
                "If memory is insufficient, the intended fallback is conditioning-related transformer submodules only.",
            ],
        }
    if family in {"pixart_sigma", "pixart"}:
        return {
            "family": family,
            "notes": [
                "Current target family is PixArt text-to-image diffusion transformers via diffusers.",
                "Stage 2 semantics stay canonical-caption-conditioned generation on real images, not image editing.",
                "The practical first fallback for constrained hardware is transformer LoRA over selected PixArt attention/feed-forward modules.",
            ],
        }
    return {
        "family": "generic_diffusion_backbone",
        "notes": [
            "Stage 2 wording stays generic at the method level.",
            "Module-group selectors may need replacement for a different backbone family.",
        ],
    }
