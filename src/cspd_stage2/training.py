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
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
import math
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
from cspd_stage2.data import build_stage2_pairs, make_stage2_dataloader, write_pairing_artifacts


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
    output_dir: str
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
    conditioning_objective: str = "finetune_full_flux_transformer_on_real_image_and_stage1_canonical_caption_pairs"
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
    run_dir = Path(config.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

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

    manifest_paths = write_pairing_artifacts(pairing, run_dir)
    backbone_runtime = _build_backbone_runtime_summary(config)
    component_plan = _build_component_plan(config, backbone_runtime)
    trainer_plan = _build_trainer_plan(config, manifest_paths.manifest_path, len(pairing.pairs), component_plan)
    write_json(run_dir / "stage2_config_snapshot.json", _config_to_dict(config))
    write_json(run_dir / "trainer_plan.json", trainer_plan)

    training_result: dict[str, Any] = {
        "status": "manifest_ready",
        "implemented_training": False,
        "placeholder_training": False,
        "message": "Stage 2 paired manifest is ready.",
        "component_plan_status": "implemented_metadata_only",
    }

    if not config.generate_manifest_only and not config.dry_run:
        try:
            training_result = run_real_stage2_flux_training(
                config=config,
                pairs=pairing.pairs,
                run_dir=run_dir,
                manifest_path=manifest_paths.manifest_path,
            )
        except Exception as exc:  # noqa: BLE001
            if config.allow_placeholder_loop:
                training_result = run_placeholder_transformer_core_loop(config, manifest_paths.manifest_path)
                training_result["real_training_error"] = str(exc)
                training_result["message"] = (
                    "Real Stage 2 FLUX training could not run in this environment; fell back to the explicit placeholder loop."
                )
            else:
                training_result = {
                    "status": "failed_before_training",
                    "implemented_training": False,
                    "placeholder_training": False,
                    "message": (
                        "Real Stage 2 FLUX training was attempted but could not start or complete in this environment. "
                        "See training_error for the real runtime failure."
                    ),
                    "training_error": str(exc),
                    "component_plan_status": "real_training_attempted",
                }

    summary = {
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
        "component_plan": component_plan,
        "backbone_runtime": backbone_runtime,
        "manifest": manifest_paths.manifest_path,
        "manifest_summary": manifest_paths.summary_path,
        "unmatched_images": manifest_paths.unmatched_images_path,
        "unmatched_render_records": manifest_paths.unmatched_render_records_path,
        "num_pairs": len(pairing.pairs),
        "pairing_summary": pairing.summary,
        "training": training_result,
        "artifacts": {
            "config": str((run_dir / "stage2_config_snapshot.json").resolve()),
            "trainer_plan": str((run_dir / "trainer_plan.json").resolve()),
            "run_summary": str((run_dir / "stage2_run_summary.json").resolve()),
            "memory_log_pattern": str((run_dir / f"rank*_{config.memory_log_artifact_name}").resolve()),
        },
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    write_json(run_dir / "stage2_run_summary.json", summary)
    return summary


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



def run_real_stage2_flux_training(
    *,
    config: Stage2TrainConfig,
    pairs: list[Any],
    run_dir: Path,
    manifest_path: str,
) -> dict[str, Any]:
    if importlib.util.find_spec("torch") is None:
        raise RuntimeError("PyTorch is not installed")
    if importlib.util.find_spec("diffusers") is None:
        raise RuntimeError("diffusers is not installed")
    if config.use_accelerate and importlib.util.find_spec("accelerate") is None:
        raise RuntimeError("accelerate is not installed")

    import torch
    if config.use_accelerate:
        from accelerate import Accelerator
        from accelerate.utils import set_seed

    if not pairs:
        raise ValueError("No paired training samples were available after manifest generation")

    accelerator = None
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
    rank_info = _accelerator_rank_info(accelerator, device)
    memory_log_path = run_dir / f"rank{rank_info['global_rank']:02d}_{config.memory_log_artifact_name}"
    if accelerator is not None:
        accelerator.wait_for_everyone()
    _append_memory_event(
        artifact_path=memory_log_path,
        accelerator=accelerator,
        device=device,
        phase="training_start",
        torch_module=torch,
        extra={
            "backbone_name": config.backbone_name,
            "manifest_path": str(Path(manifest_path).resolve()),
            "num_pairs": len(pairs),
            "load_dtype": _torch_dtype_label(load_dtype),
            "train_dtype": _torch_dtype_label(train_dtype),
        },
    )

    requested_device_for_load = None if config.use_accelerate else str(device)
    requested_device_map = None if config.use_accelerate else config.backbone_device_map

    backbone = load_real_backbone_module(
        config.backbone_name,
        torch_dtype=_torch_dtype_label(load_dtype),
        device=requested_device_for_load,
        device_map=requested_device_map,
        local_files_only=config.backbone_local_files_only,
        component=None,
        allow_unimplemented=False,
    )
    pipeline = backbone.root_module
    if pipeline is None:
        raise RuntimeError("Real backbone load did not return a pipeline root module")
    _append_memory_event(
        artifact_path=memory_log_path,
        accelerator=accelerator,
        device=device,
        phase="after_backbone_load",
        torch_module=torch,
        extra={
            "resolved_module_name": backbone.resolved_module_name,
            "resolved_module_type": backbone.resolved_module_type,
            "loader": backbone.loader_name,
            "loader_status": backbone.implementation_status,
        },
    )

    selection_result = _freeze_stage2_modules(pipeline, config)
    _append_memory_event(
        artifact_path=memory_log_path,
        accelerator=accelerator,
        device=device,
        phase="after_freeze_selection",
        torch_module=torch,
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
        )
    _append_memory_event(
        artifact_path=memory_log_path,
        accelerator=accelerator,
        device=device,
        phase="after_gradient_checkpointing_setup",
        torch_module=torch,
        extra={
            "gradient_checkpointing": gradient_checkpointing,
            "keep_frozen_modules_on_cpu_until_needed": config.keep_frozen_modules_on_cpu_until_needed,
            "offload_frozen_modules_after_step": config.offload_frozen_modules_after_step,
        },
    )

    optimizer = torch.optim.AdamW(
        (parameter for parameter in transformer.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
    )

    dataloader = make_stage2_dataloader(
        pairs,
        resolution=config.resolution,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        drop_last=config.dataloader_drop_last,
    )

    if accelerator is not None:
        transformer, optimizer, dataloader = accelerator.prepare(transformer, optimizer, dataloader)
        pipeline.transformer = transformer
    _append_memory_event(
        artifact_path=memory_log_path,
        accelerator=accelerator,
        device=device,
        phase="after_dataloader_accelerate_prepare",
        torch_module=torch,
        extra={
            "batch_size": config.batch_size,
            "num_workers": config.num_workers,
            "gradient_accumulation_steps": max(config.gradient_accumulation_steps, 1),
            "dataloader_batches_per_epoch": len(dataloader) if hasattr(dataloader, "__len__") else None,
            "gradient_checkpointing": gradient_checkpointing,
            "keep_frozen_modules_on_cpu_until_needed": config.keep_frozen_modules_on_cpu_until_needed,
        },
    )

    checkpoint_dir = run_dir / "checkpoints"
    is_main_process = accelerator.is_main_process if accelerator is not None else True
    if is_main_process:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if accelerator is not None:
        accelerator.wait_for_everyone()

    logs: list[dict[str, Any]] = []
    losses: list[float] = []
    global_step = 0
    optimizer_step_count = 0
    stop_after = config.max_steps if config.max_steps is not None else None
    steps_per_epoch = len(dataloader) if hasattr(dataloader, "__len__") else None
    if steps_per_epoch in (0, None):
        total_optimizer_steps = None
    else:
        optimizer_updates_per_epoch = max(
            math.ceil(steps_per_epoch / max(config.gradient_accumulation_steps, 1)),
            1,
        )
        total_optimizer_steps = optimizer_updates_per_epoch * max(config.epochs, 1)
        if stop_after is not None:
            total_optimizer_steps = min(total_optimizer_steps, stop_after)

    for epoch in range(max(config.epochs, 1)):
        for batch in dataloader:
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
                epoch=epoch + 1,
                global_step=global_step + 1,
                optimizer_step=optimizer_step_count + 1,
                keep_frozen_modules_on_cpu_until_needed=config.keep_frozen_modules_on_cpu_until_needed,
                offload_frozen_modules_after_step=config.offload_frozen_modules_after_step,
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
                if is_main_process and optimizer_step_count % max(config.save_every, 1) == 0:
                    checkpoint_model = accelerator.unwrap_model(transformer) if accelerator is not None else transformer
                    _save_transformer_checkpoint(
                        checkpoint_model,
                        checkpoint_dir / f"step_{optimizer_step_count:06d}",
                    )
        if stop_after is not None and optimizer_step_count >= stop_after:
            break

    if accelerator is not None:
        accelerator.wait_for_everyone()
    _append_memory_event(
        artifact_path=memory_log_path,
        accelerator=accelerator,
        device=device,
        phase="training_loop_complete",
        torch_module=torch,
        epoch=max(config.epochs, 1),
        global_step=global_step,
        optimizer_step=optimizer_step_count,
        extra={"loss_count": len(losses)},
    )
    final_checkpoint_dir = checkpoint_dir / "final_transformer"
    if is_main_process:
        checkpoint_model = accelerator.unwrap_model(transformer) if accelerator is not None else transformer
        _save_transformer_checkpoint(checkpoint_model, final_checkpoint_dir)

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
        "load_dtype": _torch_dtype_label(load_dtype),
        "train_dtype": _torch_dtype_label(train_dtype),
        "training_parameterization": config.training_parameterization,
        "applied_transformer_module_selection": selection_result["selection"].to_dict(),
        "adapter_injection": selection_result["adapter_injection"].to_dict() if selection_result["adapter_injection"] is not None else None,
        "trainable_parameter_summary": selection_result["trainable_parameter_summary"],
        "accelerate": {
            "enabled": True,
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
        "memory_log_path": str(memory_log_path.resolve()),
        "launch_notes": launch_notes,
    }
    if is_main_process:
        write_json(run_dir / "training_metrics.json", summary)
    if accelerator is not None:
        accelerator.wait_for_everyone()
    return summary


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
    epoch: int,
    global_step: int,
    optimizer_step: int,
    keep_frozen_modules_on_cpu_until_needed: bool,
    offload_frozen_modules_after_step: bool,
) -> Any:
    import torch

    del resolution  # training uses the dataloader-prepared image tensor shape directly

    accumulation_context = accelerator.accumulate(transformer) if accelerator is not None else nullcontext()
    with accumulation_context:
        pixel_values = batch["pixel_values"].to(device=device, dtype=train_dtype)
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
                _move_named_pipeline_components(pipeline, component_names=["vae"], device=device, dtype=train_dtype)
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
                _move_named_pipeline_components(pipeline, component_names=["vae"], device=torch.device("cpu"), dtype=torch.float32)
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
        loss = torch.nn.functional.mse_loss(prediction.float(), target.float())
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
) -> None:
    for component_name in component_names:
        component = getattr(pipeline, component_name, None)
        if component is None or not hasattr(component, "to"):
            continue
        kwargs: dict[str, Any] = {"device": device}
        if dtype is not None and component_name != "image_encoder":
            kwargs["dtype"] = dtype
        component.to(**kwargs)



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


def _summarize_trainable_parameters(module: Any) -> dict[str, Any]:
    trainable_names: list[str] = []
    frozen_names: list[str] = []
    lora_parameter_names: list[str] = []
    trainable_parameter_count = 0
    frozen_parameter_count = 0
    lora_parameter_count = 0
    for name, parameter in module.named_parameters():
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
    return {
        "family": "generic_diffusion_backbone",
        "notes": [
            "Stage 2 wording stays generic at the method level.",
            "Module-group selectors may need replacement for a different backbone family.",
        ],
    }
