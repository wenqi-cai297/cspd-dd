from __future__ import annotations

"""FLUX-family Stage 2 training helpers."""

import importlib.util
import math
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from cspd_stage2.backbone import load_real_backbone_module
from cspd_stage2.data import make_stage2_dataloader
from cspd_stage2.training import (
    Stage2TrainConfig,
    _accelerator_rank_info,
    _append_memory_event,
    _classify_training_failure,
    _collect_cuda_memory_stats,
    _collect_step_metrics,
    _emit_stage2_console_event,
    _enable_transformer_gradient_checkpointing,
    _finish_wandb_run,
    _freeze_stage2_modules,
    _init_wandb_run,
    _mark_sync_point,
    _move_named_pipeline_components,
    _raise_on_non_finite_scalar,
    _record_gradient_diagnostics,
    _resolve_training_device,
    _resolve_training_dtype,
    _safe_write_json,
    _sample_flux_flow_matching_timesteps,
    _save_transformer_checkpoint,
    _set_module_mode,
    _torch_dtype_label,
    _wandb_log,
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
    wandb_run = None
    wandb_state: dict[str, Any] | None = None

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
        _move_named_pipeline_components(
            pipeline,
            component_names=["vae", "text_encoder", "text_encoder_2", "image_encoder"],
            device=device,
            dtype=train_dtype,
            torch_module=torch,
            accelerator=accelerator,
            runtime_device=device,
            memory_log_path=memory_log_path,
        )
        mark_phase(
            "after_gradient_checkpointing_setup",
            extra={
                "gradient_checkpointing": gradient_checkpointing,
                "frozen_components_runtime": "always_on_device",
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
                    "frozen_components_runtime": "always_on_device",
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
        wandb_run, wandb_state = _init_wandb_run(config=config, run_dir=run_dir, is_main_process=is_main_process)
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
                step_result = _run_real_flux_train_step(
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
                )
                loss = step_result["loss"]
                global_step += 1
                sync_gradients = accelerator.sync_gradients if accelerator is not None else True
                if sync_gradients:
                    optimizer_step_count += 1
                    if accelerator is not None:
                        loss_value = float(accelerator.gather_for_metrics(loss.detach().reshape(1)).mean().item())
                    else:
                        loss_value = float(loss.detach().cpu().item())
                    losses.append(loss_value)
                    step_metrics = dict(step_result.get("step_metrics") or {})
                    if step_metrics:
                        _wandb_log(wandb_run, {f"train/{key}": value for key, value in step_metrics.items()}, step=optimizer_step_count)
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
                "frozen_components_runtime": "always_on_device",
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
            "last_known_phase": last_known_phase,
            "phase_history": phase_history,
            "launch_notes": launch_notes,
            "wandb": wandb_state,
        }
        if is_main_process:
            _safe_write_json(run_dir / "training_metrics.json", summary)
            _finish_wandb_run(wandb_run, {"status": summary["status"], "optimizer_steps": optimizer_step_count, "last_loss": losses[-1] if losses else None})
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
                "frozen_components_runtime": "always_on_device",
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
            "last_known_phase": last_known_phase,
            "phase_history": phase_history,
            "failure_category": _classify_training_failure(exc),
            "failure": {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "rank_info": _accelerator_rank_info(accelerator, device) if torch is not None else None,
            },
            "wandb": wandb_state,
        }
        if is_main_process:
            _safe_write_json(run_dir / "training_metrics.json", failure_summary)
            _finish_wandb_run(wandb_run, {"status": failure_summary["status"], "failure_category": failure_summary["failure_category"]})
        return failure_summary

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
                "frozen_components_runtime": "always_on_device",
            },
        )
        with torch.no_grad():
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
        loss_value = _raise_on_non_finite_scalar(value=loss, name="loss", accelerator=accelerator, device=device, memory_log_path=memory_log_path, epoch=epoch, global_step=global_step, optimizer_step=optimizer_step, torch_module=torch, extra={"backbone_family": "flux"})
        del prediction, target, noisy_latents, noise, packed_latents, latent_image_ids, timesteps, sigmas, guidance
        if global_step == 1:
            _emit_stage2_console_event(accelerator=accelerator, device=device, phase="after_first_forward", extra={"optimizer_step": optimizer_step, "loss": loss_value}, main_process_only=False)
        _append_memory_event(
            artifact_path=memory_log_path,
            accelerator=accelerator,
            device=device,
            phase="after_loss",
            torch_module=torch,
            epoch=epoch,
            global_step=global_step,
            optimizer_step=optimizer_step,
            extra={"loss": loss_value},
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
        grad_diagnostics = _record_gradient_diagnostics(module=accelerator.unwrap_model(transformer) if accelerator is not None else transformer, accelerator=accelerator, device=device, memory_log_path=memory_log_path, epoch=epoch, global_step=global_step, optimizer_step=optimizer_step, torch_module=torch)
        if not grad_diagnostics["all_gradients_finite"]:
            raise RuntimeError(f"Detected non-finite gradients during Stage 2 training at optimizer step {optimizer_step}: {grad_diagnostics['non_finite_gradient_parameter_names_sample']}")
        if global_step == 1:
            _emit_stage2_console_event(accelerator=accelerator, device=device, phase="after_first_backward", extra={"optimizer_step": optimizer_step, "grad_global_norm": grad_diagnostics["grad_global_norm"]}, main_process_only=False)
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
        parameter_diagnostics = _collect_trainable_parameter_diagnostics(accelerator.unwrap_model(transformer) if accelerator is not None else transformer)
        memory_stats = _collect_cuda_memory_stats(torch, device)
        if global_step == 1:
            _emit_stage2_console_event(accelerator=accelerator, device=device, phase="after_first_optimizer_step", extra={"optimizer_step": optimizer_step}, main_process_only=False)
    return {"loss": loss.detach(), "step_metrics": _collect_step_metrics(loss_value=loss_value, optimizer=optimizer, grad_diagnostics=grad_diagnostics, parameter_diagnostics=parameter_diagnostics, memory_stats=memory_stats)}
