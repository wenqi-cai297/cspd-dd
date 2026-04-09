from __future__ import annotations

"""PixArt-family Stage 2 training and sampling helpers."""

import importlib.util
import json
import math
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from cspd_stage2.backbone import infer_backbone_family, load_real_backbone_module
from cspd_stage2.data import make_stage2_dataloader
from cspd_stage2.training import (
    Stage2TrainConfig,
    _accelerator_rank_info,
    _append_memory_event,
    _assert_trainable_parameters_finite,
    _build_lr_scheduler,
    _build_optimizer,
    _collect_cuda_memory_stats,
    _collect_step_metrics,
    _classify_training_failure,
    _effective_prompt_max_sequence_length,
    _emit_stage2_console_event,
    _finish_wandb_run,
    _freeze_stage2_modules,
    _init_wandb_run,
    _mark_sync_point,
    _maybe_apply_conditioning_dropout,
    _move_named_pipeline_components,
    _prepare_pixart_forward_inputs,
    _raise_on_non_finite_scalar,
    _record_gradient_diagnostics,
    _resolve_sample_prompts,
    _resolve_training_device,
    _resolve_training_dtype,
    _resolve_pixart_partial_full_update_fp32_exclude_patterns,
    _run_pixart_wandb_sampling,
    _safe_write_json,
    _save_transformer_checkpoint,
    _set_module_mode,
    _should_force_full_update_fp32,
    _summarize_tensor_like,
    _torch_dtype_label,
    _upcast_trainable_parameters_,
    _wandb_log,
    derive_stage2_baseline_sample_output_dir,
    tqdm,
)

def run_stage2_pixart_baseline_sampling(
    *,
    dataset_root: str,
    backbone_name: str,
    output_dir: str | None = None,
    sample_prompt_file: str | None = None,
    sample_prompts: list[str] | None = None,
    sample_num_prompts: int = 4,
    sample_num_inference_steps: int = 50,
    sample_guidance_scale: float = 7.0,
    sample_seed: int = 42,
    resolution: int = 512,
    backbone_torch_dtype: str = "float16",
    backbone_device: str | None = None,
    backbone_device_map: str | None = None,
    backbone_local_files_only: bool = False,
) -> dict[str, Any]:
    """Run standalone PixArt baseline sampling outside the training loop."""
    family = infer_backbone_family(backbone_name)
    if family not in {"pixart", "pixart_sigma"}:
        raise ValueError(f"Standalone baseline sampling currently supports PixArt backbones only, got: {backbone_name}")

    config = Stage2TrainConfig(
        dataset_root=dataset_root,
        render_input="baseline_sampling",
        output_dir=output_dir or derive_stage2_baseline_sample_output_dir(dataset_root, backbone_name),
        backbone_name=backbone_name,
        sample_prompt_file=sample_prompt_file,
        sample_prompts=list(sample_prompts or []),
        sample_num_prompts=sample_num_prompts,
        sample_num_inference_steps=sample_num_inference_steps,
        sample_guidance_scale=sample_guidance_scale,
        sample_seed=sample_seed,
        resolution=resolution,
        backbone_torch_dtype=backbone_torch_dtype,
        backbone_device=backbone_device,
        backbone_device_map=backbone_device_map,
        backbone_local_files_only=backbone_local_files_only,
    )
    run_dir = Path(config.output_dir or derive_stage2_baseline_sample_output_dir(dataset_root, backbone_name))
    run_dir.mkdir(parents=True, exist_ok=True)

    prompts = _resolve_sample_prompts(config=config, pairs=[])
    if not prompts:
        raise ValueError("No baseline sampling prompts resolved. Provide --sample-prompt-file or --sample-prompt.")

    backbone = load_real_backbone_module(
        backbone_name,
        torch_dtype=backbone_torch_dtype,
        device=backbone_device,
        device_map=backbone_device_map,
        local_files_only=backbone_local_files_only,
        component=None,
        allow_unimplemented=False,
    )
    pipeline = backbone.root_module
    if pipeline is None:
        raise RuntimeError("Real PixArt backbone load did not return a pipeline root module")
    transformer = getattr(pipeline, "transformer", None)
    if transformer is None:
        raise RuntimeError("Loaded PixArt pipeline does not expose a transformer component")

    import torch

    if backbone_device is None and torch.cuda.is_available():
        backbone_device = "cuda"

    if backbone_device is not None:
        pipeline.to(backbone_device)
        runtime_device = torch.device(backbone_device)
    else:
        runtime_device = next(transformer.parameters()).device
        if runtime_device.type == "cpu" and getattr(pipeline, "_execution_device", None) is not None:
            execution_device = getattr(pipeline, "_execution_device")
            runtime_device = execution_device if isinstance(execution_device, torch.device) else torch.device(str(execution_device))

    sample_event = _run_pixart_wandb_sampling(
        pipeline=pipeline,
        transformer=transformer,
        accelerator=None,
        config=config,
        run_dir=run_dir,
        epoch=0,
        optimizer_step=0,
        prompts=prompts,
        device=runtime_device,
        train_dtype=getattr(torch, backbone_torch_dtype, torch.float16),
        wandb_run=None,
    )

    summary = {
        "status": sample_event.get("status", "unknown"),
        "mode": "standalone_pixart_baseline_sampling",
        "dataset_root": str(Path(dataset_root).expanduser().resolve()),
        "output_dir": str(run_dir.resolve()),
        "backbone_name": backbone_name,
        "backbone_family": family,
        "loader": backbone.loader_name,
        "loader_status": backbone.implementation_status,
        "resolved_module_name": backbone.resolved_module_name,
        "resolved_module_type": backbone.resolved_module_type,
        "sample_prompt_file": str(Path(sample_prompt_file).expanduser().resolve()) if sample_prompt_file else None,
        "sample_prompts": prompts,
        "sample_num_prompts": len(prompts),
        "sample_num_inference_steps": sample_num_inference_steps,
        "sample_guidance_scale": sample_guidance_scale,
        "sample_seed": sample_seed,
        "resolution": resolution,
        "backbone_torch_dtype": backbone_torch_dtype,
        "backbone_device": backbone_device,
        "backbone_device_map": backbone_device_map,
        "backbone_local_files_only": backbone_local_files_only,
        "sample_event": sample_event,
    }
    (run_dir / "baseline_sampling_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary

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
    fp32_full_update_summary: dict[str, Any] = {"enabled": False, "reason": "not_applicable"}
    logs: list[dict[str, Any]] = []
    losses: list[float] = []
    global_step = 0
    optimizer_step_count = 0
    steps_per_epoch = None
    total_optimizer_steps = None
    last_known_phase = "preflight"
    phase_history: list[str] = []
    wandb_run = None
    wandb_state: dict[str, Any] | None = None
    sample_prompts: list[str] = []
    sample_events: list[dict[str, Any]] = []

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
        if _should_force_full_update_fp32(config=config):
            fp32_upcast_exclude_patterns = _resolve_pixart_partial_full_update_fp32_exclude_patterns(config=config)
            fp32_full_update_summary = _upcast_trainable_parameters_(transformer, dtype=torch.float32, exclude_patterns=fp32_upcast_exclude_patterns)
            if fp32_full_update_summary["enabled"] or fp32_full_update_summary.get("skipped_parameter_count", 0) > 0:
                mark_phase("after_full_update_fp32_upcast", extra=fp32_full_update_summary)
            else:
                fp32_full_update_summary = {**fp32_full_update_summary, "reason": "no_trainable_parameter_dtype_changes_needed"}
        transformer.train()
        if config.enable_gradient_checkpointing:
            gradient_checkpointing = _enable_transformer_gradient_checkpointing(transformer)
        else:
            gradient_checkpointing = {"enabled": False, "method": None, "attempted_methods": [], "reason": "disabled_by_config"}
        _set_module_mode(getattr(pipeline, "vae", None), training=False)
        _set_module_mode(getattr(pipeline, "text_encoder", None), training=False)
        _set_module_mode(getattr(pipeline, "text_encoder_2", None), training=False)
        _move_named_pipeline_components(
            pipeline,
            component_names=["vae", "text_encoder", "text_encoder_2"],
            device=device,
            dtype=train_dtype,
            torch_module=torch,
            accelerator=accelerator,
            runtime_device=device,
            memory_log_path=memory_log_path,
        )
        mark_phase("after_frozen_component_device_setup", extra={"component_names": ["vae", "text_encoder", "text_encoder_2"], "frozen_components_runtime": "always_on_device"})

        mark_phase("before_optimizer_setup")
        trainable_parameters = [parameter for parameter in transformer.parameters() if parameter.requires_grad]
        optimizer = _build_optimizer(parameters=trainable_parameters, config=config, torch_module=torch)
        mark_phase("after_optimizer_setup", extra={"optimizer_name": config.optimizer_name, "learning_rate": config.learning_rate, "adam_beta1": config.adam_beta1, "adam_beta2": config.adam_beta2, "adam_weight_decay": config.adam_weight_decay, "adam_epsilon": config.adam_epsilon})
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
        wandb_run, wandb_state = _init_wandb_run(config=config, run_dir=run_dir, is_main_process=is_main_process)
        if is_main_process:
            sample_prompts = _resolve_sample_prompts(config=config, pairs=pairs)
            if wandb_run is not None and sample_prompts:
                _wandb_log(wandb_run, {"samples/prompts": "\\n".join(sample_prompts)}, step=0)
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
        lr_scheduler, lr_scheduler_summary = _build_lr_scheduler(optimizer=optimizer, config=config, total_optimizer_steps=total_optimizer_steps)
        mark_phase("after_lr_scheduler_setup", extra=lr_scheduler_summary)

        if is_main_process and config.sample_every > 0 and sample_prompts:
            initial_sample_event = _run_pixart_wandb_sampling(
                pipeline=pipeline,
                transformer=transformer,
                accelerator=accelerator,
                config=config,
                run_dir=run_dir,
                epoch=0,
                optimizer_step=0,
                prompts=sample_prompts,
                device=device,
                train_dtype=train_dtype,
                wandb_run=wandb_run,
            )
            sample_events.append(initial_sample_event)
            _wandb_log(wandb_run, {"samples/last_status": initial_sample_event.get("status"), "samples/last_prompt_count": initial_sample_event.get("prompt_count")}, step=0)
            mark_phase("training_loop_complete", epoch=0, global_step_value=0, optimizer_step_value=0, extra={"loss_total": None, "loss_last": None, "optimizer_steps": 0}, main_process_only=True)

        epoch_summaries: list[dict[str, Any]] = []
        for epoch in range(max(config.epochs, 1)):
            epoch_losses: list[float] = []
            epoch_optimizer_steps_start = optimizer_step_count
            mark_phase("epoch_start", epoch=epoch + 1, extra={"epoch_index": epoch + 1, "planned_epochs": max(config.epochs, 1)})
            progress_bar = None
            dataloader_iterable = dataloader
            if is_main_process and tqdm is not None and hasattr(dataloader, "__len__"):
                progress_bar = tqdm(dataloader, desc=f"Stage2 Epoch {epoch + 1}/{max(config.epochs, 1)}", leave=True, dynamic_ncols=True)
                dataloader_iterable = progress_bar
            try:
                for batch_index, batch in enumerate(dataloader_iterable, start=1):
                    if epoch == 0 and batch_index == 1:
                        mark_phase("first_batch_fetched", epoch=epoch + 1, global_step_value=global_step + 1, optimizer_step_value=optimizer_step_count + 1, extra={"batch_index": batch_index, "batch_size": len(batch.get("conditioning_text", [])) if isinstance(batch, dict) else None}, main_process_only=False)
                    if stop_after is not None and optimizer_step_count >= stop_after:
                        break
                    step_result = _run_real_pixart_train_step(
                        pipeline=pipeline,
                        transformer=transformer,
                        batch=batch,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        accelerator=accelerator,
                        device=device,
                        train_dtype=train_dtype,
                        memory_log_path=memory_log_path,
                        epoch=epoch + 1,
                        global_step=global_step + 1,
                        optimizer_step=optimizer_step_count + 1,
                        config=config,
                    )
                    loss = step_result["loss"]
                    global_step += 1
                    sync_gradients = accelerator.sync_gradients if accelerator is not None else True
                    if sync_gradients:
                        optimizer_step_count += 1
                        loss_value = float(accelerator.gather_for_metrics(loss.detach().reshape(1)).mean().item()) if accelerator is not None else float(loss.detach().cpu().item())
                        losses.append(loss_value)
                        epoch_losses.append(loss_value)
                        if progress_bar is not None:
                            progress_bar.set_postfix(loss=f"{loss_value:.4f}", opt_step=optimizer_step_count)
                        step_metrics = dict(step_result.get("step_metrics") or {})
                        if step_metrics:
                            _wandb_log(wandb_run, {f"train/{key}": value for key, value in step_metrics.items()}, step=optimizer_step_count)
                        if is_main_process and config.sample_every > 0 and optimizer_step_count % max(config.sample_every, 1) == 0:
                            sample_event = _run_pixart_wandb_sampling(
                                pipeline=pipeline,
                                transformer=transformer,
                                accelerator=accelerator,
                                config=config,
                                run_dir=run_dir,
                                epoch=epoch + 1,
                                optimizer_step=optimizer_step_count,
                                prompts=sample_prompts,
                                device=device,
                                train_dtype=train_dtype,
                                wandb_run=wandb_run,
                            )
                            sample_events.append(sample_event)
                            _wandb_log(wandb_run, {"samples/last_status": sample_event.get("status"), "samples/last_prompt_count": sample_event.get("prompt_count")}, step=optimizer_step_count)
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
            finally:
                if progress_bar is not None:
                    progress_bar.close()
            epoch_optimizer_steps = optimizer_step_count - epoch_optimizer_steps_start
            epoch_summary = {
                "epoch": epoch + 1,
                "optimizer_steps": epoch_optimizer_steps,
                "loss_total": float(sum(epoch_losses)) if epoch_losses else None,
                "loss_mean": float(sum(epoch_losses) / len(epoch_losses)) if epoch_losses else None,
                "loss_min": float(min(epoch_losses)) if epoch_losses else None,
                "loss_max": float(max(epoch_losses)) if epoch_losses else None,
                "loss_last": float(epoch_losses[-1]) if epoch_losses else None,
            }
            epoch_summaries.append(epoch_summary)
            if is_main_process and epoch_optimizer_steps > 0:
                epoch_wandb_payload = {
                    "epoch/index": epoch + 1,
                    "epoch/optimizer_steps": epoch_optimizer_steps,
                }
                if epoch_summary["loss_total"] is not None:
                    epoch_wandb_payload["epoch/loss_total"] = epoch_summary["loss_total"]
                    epoch_wandb_payload["epoch/loss_mean"] = epoch_summary["loss_mean"]
                    epoch_wandb_payload["epoch/loss_min"] = epoch_summary["loss_min"]
                    epoch_wandb_payload["epoch/loss_max"] = epoch_summary["loss_max"]
                    epoch_wandb_payload["epoch/loss_last"] = epoch_summary["loss_last"]
                _wandb_log(wandb_run, epoch_wandb_payload, step=max(optimizer_step_count, 1))
                mark_phase("training_loop_complete", epoch=epoch + 1, global_step_value=global_step, optimizer_step_value=optimizer_step_count, extra={"loss_total": epoch_summary["loss_total"], "loss_last": epoch_summary["loss_last"], "optimizer_steps": epoch_optimizer_steps}, main_process_only=True)

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
            "optimizer": {"name": config.optimizer_name, "learning_rate": config.learning_rate, "adam_beta1": config.adam_beta1, "adam_beta2": config.adam_beta2, "adam_weight_decay": config.adam_weight_decay, "adam_epsilon": config.adam_epsilon},
            "lr_scheduler": lr_scheduler_summary,
            "gradient_checkpointing": gradient_checkpointing,
            "full_update_fp32": fp32_full_update_summary,
            "memory_strategy": {"frozen_components_runtime": "always_on_device"},
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
            "wandb": wandb_state,
            "sample_prompts": sample_prompts,
            "sample_events": sample_events,
            "epoch_summaries": epoch_summaries,
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
            "optimizer": {"name": config.optimizer_name, "learning_rate": config.learning_rate, "adam_beta1": config.adam_beta1, "adam_beta2": config.adam_beta2, "adam_weight_decay": config.adam_weight_decay, "adam_epsilon": config.adam_epsilon},
            "lr_scheduler": lr_scheduler_summary if 'lr_scheduler_summary' in locals() else None,
            "gradient_checkpointing": gradient_checkpointing,
            "full_update_fp32": fp32_full_update_summary,
            "memory_strategy": {"frozen_components_runtime": "always_on_device"},
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
            "failure": {"error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(), "rank_info": _accelerator_rank_info(accelerator, device) if torch is not None else None},
            "wandb": wandb_state,
            "sample_prompts": sample_prompts,
            "sample_events": sample_events,
        }
        if is_main_process:
            _safe_write_json(run_dir / "training_metrics.json", failure_summary)
            _finish_wandb_run(wandb_run, {"status": failure_summary["status"], "failure_category": failure_summary["failure_category"]})
        return failure_summary

def _run_real_pixart_train_step(
    *,
    pipeline: Any,
    transformer: Any,
    batch: dict[str, Any],
    optimizer: Any,
    lr_scheduler: Any | None,
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
            _append_memory_event(
                artifact_path=memory_log_path,
                accelerator=accelerator,
                device=device,
                phase="vae_encode_runtime",
                torch_module=torch,
                epoch=epoch,
                global_step=global_step,
                optimizer_step=optimizer_step,
                extra={"vae_device": str(vae_device), "vae_dtype": _torch_dtype_label(vae_dtype)},
            )
            latents = pipeline.vae.encode(pixel_values.to(device=vae_device, dtype=vae_dtype)).latent_dist.sample()
            if global_step == 1:
                _emit_stage2_console_event(accelerator=accelerator, device=device, phase="after_first_vae_encode", extra={"optimizer_step": optimizer_step, "latents_shape": list(latents.shape), "vae_device": str(vae_device)}, main_process_only=False)
            scaling_factor = float(getattr(getattr(pipeline.vae, "config", None), "scaling_factor", 0.13025))
            shift_factor = float(getattr(getattr(pipeline.vae, "config", None), "shift_factor", 0.0) or 0.0)
            latents = (latents - shift_factor) * scaling_factor
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
                extra={"latents_shape": list(latents.shape), "vae_device": str(vae_device)},
            )
            if global_step == 1:
                _emit_stage2_console_event(accelerator=accelerator, device=device, phase="before_first_text_encode", extra={"optimizer_step": optimizer_step}, main_process_only=False)
            text_encoder = getattr(pipeline, "text_encoder", None)
            text_device = next(text_encoder.parameters()).device if text_encoder is not None and hasattr(text_encoder, "parameters") else device
            _append_memory_event(
                artifact_path=memory_log_path,
                accelerator=accelerator,
                device=device,
                phase="before_prompt_encode",
                torch_module=torch,
                epoch=epoch,
                global_step=global_step,
                optimizer_step=optimizer_step,
                extra={
                    "prompt_sample": batch["conditioning_text"][0] if batch.get("conditioning_text") else None,
                    "text_encoder_device": str(text_device),
                },
            )
            conditioning_text, dropped_prompt_count = _maybe_apply_conditioning_dropout(batch["conditioning_text"], config=config, torch_module=torch)
            prompt_embeds, prompt_attention_mask, _, _ = pipeline.encode_prompt(
                prompt=conditioning_text,
                do_classifier_free_guidance=False,
                device=device,
                num_images_per_prompt=1,
                max_sequence_length=_effective_prompt_max_sequence_length(config),
            )
            prompt_embeds = prompt_embeds.to(device=device, dtype=train_dtype)
            prompt_attention_mask = prompt_attention_mask.to(device=device)
            if global_step == 1:
                _emit_stage2_console_event(accelerator=accelerator, device=device, phase="after_first_text_encode", extra={"optimizer_step": optimizer_step, "prompt_embeds_shape": list(prompt_embeds.shape), "text_encoder_device": str(text_device)}, main_process_only=False)
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
                    "prompt_attention_mask_shape": list(prompt_attention_mask.shape),
                    "text_encoder_device": str(text_device),
                    "dropped_prompt_count": dropped_prompt_count,
                },
            )

        noise = torch.randn_like(latents)
        batch_size = latents.shape[0]
        if not hasattr(pipeline.scheduler, "config") or not hasattr(pipeline.scheduler, "add_noise"):
            raise RuntimeError("Loaded PixArt scheduler does not expose add_noise for training")
        num_train_timesteps = int(getattr(pipeline.scheduler.config, "num_train_timesteps", 1000))
        timesteps = torch.randint(0, num_train_timesteps, (batch_size,), device=device, dtype=torch.long)
        noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps)
        prepared_forward_inputs = _prepare_pixart_forward_inputs(
            transformer=transformer,
            noisy_latents=noisy_latents,
            prompt_embeds=prompt_embeds,
            device=device,
            train_dtype=train_dtype,
        )
        noisy_latents = prepared_forward_inputs["hidden_states"]
        prompt_embeds = prepared_forward_inputs["encoder_hidden_states"]
        added_cond_kwargs = prepared_forward_inputs["added_cond_kwargs"]
        forward_input_summary = {
            "hidden_states": _summarize_tensor_like(noisy_latents),
            "encoder_hidden_states": _summarize_tensor_like(prompt_embeds),
            "encoder_attention_mask": _summarize_tensor_like(prompt_attention_mask),
            "timestep": _summarize_tensor_like(timesteps),
            "added_cond_kwargs": {key: _summarize_tensor_like(value) for key, value in added_cond_kwargs.items()},
            "latent_hw": prepared_forward_inputs["latent_hw"],
            "pixart_input_dtype_plan": prepared_forward_inputs["dtype_plan"],
        }
        _append_memory_event(
            artifact_path=memory_log_path,
            accelerator=accelerator,
            device=device,
            phase="before_pixart_forward_input_prep",
            torch_module=torch,
            epoch=epoch,
            global_step=global_step,
            optimizer_step=optimizer_step,
            extra=forward_input_summary,
        )
        if global_step == 1:
            _emit_stage2_console_event(accelerator=accelerator, device=device, phase="before_first_forward", extra={"optimizer_step": optimizer_step, "forward_inputs": forward_input_summary}, main_process_only=False)
        model_output = transformer(hidden_states=noisy_latents, encoder_hidden_states=prompt_embeds, encoder_attention_mask=prompt_attention_mask, timestep=timesteps, added_cond_kwargs=added_cond_kwargs, return_dict=True)
        prediction = model_output.sample if hasattr(model_output, "sample") else model_output[0]
        out_channels = int(getattr(getattr(transformer, "config", None), "out_channels", prediction.shape[1]))
        latent_channels = int(getattr(getattr(transformer, "config", None), "in_channels", latents.shape[1]))
        if out_channels // 2 == latent_channels:
            prediction = prediction.chunk(2, dim=1)[0]
        loss = torch.nn.functional.mse_loss(prediction.float(), noise.float())
        loss_value = _raise_on_non_finite_scalar(value=loss, name="loss", accelerator=accelerator, device=device, memory_log_path=memory_log_path, epoch=epoch, global_step=global_step, optimizer_step=optimizer_step, torch_module=torch, extra={"backbone_family": "pixart"})
        _append_memory_event(
            artifact_path=memory_log_path,
            accelerator=accelerator,
            device=device,
            phase="after_pixart_forward",
            torch_module=torch,
            epoch=epoch,
            global_step=global_step,
            optimizer_step=optimizer_step,
            extra={
                "prediction": _summarize_tensor_like(prediction),
                "loss": loss_value,
            },
        )
        if global_step == 1:
            _emit_stage2_console_event(accelerator=accelerator, device=device, phase="after_first_forward", extra={"optimizer_step": optimizer_step, "loss": loss_value, "prediction": _summarize_tensor_like(prediction)}, main_process_only=False)
        optimizer.zero_grad(set_to_none=True)
        _append_memory_event(
            artifact_path=memory_log_path,
            accelerator=accelerator,
            device=device,
            phase="before_pixart_backward",
            torch_module=torch,
            epoch=epoch,
            global_step=global_step,
            optimizer_step=optimizer_step,
            extra={"loss": loss_value},
        )
        if accelerator is not None:
            accelerator.backward(loss)
        else:
            loss.backward()
        if accelerator is None or accelerator.sync_gradients:
            grad_norm_value = float(torch.nn.utils.clip_grad_norm_(transformer.parameters(), max(config.max_grad_norm, 0.0)).detach().cpu().item()) if config.max_grad_norm > 0 else None
        else:
            grad_norm_value = None
        grad_diagnostics = _record_gradient_diagnostics(module=accelerator.unwrap_model(transformer) if accelerator is not None else transformer, accelerator=accelerator, device=device, memory_log_path=memory_log_path, epoch=epoch, global_step=global_step, optimizer_step=optimizer_step, torch_module=torch)
        if not grad_diagnostics["all_gradients_finite"]:
            raise RuntimeError(f"Detected non-finite gradients during Stage 2 training at optimizer step {optimizer_step}: {grad_diagnostics['non_finite_gradient_parameter_names_sample']}")
        _append_memory_event(
            artifact_path=memory_log_path,
            accelerator=accelerator,
            device=device,
            phase="after_pixart_backward",
            torch_module=torch,
            epoch=epoch,
            global_step=global_step,
            optimizer_step=optimizer_step,
            extra={"grad_global_norm": grad_diagnostics["grad_global_norm"], "clipped_grad_norm": grad_norm_value, "all_gradients_finite": grad_diagnostics["all_gradients_finite"]},
        )
        if global_step == 1:
            _emit_stage2_console_event(accelerator=accelerator, device=device, phase="after_first_backward", extra={"optimizer_step": optimizer_step, "grad_global_norm": grad_diagnostics["grad_global_norm"]}, main_process_only=False)
        parameter_diagnostics = None
        if accelerator is None or accelerator.sync_gradients:
            optimizer.step()
            parameter_diagnostics = _assert_trainable_parameters_finite(
                module=accelerator.unwrap_model(transformer) if accelerator is not None else transformer,
                accelerator=accelerator,
                device=device,
                memory_log_path=memory_log_path,
                epoch=epoch,
                global_step=global_step,
                optimizer_step=optimizer_step,
                torch_module=torch,
            )
            if lr_scheduler is not None:
                lr_scheduler.step()
            if global_step == 1:
                _emit_stage2_console_event(accelerator=accelerator, device=device, phase="after_first_optimizer_step", extra={"optimizer_step": optimizer_step, "grad_global_norm": grad_diagnostics["grad_global_norm"], "clipped_grad_norm": grad_norm_value, "lr": float(optimizer.param_groups[0]["lr"]), "max_abs_trainable_parameter_value": None if parameter_diagnostics is None else parameter_diagnostics["max_abs_trainable_parameter_value"]}, main_process_only=False)
        transformer_train_dtype_label = _torch_dtype_label(train_dtype)
        if parameter_diagnostics is not None:
            dtype_counts = parameter_diagnostics.get("trainable_parameter_dtype_counts") or {}
            if len(dtype_counts) == 1:
                transformer_train_dtype_label = next(iter(dtype_counts))
            elif dtype_counts:
                transformer_train_dtype_label = f"mixed:{dtype_counts}"
        memory_stats = _collect_cuda_memory_stats(torch, device)
        _append_memory_event(artifact_path=memory_log_path, accelerator=accelerator, device=device, phase="after_pixart_step", torch_module=torch, epoch=epoch, global_step=global_step, optimizer_step=optimizer_step, extra={"pixel_values_shape": list(batch['pixel_values'].shape), "latents_shape": list(latents.shape), "timesteps_shape": list(timesteps.shape), "prompt_embeds_shape": list(prompt_embeds.shape), "prompt_attention_mask_shape": list(prompt_attention_mask.shape), "dropped_prompt_count": dropped_prompt_count, "lr": float(optimizer.param_groups[0]["lr"]), "clipped_grad_norm": grad_norm_value, "transformer_train_dtype": transformer_train_dtype_label, "trainable_parameter_diagnostics": parameter_diagnostics})
        return {"loss": loss.detach(), "step_metrics": _collect_step_metrics(loss_value=loss_value, optimizer=optimizer, grad_diagnostics=grad_diagnostics, parameter_diagnostics=parameter_diagnostics, memory_stats=memory_stats, extra={"clipped_grad_norm": grad_norm_value, "dropped_prompt_count": dropped_prompt_count})}
