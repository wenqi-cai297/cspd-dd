from __future__ import annotations

"""Training utilities for CSPD Stage 2.

This module is deliberately honest about scope:
- it prepares run directories and paired manifests,
- records text-conditioning-focused adaptation intent,
- separates trainable and frozen component plans,
- implements a minimal real FLUX training path over (image, canonical_caption) pairs when the runtime supports it,
- keeps the older tiny placeholder loop as an explicit plumbing fallback,
- does not pretend every environment can actually load or fine-tune gated FLUX checkpoints.
"""

import importlib.util
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

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

DEFAULT_FLUX_KONTEXT_INCLUDE_PATTERNS = [
    "*",
]

DEFAULT_EXCLUDE_PATTERNS = [
    "vae",
    "autoencoder",
    "decoder",
    "image_encoder",
]


@dataclass(slots=True)
class AdapterPlan:
    adapter_type: str = "lora"
    rank: int = 16
    alpha: float = 16.0
    dropout: float = 0.0
    target_strategy: str = "module_name_patterns"
    target_module_patterns: list[str] = field(default_factory=lambda: list(DEFAULT_FLUX_KONTEXT_INCLUDE_PATTERNS))
    exclude_module_patterns: list[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDE_PATTERNS))
    bias: str = "none"
    task_note: str = "placeholder adapter config until a concrete backbone-specific training stack is integrated"


@dataclass(slots=True)
class Stage2TrainConfig:
    dataset_root: str
    render_input: str
    output_dir: str
    backbone_name: str = "black-forest-labs/FLUX.1-Kontext-dev"
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
    module_include_patterns: list[str] = field(default_factory=lambda: list(DEFAULT_FLUX_KONTEXT_INCLUDE_PATTERNS))
    module_exclude_patterns: list[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDE_PATTERNS))
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
        },
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    write_json(run_dir / "stage2_run_summary.json", summary)
    return summary


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

    import torch

    if not pairs:
        raise ValueError("No paired training samples were available after manifest generation")

    device = _resolve_training_device(config)
    load_dtype = _resolve_training_dtype(config, device)
    train_dtype = torch.float32 if device.type == "cpu" else load_dtype

    backbone = load_real_backbone_module(
        config.backbone_name,
        torch_dtype=_torch_dtype_label(load_dtype),
        device=str(device),
        device_map=config.backbone_device_map,
        local_files_only=config.backbone_local_files_only,
        component=None,
        allow_unimplemented=False,
    )
    pipeline = backbone.root_module
    if pipeline is None:
        raise RuntimeError("Real backbone load did not return a pipeline root module")

    _freeze_stage2_modules(pipeline, config)
    transformer = pipeline.transformer
    transformer.train()
    if device.type == "cpu":
        transformer.to(device=device, dtype=train_dtype)
    else:
        transformer.to(device=device)

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
    )

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logs: list[dict[str, Any]] = []
    losses: list[float] = []
    global_step = 0
    stop_after = config.max_steps if config.max_steps is not None else None

    torch.manual_seed(config.seed)

    for epoch in range(max(config.epochs, 1)):
        for batch in dataloader:
            if stop_after is not None and global_step >= stop_after:
                break
            loss = _run_real_flux_train_step(
                pipeline=pipeline,
                transformer=transformer,
                batch=batch,
                optimizer=optimizer,
                device=device,
                train_dtype=train_dtype,
                resolution=config.resolution,
            )
            global_step += 1
            losses.append(loss)
            if global_step == 1 or global_step % max(config.log_every, 1) == 0:
                logs.append({"step": global_step, "epoch": epoch + 1, "loss": loss})
            if global_step % max(config.save_every, 1) == 0:
                _save_transformer_checkpoint(transformer, checkpoint_dir / f"step_{global_step:06d}")
        if stop_after is not None and global_step >= stop_after:
            break

    final_checkpoint_dir = checkpoint_dir / "final_transformer"
    _save_transformer_checkpoint(transformer, final_checkpoint_dir)

    summary = {
        "status": "completed",
        "implemented_training": True,
        "placeholder_training": False,
        "message": "Completed a minimal real FLUX Stage 2 training run on (image, canonical_caption) pairs.",
        "component_plan_status": "real_training_ran",
        "manifest_path": str(Path(manifest_path).resolve()),
        "device": str(device),
        "load_dtype": _torch_dtype_label(load_dtype),
        "train_dtype": _torch_dtype_label(train_dtype),
        "steps": global_step,
        "epochs": max(config.epochs, 1),
        "num_pairs": len(pairs),
        "losses": losses,
        "logs": logs,
        "final_checkpoint_dir": str(final_checkpoint_dir.resolve()),
    }
    write_json(run_dir / "training_metrics.json", summary)
    return summary


def _run_real_flux_train_step(
    *,
    pipeline: Any,
    transformer: Any,
    batch: dict[str, Any],
    optimizer: Any,
    device: Any,
    train_dtype: Any,
    resolution: int,
) -> float:
    import torch

    del resolution  # training uses the dataloader-prepared image tensor shape directly

    pixel_values = batch["pixel_values"].to(device=device, dtype=train_dtype)
    with torch.no_grad():
        vae_dtype = next(pipeline.vae.parameters()).dtype
        vae_device = next(pipeline.vae.parameters()).device
        latents = pipeline.vae.encode(pixel_values.to(device=vae_device, dtype=vae_dtype)).latent_dist.sample()
        latents = (latents - pipeline.vae.config.shift_factor) * pipeline.vae.config.scaling_factor
        latents = latents.to(device=device, dtype=train_dtype)
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
    if getattr(getattr(transformer, "config", None), "guidance_embeds", False):
        guidance = torch.ones((packed_latents.shape[0],), device=device, dtype=torch.float32)

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

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu().item())



def _sample_flux_flow_matching_timesteps(*, batch_size: int, device: Any, dtype: Any) -> tuple[Any, Any]:
    import torch

    timesteps = torch.rand((batch_size,), device=device, dtype=torch.float32)
    sigmas = timesteps.to(device=device, dtype=dtype)
    while sigmas.ndim < 3:
        sigmas = sigmas.unsqueeze(-1)
    return timesteps, sigmas



def _freeze_stage2_modules(pipeline: Any, config: Stage2TrainConfig) -> None:
    for component_name in ["transformer", "text_encoder", "text_encoder_2", "vae", "image_encoder"]:
        component = getattr(pipeline, component_name, None)
        if component is None or not hasattr(component, "parameters"):
            continue
        for parameter in component.parameters():
            parameter.requires_grad = False

    if not hasattr(pipeline, "transformer") or pipeline.transformer is None:
        raise RuntimeError("Loaded pipeline does not expose a transformer component")

    transformer = pipeline.transformer
    if config.train_transformer_core_only:
        for parameter in transformer.parameters():
            parameter.requires_grad = True
    if config.apply_real_module_selection:
        apply_trainable_parameter_selection(
            transformer,
            include_patterns=config.module_include_patterns,
            exclude_patterns=config.module_exclude_patterns,
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
    return payload


def _build_component_plan(config: Stage2TrainConfig, backbone_runtime: dict[str, Any]) -> dict[str, Any]:
    trainable_groups = list(dict.fromkeys(config.trainable_component_groups))
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
        "frozen_component_groups": frozen_groups,
        "module_selection": {
            "include_patterns": config.module_include_patterns,
            "exclude_patterns": config.module_exclude_patterns,
            "selection_semantics": (
                "pattern_inspection_with_optional_requires_grad_and_adapter_injection"
                if config.inspect_module_reference
                else "pattern_metadata_only"
            ),
        },
        "adapter_plan": asdict(config.adapter_plan),
        "backbone_assumptions": _infer_backbone_assumptions(config.backbone_name),
        "backbone_runtime": backbone_runtime,
        "implementation_boundary": (
            "Pattern selectors support inspection and optional requires_grad application on a real torch module tree, "
            "and the repo now includes a minimal real diffusers-backed FLUX-family transformer fine-tuning path over "
            "(image, canonical_caption) pairs when the runtime can load the requested backbone."
        ),
    }


def _build_backbone_runtime_summary(config: Stage2TrainConfig) -> dict[str, Any]:
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
        if config.apply_real_module_selection:
            targeting = apply_trainable_parameter_selection(
                module,
                include_patterns=config.module_include_patterns,
                exclude_patterns=config.module_exclude_patterns,
            )
            summary["module_selection_applied"] = True
        else:
            targeting = inspect_target_modules(
                module,
                include_patterns=config.module_include_patterns,
                exclude_patterns=config.module_exclude_patterns,
                limit=config.inspect_limit,
            )
        summary["inspection_status"] = "ok"
        summary["module_targeting"] = targeting.to_dict()

        if config.inject_adapters_on_real_module:
            injection = inject_lora_adapters(
                module,
                include_patterns=config.adapter_plan.target_module_patterns,
                exclude_patterns=config.adapter_plan.exclude_module_patterns,
                rank=config.adapter_plan.rank,
                alpha=config.adapter_plan.alpha,
                dropout=config.adapter_plan.dropout,
            )
            summary["adapter_injection_applied"] = True
            summary["adapter_injection"] = injection.to_dict()
            summary["module_targeting_after_adapter_injection"] = inspect_target_modules(
                module,
                include_patterns=config.module_include_patterns,
                exclude_patterns=config.module_exclude_patterns,
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
            "real_module_target_selection": "optional_when_explicit_module_reference_is_provided",
            "real_module_adapter_injection": "optional_when_explicit_module_reference_is_provided",
            "placeholder_loop": "optional",
            "full_flux_kontext_finetuning": "minimally_implemented_when_runtime_supports_real_backbone_loading",
        },
        "notes": [
            "This scaffold is intentionally conservative.",
            "Stage 2 no longer means render; render belongs to Stage 1.",
            "Current code records a default policy of freezing non-transformer top-level modules and fine-tuning the full transformer.",
            "Real diffusers-backed FLUX-family training is wired conservatively around packed VAE latents and canonical-caption prompt encoding, but successful execution still depends on the local runtime actually loading the requested backbone.",
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
