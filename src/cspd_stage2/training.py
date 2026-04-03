from __future__ import annotations

"""Training scaffold for CSPD Stage 2.

This module is deliberately honest about scope:
- it prepares run directories and paired manifests,
- records text-conditioning-focused adaptation intent,
- separates trainable and frozen component plans,
- exposes a minimal trainer contract,
- optionally runs a tiny PyTorch-backed placeholder loop,
- does not claim full FLUX.1 Kontext [dev] fine-tuning is implemented here.
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
from cspd_stage2.data import build_stage2_pairs, write_pairing_artifacts


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
    """Build Stage 2 artifacts and optionally run a placeholder trainer."""
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
        "message": (
            "Stage 2 paired manifest is ready. The code now records a concrete full-transformer fine-tuning plan "
            "for FLUX.1 Kontext, but executable real training is still not wired in this repo."
        ),
        "component_plan_status": "implemented_metadata_only",
    }

    if not config.generate_manifest_only and not config.dry_run:
        if config.allow_placeholder_loop:
            training_result = run_placeholder_transformer_core_loop(config, manifest_paths.manifest_path)
        else:
            training_result = {
                "status": "not_run",
                "implemented_training": False,
                "placeholder_training": False,
                "message": (
                    "Manifest/data prep completed. Actual generative-backbone training remains a scaffold until "
                    "a concrete FLUX Kontext or equivalent backbone-specific dependency stack is selected and integrated."
                ),
                "component_plan_status": "implemented_metadata_only",
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
            "Pattern selectors now support inspection and optional requires_grad application on a real torch module tree "
            "when one is explicitly provided. Concrete FLUX Kontext loading/training is still not implemented in this repo."
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
            "full_flux_kontext_finetuning": "not_implemented",
        },
        "notes": [
            "This scaffold is intentionally conservative.",
            "Stage 2 no longer means render; render belongs to Stage 1.",
            "Current code records a default policy of freezing non-transformer top-level modules and fine-tuning the full transformer.",
            "Current FLUX.1 Kontext references are assumption labels for planning, not proof that executable module surgery or real training is implemented.",
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
