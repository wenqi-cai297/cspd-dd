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
    inspect_target_modules,
    load_generative_backbone,
    load_module_from_reference,
)
from cspd_stage2.data import build_stage2_pairs, write_pairing_artifacts


DEFAULT_TEXT_CONDITIONING_GROUPS = [
    "conditioning_bridge",
    "cross_attention",
    "transformer_text_conditioning",
]

DEFAULT_FLUX_KONTEXT_INCLUDE_PATTERNS = [
    "transformer.*attn",
    "transformer.*cross_attn",
    "transformer.*context",
    "transformer.*txt",
    "context_embedder",
    "conditioning_bridge",
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
    stage2_focus: str = "text_conditioning_adaptation"
    conditioning_objective: str = "align_stage1_canonical_captions_with_backbone_text_conditioning_path"
    conditioning_text_field: str = "canonical_caption"
    trainable_component_groups: list[str] = field(default_factory=lambda: list(DEFAULT_TEXT_CONDITIONING_GROUPS))
    module_include_patterns: list[str] = field(default_factory=lambda: list(DEFAULT_FLUX_KONTEXT_INCLUDE_PATTERNS))
    module_exclude_patterns: list[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDE_PATTERNS))
    adapter_plan: AdapterPlan = field(default_factory=AdapterPlan)
    inspect_module_reference: str | None = None
    inspect_limit: int = 200
    apply_real_module_selection: bool = False


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
            "Stage 2 paired manifest is ready. The code now records a concrete text-conditioning adaptation plan, "
            "but full FLUX.1 Kontext fine-tuning is still not wired in this repo."
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
        "definition": "generative-backbone adaptation / canonical-semantic-space familiarization",
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
        frozen_groups.append("non_transformer_blocks")
    if config.freeze_text_encoder:
        frozen_groups.append("text_encoder")
    if config.freeze_vae:
        frozen_groups.append("vae")

    return {
        "focus": config.stage2_focus,
        "conditioning_objective": config.conditioning_objective,
        "trainable_component_groups": trainable_groups,
        "frozen_component_groups": frozen_groups,
        "module_selection": {
            "include_patterns": config.module_include_patterns,
            "exclude_patterns": config.module_exclude_patterns,
            "selection_semantics": "pattern_metadata_only",
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
    load_result = load_generative_backbone(config.backbone_name, allow_unimplemented=True)
    summary: dict[str, Any] = {
        "backbone_name": config.backbone_name,
        "family": infer_backbone_family(config.backbone_name),
        "loader": load_result.loader_name,
        "loader_status": load_result.implementation_status,
        "loader_notes": list(load_result.notes or []),
        "module_reference": config.inspect_module_reference,
        "module_selection_applied": False,
        "module_targeting": None,
    }

    if not config.inspect_module_reference:
        summary["inspection_status"] = "not_requested"
        return summary

    try:
        module = load_module_from_reference(config.inspect_module_reference)
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
    except Exception as exc:  # noqa: BLE001
        summary["inspection_status"] = "failed"
        summary["inspection_error"] = str(exc)

    return summary


def inspect_stage2_backbone_targets(
    *,
    backbone_name: str,
    module_reference: str,
    include_patterns: list[str],
    exclude_patterns: list[str] | None = None,
    limit: int = 200,
    apply_selection: bool = False,
) -> dict[str, Any]:
    module = load_module_from_reference(module_reference)
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

    return {
        "backbone_name": backbone_name,
        "backbone_family": infer_backbone_family(backbone_name),
        "module_reference": module_reference,
        "apply_selection": apply_selection,
        "targeting": targeting.to_dict(),
        "notes": [
            "This utility inspects an explicitly provided torch module tree.",
            "It does not imply that full backbone loading or full Stage 2 training is wired for this backbone family.",
        ],
    }


def _build_trainer_plan(
    config: Stage2TrainConfig,
    manifest_path: str,
    num_pairs: int,
    component_plan: dict[str, Any],
) -> dict[str, Any]:
    return {
        "stage": "stage2_v1",
        "objective": "text-conditioning-focused adaptation of the selected generative backbone using real-image + Stage-1-canonical-caption pairs",
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
            "adapter_target_plan": "implemented_metadata_only",
            "real_module_target_selection": "optional_when_explicit_module_reference_is_provided",
            "placeholder_loop": "optional",
            "full_flux_kontext_finetuning": "not_implemented",
        },
        "notes": [
            "This scaffold is intentionally conservative.",
            "Stage 2 no longer means render; render belongs to Stage 1.",
            "Current code treats Stage 2 as generative-backbone adaptation with text-conditioning focus.",
            "Current FLUX.1 Kontext references are assumption labels for planning, not proof that module surgery is implemented.",
        ],
    }


def _infer_backbone_assumptions(backbone_name: str) -> dict[str, Any]:
    family = infer_backbone_family(backbone_name)
    if family == "flux_kontext":
        return {
            "family": "flux_kontext",
            "notes": [
                "Current target family is experimental FLUX.1 Kontext [dev].",
                "Text-conditioning-related module groups are represented conservatively as planning labels/patterns.",
            ],
        }
    return {
        "family": "generic_diffusion_backbone",
        "notes": [
            "Stage 2 wording stays generic at the method level.",
            "Module-group selectors may need replacement for a different backbone family.",
        ],
    }
