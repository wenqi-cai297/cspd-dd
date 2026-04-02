from __future__ import annotations

"""Training scaffold for CSPD Stage 2.

This module is deliberately honest about scope:
- it prepares run directories and paired manifests,
- records transformer-core-only adaptation intent,
- exposes a minimal trainer contract,
- optionally runs a tiny PyTorch-backed placeholder loop,
- does not claim full FLUX.1 Kontext [dev] fine-tuning is implemented here.
"""

import importlib.util
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from cspd_stage1.io_utils import write_json
from cspd_stage2.data import build_stage2_pairs, write_pairing_artifacts


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
    trainer_plan = _build_trainer_plan(config, manifest_paths.manifest_path, len(pairing.pairs))
    write_json(run_dir / "stage2_config_snapshot.json", asdict(config))
    write_json(run_dir / "trainer_plan.json", trainer_plan)

    status = "manifest_ready"
    training_result: dict[str, Any] = {
        "status": status,
        "implemented_training": False,
        "placeholder_training": False,
        "message": (
            "Stage 2 paired manifest is ready. Full FLUX.1 Kontext transformer-core fine-tuning is not wired in this repo yet."
        ),
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
                    "a concrete FLUX Kontext training dependency stack is selected and integrated."
                ),
            }

    summary = {
        "stage": "stage2_v1",
        "definition": "generative-backbone adaptation / canonical-semantic-space familiarization",
        "backbone_name": config.backbone_name,
        "run_dir": str(run_dir.resolve()),
        "train_transformer_core_only": config.train_transformer_core_only,
        "freeze_text_encoder": config.freeze_text_encoder,
        "freeze_vae": config.freeze_vae,
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
        }

    import torch

    torch.manual_seed(config.seed)
    model = torch.nn.Linear(8, 8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    max_steps = config.max_steps or min(5, max(config.epochs, 1) * 2)
    losses: list[float] = []

    for step in range(max_steps):
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
    }


def _build_trainer_plan(config: Stage2TrainConfig, manifest_path: str, num_pairs: int) -> dict[str, Any]:
    return {
        "stage": "stage2_v1",
        "objective": "transformer-core adaptation of the selected generative backbone using real-image + Stage-1-canonical-caption pairs",
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
        "freeze_plan": {
            "train_transformer_core_only": config.train_transformer_core_only,
            "freeze_text_encoder": config.freeze_text_encoder,
            "freeze_vae": config.freeze_vae,
        },
        "implementation_status": {
            "pairing_manifest": "implemented",
            "run_directory_setup": "implemented",
            "config_snapshot": "implemented",
            "placeholder_loop": "optional",
            "full_flux_kontext_finetuning": "not_implemented",
        },
        "notes": [
            "This scaffold is intentionally conservative.",
            "Stage 2 no longer means render; render belongs to Stage 1.",
            "Current code treats Stage 2 as transformer-core / generative-backbone adaptation only.",
        ],
    }
