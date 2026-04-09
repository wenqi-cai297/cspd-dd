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


def derive_stage2_baseline_sample_output_dir(dataset_root: str | os.PathLike[str], backbone_name: str, *, timestamp: str | None = None) -> str:
    """Return the default run directory for standalone baseline sampling outputs."""
    resolved_timestamp = timestamp or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    dataset_label = derive_stage2_dataset_label(dataset_root)
    backbone_slug = sanitize_stage2_backbone_slug(backbone_name)
    return str(Path("runs") / "stage2" / "baseline_samples" / dataset_label / backbone_slug / resolved_timestamp)


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
    backbone_name: str = "black-forest-labs/FLUX.1-Kontext-dev"
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
    sample_every: int = 0
    sample_prompt_file: str | None = None
    sample_prompts: list[str] = field(default_factory=list)
    sample_num_prompts: int = 4
    sample_num_inference_steps: int = 50
    sample_guidance_scale: float = 7.0
    sample_seed: int = 42
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
    pixart_sigma_prompt_dropout_prob: float = 0.1
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
    full_update_fp32_for_pixart: bool = True
    lora_fp32_for_pixart: bool = True


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


def _read_sample_prompts_from_file(path: str | Path) -> list[str]:
    prompt_path = Path(path)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Sample prompt file not found: {prompt_path}")
    suffix = prompt_path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        rows: list[str] = []
        with prompt_path.open("r", encoding="utf-8-sig") as handle:
            if suffix == ".json":
                payload = json.load(handle)
                if isinstance(payload, list):
                    for item in payload:
                        if isinstance(item, str) and item.strip():
                            rows.append(item.strip())
                        elif isinstance(item, dict):
                            prompt = str(item.get("prompt") or item.get("text") or "").strip()
                            if prompt:
                                rows.append(prompt)
                else:
                    raise ValueError(f"Expected a list in sample prompt JSON file: {prompt_path}")
            else:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    payload = json.loads(line)
                    if isinstance(payload, str) and payload.strip():
                        rows.append(payload.strip())
                    elif isinstance(payload, dict):
                        prompt = str(payload.get("prompt") or payload.get("text") or "").strip()
                        if prompt:
                            rows.append(prompt)
        return rows
    prompts: list[str] = []
    with prompt_path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                prompts.append(line)
    return prompts


def _resolve_sample_prompts(*, config: Stage2TrainConfig, pairs: list[Any]) -> list[str]:
    prompts: list[str] = []
    if config.sample_prompt_file:
        prompts.extend(_read_sample_prompts_from_file(config.sample_prompt_file))
    prompts.extend([prompt.strip() for prompt in config.sample_prompts if str(prompt).strip()])
    if not prompts:
        for pair in pairs:
            caption = str(getattr(pair, "canonical_caption", "") or "").strip()
            if caption:
                prompts.append(caption)
    deduped: list[str] = []
    seen: set[str] = set()
    for prompt in prompts:
        if prompt not in seen:
            deduped.append(prompt)
            seen.add(prompt)
    limit = max(int(config.sample_num_prompts), 0) or len(deduped)
    return deduped[:limit]


def _prompt_slug(prompt: str, *, limit: int = 48) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", prompt.strip()).strip("_").lower()
    if not cleaned:
        cleaned = hashlib.md5(prompt.encode("utf-8")).hexdigest()[:8]
    return cleaned[:limit]


def _save_pil_like_image(image: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(image, "save"):
        image.save(path)
        return path
    try:
        from PIL import Image
        import numpy as np
        array = image
        if hasattr(array, "detach"):
            array = array.detach().cpu().numpy()
        array = np.asarray(array)
        if array.ndim == 3 and array.shape[0] in {1, 3}:
            array = np.transpose(array, (1, 2, 0))
        if array.dtype != np.uint8:
            array = np.clip(array, 0.0, 1.0)
            array = (array * 255.0).astype(np.uint8)
        Image.fromarray(array).save(path)
        return path
    except Exception as exc:
        raise RuntimeError(f"Could not save sample image to {path}: {exc}") from exc


def _run_pixart_wandb_sampling(*, pipeline: Any, transformer: Any, accelerator: Any | None, config: Stage2TrainConfig, run_dir: Path, epoch: int, optimizer_step: int, prompts: list[str], device: Any, train_dtype: Any, wandb_run: Any | None) -> dict[str, Any]:
    result: dict[str, Any] = {"enabled": True, "attempted": True, "optimizer_step": optimizer_step, "epoch": epoch, "prompt_count": len(prompts)}
    if not prompts:
        result.update({"status": "skipped", "reason": "no_prompts"})
        return result
    import torch
    sample_dir = run_dir / "samples" / f"step_{optimizer_step:06d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    resolved_transformer = accelerator.unwrap_model(transformer) if accelerator is not None else transformer
    original_transformer = pipeline.transformer
    training_mode = resolved_transformer.training if hasattr(resolved_transformer, "training") else None
    resolved_transformer.eval()
    pipeline.transformer = resolved_transformer
    image_paths: list[Path] = []
    table_rows: list[list[Any]] = []
    try:
        generator = torch.Generator(device=device.type if hasattr(device, "type") else "cpu")
        generator.manual_seed(int(config.sample_seed) + int(optimizer_step))
        with torch.no_grad():
            outputs = pipeline(
                prompt=prompts,
                num_inference_steps=max(int(config.sample_num_inference_steps), 1),
                guidance_scale=float(config.sample_guidance_scale),
                height=int(config.resolution),
                width=int(config.resolution),
                generator=generator,
                output_type="pil",
            )
        images = list(getattr(outputs, "images", []) or [])
        if len(images) != len(prompts):
            raise RuntimeError(f"PixArt sampling returned {len(images)} images for {len(prompts)} prompts")
        result["status"] = "completed"
        result["sample_dir"] = str(sample_dir.resolve())
        for index, (prompt, image) in enumerate(zip(prompts, images), start=1):
            image_path = sample_dir / f"{index:02d}_{_prompt_slug(prompt)}.png"
            _save_pil_like_image(image, image_path)
            image_paths.append(image_path)
            table_rows.append([optimizer_step, epoch, prompt, str(image_path.resolve())])
        if wandb_run is not None:
            wandb = _try_import_wandb()
            if wandb is not None:
                images_payload = [wandb.Image(str(image_path), caption=prompt) for image_path, prompt in zip(image_paths, prompts)]
                _wandb_log(wandb_run, {"samples/images": images_payload}, step=optimizer_step)
                table = wandb.Table(columns=["optimizer_step", "epoch", "prompt", "image_path"], data=table_rows)
                _wandb_log(wandb_run, {"samples/table": table}, step=optimizer_step)
    except Exception as exc:
        result.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
    finally:
        pipeline.transformer = original_transformer
        if training_mode is not None:
            resolved_transformer.train(training_mode)
    result["saved_images"] = [str(path.resolve()) for path in image_paths]
    return result


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
    """Dispatch standalone baseline sampling to the PixArt family module."""
    from cspd_stage2.families.pixart.training import run_stage2_pixart_baseline_sampling as _impl

    return _impl(
        dataset_root=dataset_root,
        backbone_name=backbone_name,
        output_dir=output_dir,
        sample_prompt_file=sample_prompt_file,
        sample_prompts=sample_prompts,
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


def _should_force_full_update_fp32(*, config: Stage2TrainConfig) -> bool:
    family = infer_backbone_family(config.backbone_name)
    parameterization = str(getattr(config, "training_parameterization", "full")).strip().lower()
    return bool(config.full_update_fp32_for_pixart and parameterization == "full" and family in {"pixart", "pixart_sigma"})


def _resolve_lora_master_weight_dtype(*, config: Stage2TrainConfig) -> str | None:
    family = infer_backbone_family(config.backbone_name)
    parameterization = str(getattr(config, "training_parameterization", "full")).strip().lower()
    if parameterization != "lora":
        return None
    if family in {"pixart", "pixart_sigma"} and bool(getattr(config, "lora_fp32_for_pixart", True)):
        return "float32"
    return None


def _upcast_trainable_parameters_(module: Any, *, dtype: Any, exclude_patterns: list[str] | None = None) -> dict[str, Any]:
    converted_parameter_names: list[str] = []
    skipped_parameter_names: list[str] = []
    converted_parameter_count = 0
    converted_value_count = 0
    exclude_patterns = list(dict.fromkeys(exclude_patterns or []))
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        if exclude_patterns and any(fnmatch.fnmatchcase(name, pattern) for pattern in exclude_patterns):
            skipped_parameter_names.append(name)
            continue
        if parameter.dtype == dtype:
            continue
        parameter.data = parameter.data.to(dtype=dtype)
        if parameter.grad is not None:
            parameter.grad = parameter.grad.to(dtype=dtype)
        converted_parameter_names.append(name)
        converted_parameter_count += 1
        converted_value_count += int(parameter.numel())
    return {
        "enabled": converted_parameter_count > 0,
        "target_dtype": _torch_dtype_label(dtype),
        "converted_parameter_count": converted_parameter_count,
        "converted_value_count": converted_value_count,
        "converted_parameter_names_sample": converted_parameter_names[:20],
        "excluded_parameter_patterns": exclude_patterns,
        "skipped_parameter_count": len(skipped_parameter_names),
        "skipped_parameter_names_sample": skipped_parameter_names[:20],
    }


def _resolve_pixart_partial_full_update_fp32_exclude_patterns(*, config: Stage2TrainConfig) -> list[str]:
    family = infer_backbone_family(config.backbone_name)
    if family not in {"pixart", "pixart_sigma"}:
        return []
    if not _should_force_full_update_fp32(config=config):
        return []
    selection = resolve_effective_module_selection(config)
    if selection["selection_is_full_transformer"]:
        return []
    return ["adaln_single.*"]


def _infer_trainable_parameter_dtype(module: Any, *, fallback: Any) -> Any:
    for parameter in module.parameters():
        if parameter.requires_grad:
            return parameter.dtype
    return fallback


def _infer_module_parameter_dtype(module: Any | None, *, fallback: Any) -> Any:
    if module is None or not hasattr(module, "parameters"):
        return fallback
    for parameter in module.parameters():
        return parameter.dtype
    return fallback


def _infer_pixart_input_boundary_dtypes(transformer: Any, *, fallback: Any) -> dict[str, Any]:
    hidden_states_dtype = _infer_module_parameter_dtype(getattr(transformer, "pos_embed", None), fallback=fallback)
    encoder_hidden_states_dtype = _infer_module_parameter_dtype(
        getattr(transformer, "caption_projection", None),
        fallback=_infer_module_parameter_dtype(getattr(transformer, "context_embedder", None), fallback=hidden_states_dtype),
    )
    added_cond_kwargs_dtype = _infer_module_parameter_dtype(getattr(transformer, "adaln_single", None), fallback=hidden_states_dtype)
    return {
        "hidden_states": hidden_states_dtype,
        "encoder_hidden_states": encoder_hidden_states_dtype,
        "added_cond_kwargs": added_cond_kwargs_dtype,
    }


def _prepare_pixart_forward_inputs(
    *,
    transformer: Any,
    noisy_latents: Any,
    prompt_embeds: Any,
    device: Any,
    train_dtype: Any,
) -> dict[str, Any]:
    import torch

    boundary_dtypes = _infer_pixart_input_boundary_dtypes(transformer, fallback=train_dtype)
    batch_size = int(noisy_latents.shape[0])
    latent_height = int(noisy_latents.shape[-2])
    latent_width = int(noisy_latents.shape[-1])
    aspect_ratio_value = float(latent_width) / float(latent_height) if latent_height > 0 else 1.0
    added_cond_dtype = boundary_dtypes["added_cond_kwargs"]
    resolution = torch.tensor([[latent_height * 8.0, latent_width * 8.0]], device=device, dtype=added_cond_dtype).repeat(batch_size, 1)
    aspect_ratio = torch.tensor([[aspect_ratio_value]], device=device, dtype=added_cond_dtype).repeat(batch_size, 1)
    return {
        "hidden_states": noisy_latents.to(device=device, dtype=boundary_dtypes["hidden_states"]),
        "encoder_hidden_states": prompt_embeds.to(device=device, dtype=boundary_dtypes["encoder_hidden_states"]),
        "added_cond_kwargs": {"resolution": resolution, "aspect_ratio": aspect_ratio},
        "dtype_plan": {key: _torch_dtype_label(value) for key, value in boundary_dtypes.items()},
        "latent_hw": [latent_height, latent_width],
    }


def _build_optimizer(*, parameters: list[Any], config: Stage2TrainConfig, torch_module: Any) -> Any:
    optimizer_name = config.optimizer_name.strip().lower()
    if optimizer_name != "adamw":
        raise ValueError(f"Unsupported optimizer_name for real Stage 2 training: {config.optimizer_name}")
    return torch_module.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay,
        eps=config.adam_epsilon,
    )


def _build_lr_scheduler(*, optimizer: Any, config: Stage2TrainConfig, total_optimizer_steps: int | None) -> tuple[Any | None, dict[str, Any]]:
    if optimizer is None:
        return None, {"name": config.lr_scheduler, "enabled": False, "reason": "optimizer_missing"}
    scheduler_name = config.lr_scheduler.strip().lower()
    warmup_steps = max(int(config.lr_warmup_steps), 0)
    effective_total_steps = None if total_optimizer_steps is None else max(int(total_optimizer_steps), 1)
    if scheduler_name == "constant":
        return None, {"name": scheduler_name, "enabled": False, "warmup_steps": 0}
    if scheduler_name == "constant_with_warmup":
        from torch.optim.lr_scheduler import LambdaLR

        def lr_lambda(step: int) -> float:
            if warmup_steps <= 0:
                return 1.0
            return min(float(step + 1) / float(warmup_steps), 1.0)

        return LambdaLR(optimizer, lr_lambda=lr_lambda), {
            "name": scheduler_name,
            "enabled": True,
            "warmup_steps": warmup_steps,
            "total_optimizer_steps": effective_total_steps,
        }
    raise ValueError(f"Unsupported lr_scheduler for real Stage 2 training: {config.lr_scheduler}")


def _maybe_apply_conditioning_dropout(conditioning_text: list[str], *, config: Stage2TrainConfig, torch_module: Any) -> tuple[list[str], int]:
    family = infer_backbone_family(config.backbone_name)
    if family not in {"pixart", "pixart_sigma"}:
        return list(conditioning_text), 0
    dropout_prob = float(config.pixart_sigma_prompt_dropout_prob)
    if dropout_prob <= 0.0:
        return list(conditioning_text), 0
    dropped_count = 0
    conditioned: list[str] = []
    for text in conditioning_text:
        if float(torch_module.rand(1).item()) < dropout_prob:
            conditioned.append("")
            dropped_count += 1
        else:
            conditioned.append(text)
    return conditioned, dropped_count


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
    """Dispatch FLUX-family training to the dedicated family module."""
    from cspd_stage2.families.flux.training import run_real_stage2_flux_training as _impl

    return _impl(config=config, pairs=pairs, run_dir=run_dir, manifest_path=manifest_path)



def run_real_stage2_pixart_training(
    *,
    config: Stage2TrainConfig,
    pairs: list[Any],
    run_dir: Path,
    manifest_path: str,
) -> dict[str, Any]:
    """Dispatch PixArt-family training to the dedicated family module."""
    from cspd_stage2.families.pixart.training import run_real_stage2_pixart_training as _impl

    return _impl(config=config, pairs=pairs, run_dir=run_dir, manifest_path=manifest_path)



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
    """Backward-compatible wrapper around the PixArt family step implementation."""
    from cspd_stage2.families.pixart.training import _run_real_pixart_train_step as _impl

    return _impl(
        pipeline=pipeline,
        transformer=transformer,
        batch=batch,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        device=device,
        train_dtype=train_dtype,
        memory_log_path=memory_log_path,
        epoch=epoch,
        global_step=global_step,
        optimizer_step=optimizer_step,
        config=config,
    )



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
    """Backward-compatible wrapper around the FLUX family step implementation."""
    from cspd_stage2.families.flux.training import _run_real_flux_train_step as _impl

    return _impl(
        pipeline=pipeline,
        transformer=transformer,
        batch=batch,
        optimizer=optimizer,
        accelerator=accelerator,
        device=device,
        train_dtype=train_dtype,
        resolution=resolution,
        memory_log_path=memory_log_path,
        epoch=epoch,
        global_step=global_step,
        optimizer_step=optimizer_step,
    )




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
        adapter_master_weight_dtype = adapter_plan.master_weight_dtype or _resolve_lora_master_weight_dtype(config=config)
        adapter_injection = inject_lora_adapters(
            transformer,
            include_patterns=target_patterns,
            exclude_patterns=adapter_plan.exclude_module_patterns,
            rank=adapter_plan.rank,
            alpha=adapter_plan.alpha,
            dropout=adapter_plan.dropout,
            adapter_dtype=adapter_master_weight_dtype,
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
            "Frozen VAE/text components now stay on the active runtime device for the whole training run; Stage 2 no longer shuttles them between CPU and GPU.",
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
