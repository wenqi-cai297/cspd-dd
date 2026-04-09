from __future__ import annotations

"""Shared Stage 2 training helpers that are family-neutral."""

import fnmatch
import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from cspd_stage1.io_utils import write_json
from cspd_stage2.backbone import apply_trainable_parameter_selection, infer_backbone_family, inject_lora_adapters, inspect_target_modules
from cspd_stage2.families.pixart.backbone import resolve_pixart_conditioning_transformer_patterns

DEFAULT_TEXT_CONDITIONING_GROUPS = ["full_transformer"]
DEFAULT_LORA_TARGET_GROUPS = ["conditioning_transformer"]
DEFAULT_EXCLUDE_PATTERNS = ["vae", "autoencoder", "decoder", "image_encoder"]

CONDITIONING_RELATED_GROUP_PATTERNS = {
    "full_transformer": ["*"],
    "conditioning_context_embedder": ["context_embedder", "context_embedder.*"],
    "conditioning_time_text_embed": ["time_text_embed*", "time_text_embed*.*"],
    "conditioning_norm1_context": ["transformer_blocks.*.norm1_context*", "transformer_blocks.*.norm1_context*.*"],
    "conditioning_added_kv_attention": [
        "transformer_blocks.*.attn.add_q_proj",
        "transformer_blocks.*.attn.add_k_proj",
        "transformer_blocks.*.attn.add_v_proj",
        "transformer_blocks.*.attn.to_add_out",
        "transformer_blocks.*.attn.to_add_out.*",
    ],
    "conditioning_ff_context": ["transformer_blocks.*.ff_context*", "transformer_blocks.*.ff_context*.*"],
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

SPLIT_ONLY_DATASET_ROOT_NAMES = {"train", "val", "valid", "validation", "test", "testing"}


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
    return str(Path("runs") / "stage2" / "train" / derive_stage2_dataset_label(dataset_root) / sanitize_stage2_backbone_slug(backbone_name) / resolved_timestamp)


def derive_stage2_baseline_sample_output_dir(dataset_root: str | os.PathLike[str], backbone_name: str, *, timestamp: str | None = None) -> str:
    resolved_timestamp = timestamp or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return str(Path("runs") / "stage2" / "baseline_samples" / derive_stage2_dataset_label(dataset_root) / sanitize_stage2_backbone_slug(backbone_name) / resolved_timestamp)


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


def _build_optimizer(*, parameters: list[Any], config: Any, torch_module: Any) -> Any:
    optimizer_name = config.optimizer_name.strip().lower()
    if optimizer_name != "adamw":
        raise ValueError(f"Unsupported optimizer_name for real Stage 2 training: {config.optimizer_name}")
    return torch_module.optim.AdamW(parameters, lr=config.learning_rate, betas=(config.adam_beta1, config.adam_beta2), weight_decay=config.adam_weight_decay, eps=config.adam_epsilon)


def _build_lr_scheduler(*, optimizer: Any, config: Any, total_optimizer_steps: int | None) -> tuple[Any | None, dict[str, Any]]:
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
        return LambdaLR(optimizer, lr_lambda=lr_lambda), {"name": scheduler_name, "enabled": True, "warmup_steps": warmup_steps, "total_optimizer_steps": effective_total_steps}
    raise ValueError(f"Unsupported lr_scheduler for real Stage 2 training: {config.lr_scheduler}")


def _should_force_full_update_fp32(*, config: Any) -> bool:
    family = infer_backbone_family(config.backbone_name)
    parameterization = str(getattr(config, "training_parameterization", "full")).strip().lower()
    return bool(config.full_update_fp32_for_pixart and parameterization == "full" and family in {"pixart", "pixart_sigma"})


def _resolve_lora_master_weight_dtype(*, config: Any) -> str | None:
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
    return {"enabled": converted_parameter_count > 0, "target_dtype": _torch_dtype_label(dtype), "converted_parameter_count": converted_parameter_count, "converted_value_count": converted_value_count, "converted_parameter_names_sample": converted_parameter_names[:20], "excluded_parameter_patterns": exclude_patterns, "skipped_parameter_count": len(skipped_parameter_names), "skipped_parameter_names_sample": skipped_parameter_names[:20]}


def _resolve_trainable_component_groups(config: Any) -> list[str]:
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


def resolve_effective_module_selection(config: Any) -> dict[str, Any]:
    groups = _resolve_trainable_component_groups(config)
    group_patterns, unknown_groups = _expand_component_group_patterns(groups)
    manual_patterns = list(dict.fromkeys(config.module_include_patterns or []))
    effective_include_patterns = list(dict.fromkeys(group_patterns + manual_patterns))
    family = infer_backbone_family(config.backbone_name)
    if family in {"pixart_sigma", "pixart"} and not manual_patterns and groups == ["conditioning_transformer"]:
        effective_include_patterns = resolve_pixart_conditioning_transformer_patterns()
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


def _freeze_stage2_modules(pipeline: Any, config: Any) -> dict[str, Any]:
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
        adapter_injection = inject_lora_adapters(transformer, include_patterns=target_patterns, exclude_patterns=adapter_plan.exclude_module_patterns, rank=adapter_plan.rank, alpha=adapter_plan.alpha, dropout=adapter_plan.dropout, adapter_dtype=adapter_master_weight_dtype)
        targeting = inspect_target_modules(transformer, include_patterns=target_patterns, exclude_patterns=adapter_plan.exclude_module_patterns, limit=None)
    else:
        if config.train_transformer_core_only:
            for parameter in transformer.parameters():
                parameter.requires_grad = True
        if selection["should_apply_real_transformer_selection"]:
            targeting = apply_trainable_parameter_selection(transformer, include_patterns=selection["effective_include_patterns"], exclude_patterns=selection["effective_exclude_patterns"])
        else:
            targeting = inspect_target_modules(transformer, include_patterns=selection["effective_include_patterns"], exclude_patterns=selection["effective_exclude_patterns"], limit=None)
        if not config.freeze_text_encoder:
            for component_name in ["text_encoder", "text_encoder_2"]:
                component = getattr(pipeline, component_name, None)
                if component is not None and hasattr(component, "parameters"):
                    for parameter in component.parameters():
                        parameter.requires_grad = True
        if not config.freeze_vae and getattr(pipeline, "vae", None) is not None:
            for parameter in pipeline.vae.parameters():
                parameter.requires_grad = True
    from cspd_stage2.training import _summarize_trainable_parameters
    return {"parameterization": parameterization, "selection": targeting, "adapter_injection": adapter_injection, "trainable_parameter_summary": _summarize_trainable_parameters(pipeline)}


def _safe_write_json(path: Path, payload: dict[str, Any]) -> None:
    try:
        write_json(path, payload)
    except Exception:
        fallback = _safe_jsonable(payload)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(fallback, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def _torch_dtype_label(dtype: Any) -> str:
    name = getattr(dtype, "name", None)
    if name:
        return str(name)
    value = str(dtype)
    return value.replace("torch.", "")
