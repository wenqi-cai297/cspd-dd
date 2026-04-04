from __future__ import annotations

"""Generic Stage 2 generative-backbone helpers.

This module stays honest about scope:
- resolves backbone-family assumptions from a name,
- provides a real loader path when the local dependency/runtime stack supports it,
- applies include/exclude targeting rules to a real torch module tree when available,
- offers a small inspection surface for candidate module names,
- supports optional adapter injection on explicitly provided torch modules,
- does not pretend full FLUX.1 Kontext training is already integrated.
"""

from dataclasses import dataclass
import fnmatch
import importlib
import importlib.util
from typing import Any


@dataclass(slots=True)
class BackboneLoadResult:
    backbone_name: str
    family: str
    implementation_status: str
    module: Any = None
    root_module: Any = None
    loader_name: str | None = None
    notes: list[str] | None = None
    resolved_module_name: str | None = None
    resolved_module_type: str | None = None


@dataclass(slots=True)
class ModuleTargetMatch:
    module_name: str
    module_type: str
    parameter_names: list[str]
    parameter_count: int
    trainable_parameter_count: int
    matched_include_patterns: list[str]
    matched_exclude_patterns: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "module_name": self.module_name,
            "module_type": self.module_type,
            "parameter_names": list(self.parameter_names),
            "parameter_count": self.parameter_count,
            "trainable_parameter_count": self.trainable_parameter_count,
            "matched_include_patterns": list(self.matched_include_patterns),
            "matched_exclude_patterns": list(self.matched_exclude_patterns),
        }


@dataclass(slots=True)
class ModuleTargetingResult:
    total_modules_seen: int
    matched_modules: list[ModuleTargetMatch]
    include_patterns: list[str]
    exclude_patterns: list[str]
    selected_module_names: list[str]
    selected_parameter_names: list[str]
    selected_parameter_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_modules_seen": self.total_modules_seen,
            "matched_modules": [item.to_dict() for item in self.matched_modules],
            "include_patterns": list(self.include_patterns),
            "exclude_patterns": list(self.exclude_patterns),
            "selected_module_names": list(self.selected_module_names),
            "selected_parameter_names": list(self.selected_parameter_names),
            "selected_parameter_count": self.selected_parameter_count,
        }


@dataclass(slots=True)
class AdapterInjectionMatch:
    module_name: str
    original_module_type: str
    injected_module_type: str
    rank: int
    alpha: float
    dropout: float
    base_parameter_count: int
    adapter_parameter_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "module_name": self.module_name,
            "original_module_type": self.original_module_type,
            "injected_module_type": self.injected_module_type,
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "base_parameter_count": self.base_parameter_count,
            "adapter_parameter_count": self.adapter_parameter_count,
        }


@dataclass(slots=True)
class AdapterInjectionResult:
    adapter_type: str
    target_module_type: str
    include_patterns: list[str]
    exclude_patterns: list[str]
    attempted_module_names: list[str]
    injected_modules: list[AdapterInjectionMatch]
    skipped_modules: list[dict[str, Any]]
    total_adapter_parameter_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_type": self.adapter_type,
            "target_module_type": self.target_module_type,
            "include_patterns": list(self.include_patterns),
            "exclude_patterns": list(self.exclude_patterns),
            "attempted_module_names": list(self.attempted_module_names),
            "injected_modules": [item.to_dict() for item in self.injected_modules],
            "skipped_modules": list(self.skipped_modules),
            "total_adapter_parameter_count": self.total_adapter_parameter_count,
        }


class LoRALinearAdapter:  # instantiated only when torch exists
    def __init__(self, base_layer: Any, *, rank: int, alpha: float, dropout: float) -> None:
        import torch

        if rank <= 0:
            raise ValueError("Adapter rank must be positive")
        if not hasattr(base_layer, "in_features") or not hasattr(base_layer, "out_features"):
            raise TypeError("LoRALinearAdapter requires a linear-like module with in_features/out_features")

        self._torch = torch
        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(rank)
        self.dropout = torch.nn.Dropout(dropout) if dropout and dropout > 0 else torch.nn.Identity()
        self.lora_A = torch.nn.Linear(base_layer.in_features, rank, bias=False)
        self.lora_B = torch.nn.Linear(rank, base_layer.out_features, bias=False)
        torch.nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        torch.nn.init.zeros_(self.lora_B.weight)

        for parameter in self.base_layer.parameters():
            parameter.requires_grad = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self.__dict__["base_layer"], name)

    def forward(self, inputs: Any) -> Any:
        base = self.base_layer(inputs)
        update = self.lora_B(self.lora_A(self.dropout(inputs))) * self.scaling
        return base + update

    def parameters(self, recurse: bool = True):
        for parameter in self.base_layer.parameters(recurse=recurse):
            yield parameter
        for parameter in self.lora_A.parameters(recurse=recurse):
            yield parameter
        for parameter in self.lora_B.parameters(recurse=recurse):
            yield parameter

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        for name, parameter in self.base_layer.named_parameters(prefix=f"{prefix}.base_layer" if prefix else "base_layer", recurse=recurse):
            yield name, parameter
        for name, parameter in self.lora_A.named_parameters(prefix=f"{prefix}.lora_A" if prefix else "lora_A", recurse=recurse):
            yield name, parameter
        for name, parameter in self.lora_B.named_parameters(prefix=f"{prefix}.lora_B" if prefix else "lora_B", recurse=recurse):
            yield name, parameter


if importlib.util.find_spec("torch") is not None:
    import torch

    class LoRALinearAdapter(torch.nn.Module):
        def __init__(self, base_layer: Any, *, rank: int, alpha: float, dropout: float) -> None:
            super().__init__()
            if rank <= 0:
                raise ValueError("Adapter rank must be positive")
            if not isinstance(base_layer, torch.nn.Linear):
                raise TypeError("LoRALinearAdapter currently supports torch.nn.Linear modules only")

            self.base_layer = base_layer
            self.rank = int(rank)
            self.alpha = float(alpha)
            self.scaling = float(alpha) / float(rank)
            self.dropout = torch.nn.Dropout(dropout) if dropout and dropout > 0 else torch.nn.Identity()
            self.lora_A = torch.nn.Linear(base_layer.in_features, rank, bias=False)
            self.lora_B = torch.nn.Linear(rank, base_layer.out_features, bias=False)
            torch.nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
            torch.nn.init.zeros_(self.lora_B.weight)

            for parameter in self.base_layer.parameters():
                parameter.requires_grad = False

        def forward(self, inputs: Any) -> Any:
            base = self.base_layer(inputs)
            update = self.lora_B(self.lora_A(self.dropout(inputs))) * self.scaling
            return base + update


def infer_backbone_family(backbone_name: str) -> str:
    lowered = backbone_name.lower()
    if "pixart" in lowered and ("sigma" in lowered or "pixart-sigma" in lowered):
        return "pixart_sigma"
    if "pixart" in lowered:
        return "pixart"
    if "flux" in lowered and "kontext" in lowered:
        return "flux_kontext"
    if "flux" in lowered:
        return "flux"
    return "generic_diffusion_backbone"


def load_generative_backbone(
    backbone_name: str,
    *,
    loader: str | None = None,
    allow_unimplemented: bool = True,
    torch_dtype: str = "bfloat16",
    device: str | None = None,
    device_map: str | None = None,
    local_files_only: bool = False,
    component: str | None = None,
) -> BackboneLoadResult:
    """Resolve a conservative loader hook, but use a real diffusers path when possible."""

    family = infer_backbone_family(backbone_name)
    loader_name = loader or _default_loader_name(family)

    if loader_name == "generic_python_loader":
        return BackboneLoadResult(
            backbone_name=backbone_name,
            family=family,
            implementation_status="not_loaded",
            loader_name=loader_name,
            notes=[
                "Generic loader hook selected.",
                "Provide a concrete torch.nn.Module instance separately for module inspection/targeting.",
            ],
        )

    if loader_name in {"diffusers_flux_kontext", "diffusers_flux", "diffusers_pixart_sigma", "diffusers_pixart"}:
        try:
            return _load_diffusers_backbone(
                backbone_name,
                family=family,
                loader_name=loader_name,
                torch_dtype=torch_dtype,
                device=device,
                device_map=device_map,
                local_files_only=local_files_only,
                component=component,
            )
        except Exception as exc:  # noqa: BLE001
            status = _classify_loader_exception(exc)
            notes = [f"Real diffusers load attempt failed: {type(exc).__name__}: {exc}"]
            if status == "dependency_missing":
                notes.append("Install/repair the missing runtime dependencies first.")
            elif status == "unsupported_runtime":
                notes.append("The installed diffusers build likely lacks the required FLUX pipeline class.")
            elif status == "load_failed_local_files_only":
                notes.append("The requested model was not found in the local Hugging Face cache while local_files_only=True.")
            elif status == "auth_required":
                notes.append("A Hugging Face login/token or accepted model license may still be required on this machine.")
            elif status == "download_or_resolution_failed":
                notes.append("The runtime reached a real loader path, but model resolution/download did not succeed.")
            if not allow_unimplemented:
                raise
            return BackboneLoadResult(
                backbone_name=backbone_name,
                family=family,
                implementation_status=status,
                loader_name=loader_name,
                notes=notes,
            )

    if not allow_unimplemented:
        raise ValueError(f"Unknown backbone loader: {loader_name}")
    return BackboneLoadResult(
        backbone_name=backbone_name,
        family=family,
        implementation_status="unknown_loader",
        loader_name=loader_name,
        notes=["Unknown loader label; no backbone was loaded."],
    )


def load_module_from_reference(reference: str) -> Any:
    """Load a Python object from `module.submodule:object_name` reference."""

    if ":" not in reference:
        raise ValueError("Module reference must be in the form 'package.module:object_name'")
    module_name, object_name = reference.split(":", 1)
    module = importlib.import_module(module_name)
    if not hasattr(module, object_name):
        raise AttributeError(f"Object '{object_name}' not found in module '{module_name}'")
    obj = getattr(module, object_name)
    if callable(obj):
        obj = obj()
    return obj


def inspect_target_modules(
    module: Any,
    *,
    include_patterns: list[str],
    exclude_patterns: list[str] | None = None,
    limit: int | None = None,
) -> ModuleTargetingResult:
    """List candidate module names from a real module tree when torch is available."""

    exclude_patterns = list(exclude_patterns or [])
    named_modules = _named_modules(module)
    matched: list[ModuleTargetMatch] = []
    selected_parameter_names: list[str] = []
    selected_parameter_count = 0

    for module_name, submodule in named_modules:
        include_hits = _matched_patterns(module_name, include_patterns)
        exclude_hits = _matched_patterns(module_name, exclude_patterns)
        if not include_hits or exclude_hits:
            continue
        parameter_names = [name for name, _ in submodule.named_parameters(recurse=False)]
        parameter_count = sum(parameter.numel() for _, parameter in submodule.named_parameters(recurse=False))
        trainable_count = sum(
            parameter.numel() for _, parameter in submodule.named_parameters(recurse=False) if getattr(parameter, "requires_grad", False)
        )
        qualified_parameter_names = [f"{module_name}.{name}" if module_name else name for name in parameter_names]
        selected_parameter_names.extend(qualified_parameter_names)
        selected_parameter_count += parameter_count
        matched.append(
            ModuleTargetMatch(
                module_name=module_name,
                module_type=type(submodule).__name__,
                parameter_names=parameter_names,
                parameter_count=int(parameter_count),
                trainable_parameter_count=int(trainable_count),
                matched_include_patterns=include_hits,
                matched_exclude_patterns=exclude_hits,
            )
        )
        if limit is not None and len(matched) >= limit:
            break

    return ModuleTargetingResult(
        total_modules_seen=len(named_modules),
        matched_modules=matched,
        include_patterns=list(include_patterns),
        exclude_patterns=list(exclude_patterns),
        selected_module_names=[item.module_name for item in matched],
        selected_parameter_names=selected_parameter_names,
        selected_parameter_count=int(selected_parameter_count),
    )


def apply_trainable_parameter_selection(
    module: Any,
    *,
    include_patterns: list[str],
    exclude_patterns: list[str] | None = None,
) -> ModuleTargetingResult:
    """Apply include/exclude rules to a real module tree by toggling parameter requires_grad."""

    exclude_patterns = list(exclude_patterns or [])
    _ensure_torch_module(module)

    for _, parameter in module.named_parameters():
        parameter.requires_grad = False

    targeting = inspect_target_modules(
        module,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        limit=None,
    )

    selected_prefixes = set(targeting.selected_module_names)
    for param_name, parameter in module.named_parameters():
        module_name = param_name.rsplit(".", 1)[0] if "." in param_name else ""
        if module_name in selected_prefixes:
            parameter.requires_grad = True

    return inspect_target_modules(
        module,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        limit=None,
    )


def inject_lora_adapters(
    module: Any,
    *,
    include_patterns: list[str],
    exclude_patterns: list[str] | None = None,
    rank: int = 16,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> AdapterInjectionResult:
    """Inject lightweight LoRA adapters into matching real torch submodules."""

    _ensure_torch_module(module)
    exclude_patterns = list(exclude_patterns or [])

    attempted_module_names: list[str] = []
    injected_modules: list[AdapterInjectionMatch] = []
    skipped_modules: list[dict[str, Any]] = []
    total_adapter_parameter_count = 0

    for module_name, submodule in list(module.named_modules()):
        if not module_name:
            continue
        include_hits = _matched_patterns(module_name, include_patterns)
        exclude_hits = _matched_patterns(module_name, exclude_patterns)
        if not include_hits or exclude_hits:
            continue
        attempted_module_names.append(module_name)

        parent_module, child_name = _resolve_parent_module(module, module_name)
        if parent_module is None or child_name is None:
            skipped_modules.append({"module_name": module_name, "reason": "parent_resolution_failed"})
            continue
        target_module = getattr(parent_module, child_name, None)
        if target_module is None:
            skipped_modules.append({"module_name": module_name, "reason": "target_missing"})
            continue

        if type(target_module).__name__ != "Linear":
            skipped_modules.append(
                {
                    "module_name": module_name,
                    "module_type": type(target_module).__name__,
                    "reason": "unsupported_module_type",
                }
            )
            continue

        base_parameter_count = sum(parameter.numel() for parameter in target_module.parameters())
        adapted = LoRALinearAdapter(target_module, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent_module, child_name, adapted)
        adapter_parameter_count = sum(parameter.numel() for name, parameter in adapted.named_parameters() if ".lora_" in name or name.startswith("lora_"))
        total_adapter_parameter_count += int(adapter_parameter_count)
        injected_modules.append(
            AdapterInjectionMatch(
                module_name=module_name,
                original_module_type="Linear",
                injected_module_type=type(adapted).__name__,
                rank=int(rank),
                alpha=float(alpha),
                dropout=float(dropout),
                base_parameter_count=int(base_parameter_count),
                adapter_parameter_count=int(adapter_parameter_count),
            )
        )

    return AdapterInjectionResult(
        adapter_type="lora",
        target_module_type="torch.nn.Linear",
        include_patterns=list(include_patterns),
        exclude_patterns=list(exclude_patterns),
        attempted_module_names=attempted_module_names,
        injected_modules=injected_modules,
        skipped_modules=skipped_modules,
        total_adapter_parameter_count=int(total_adapter_parameter_count),
    )


def load_real_backbone_module(
    backbone_name: str,
    *,
    torch_dtype: str = "bfloat16",
    device: str | None = None,
    device_map: str | None = None,
    local_files_only: bool = False,
    component: str | None = None,
    allow_unimplemented: bool = False,
) -> BackboneLoadResult:
    return load_generative_backbone(
        backbone_name,
        allow_unimplemented=allow_unimplemented,
        torch_dtype=torch_dtype,
        device=device,
        device_map=device_map,
        local_files_only=local_files_only,
        component=component,
    )


def _default_loader_name(family: str) -> str:
    if family == "flux_kontext":
        return "diffusers_flux_kontext"
    if family == "flux":
        return "diffusers_flux"
    if family == "pixart_sigma":
        return "diffusers_pixart_sigma"
    if family == "pixart":
        return "diffusers_pixart"
    return "generic_python_loader"


def _load_diffusers_backbone(
    backbone_name: str,
    *,
    family: str,
    loader_name: str,
    torch_dtype: str,
    device: str | None,
    device_map: str | None,
    local_files_only: bool,
    component: str | None,
) -> BackboneLoadResult:
    if importlib.util.find_spec("torch") is None:
        raise ModuleNotFoundError("torch is not installed")
    if importlib.util.find_spec("diffusers") is None:
        raise ModuleNotFoundError("diffusers is not installed")

    import torch
    import diffusers

    resolved_dtype = _resolve_torch_dtype(torch_dtype)
    if family == "flux_kontext":
        pipeline_class_name = "FluxKontextPipeline"
    elif family == "flux":
        pipeline_class_name = "FluxPipeline"
    elif family == "pixart_sigma":
        pipeline_class_name = "PixArtSigmaPipeline"
    elif family == "pixart":
        pipeline_class_name = "PixArtAlphaPipeline"
    else:
        raise RuntimeError(f"Unsupported diffusers backbone family: {family}")
    if not hasattr(diffusers, pipeline_class_name):
        raise RuntimeError(f"Installed diffusers does not expose {pipeline_class_name}")

    pipeline_class = getattr(diffusers, pipeline_class_name)
    load_kwargs: dict[str, Any] = {
        "torch_dtype": resolved_dtype,
        "local_files_only": local_files_only,
    }
    if device_map:
        load_kwargs["device_map"] = device_map

    if family == "pixart_sigma":
        transformer_class = getattr(diffusers, "Transformer2DModel")
        base_repo = "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers"
        try:
            pipeline = pipeline_class.from_pretrained(backbone_name, **load_kwargs)
        except Exception:
            transformer = transformer_class.from_pretrained(
                backbone_name,
                subfolder="transformer",
                torch_dtype=resolved_dtype,
                local_files_only=local_files_only,
            )
            pipeline = pipeline_class.from_pretrained(base_repo, transformer=transformer, **load_kwargs)
    else:
        pipeline = pipeline_class.from_pretrained(backbone_name, **load_kwargs)
    if device and not device_map and hasattr(pipeline, "to"):
        pipeline = pipeline.to(device)

    resolved_module_name, resolved_module = _resolve_loaded_module(pipeline, requested_component=component)
    notes = [
        f"Loaded via diffusers.{pipeline_class_name}.from_pretrained(...)",
        f"Resolved inspection module: {resolved_module_name} ({type(resolved_module).__name__})",
    ]
    if local_files_only:
        notes.append("local_files_only=True was used for this load attempt.")
    return BackboneLoadResult(
        backbone_name=backbone_name,
        family=family,
        implementation_status="loaded",
        module=resolved_module,
        root_module=pipeline,
        loader_name=loader_name,
        notes=notes,
        resolved_module_name=resolved_module_name,
        resolved_module_type=type(resolved_module).__name__,
    )


def _resolve_torch_dtype(torch_dtype: str) -> Any:
    import torch

    normalized = str(torch_dtype).lower().strip()
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
        raise ValueError(f"Unsupported torch dtype label: {torch_dtype}")
    return mapping[normalized]


def _resolve_loaded_module(pipeline: Any, *, requested_component: str | None) -> tuple[str, Any]:
    if requested_component:
        if not hasattr(pipeline, requested_component):
            raise AttributeError(f"Loaded pipeline does not expose component '{requested_component}'")
        module = getattr(pipeline, requested_component)
        _ensure_torch_module(module)
        return requested_component, module

    for candidate_name in ["transformer", "transformer_2d", "unet", "text_encoder", "text_encoder_2"]:
        candidate = getattr(pipeline, candidate_name, None)
        if candidate is not None and hasattr(candidate, "named_modules") and hasattr(candidate, "named_parameters"):
            return candidate_name, candidate

    if hasattr(pipeline, "named_modules") and hasattr(pipeline, "named_parameters"):
        return "pipeline", pipeline

    raise TypeError("Loaded diffusers pipeline did not expose an inspectable torch module component")


def _classify_loader_exception(exc: Exception) -> str:
    text = str(exc).lower()
    if isinstance(exc, ModuleNotFoundError):
        return "dependency_missing"
    if (
        "does not expose fluxkontextpipeline" in text
        or "does not expose fluxpipeline" in text
        or "does not expose pixartsigmapipeline" in text
        or "does not expose pixartalphapipeline" in text
    ):
        return "unsupported_runtime"
    if "local_files_only" in text or "cannot find the requested files in the disk cache" in text:
        return "load_failed_local_files_only"
    if "401" in text or "403" in text or "gated" in text or "access" in text or "token" in text or "login" in text:
        return "auth_required"
    if "not found" in text or "couldn't connect" in text or "connection" in text or "resolve" in text:
        return "download_or_resolution_failed"
    return "load_failed"


def _named_modules(module: Any) -> list[tuple[str, Any]]:
    _ensure_torch_module(module)
    return list(module.named_modules())


def _ensure_torch_module(module: Any) -> None:
    if not hasattr(module, "named_modules") or not hasattr(module, "named_parameters"):
        raise TypeError("Expected an object compatible with torch.nn.Module for module inspection/targeting")


def _matched_patterns(name: str, patterns: list[str]) -> list[str]:
    return [pattern for pattern in patterns if fnmatch.fnmatchcase(name, pattern)]


def _resolve_parent_module(root_module: Any, qualified_name: str) -> tuple[Any | None, str | None]:
    parts = qualified_name.split(".")
    if not parts:
        return None, None
    parent = root_module
    for part in parts[:-1]:
        if not hasattr(parent, part):
            return None, None
        parent = getattr(parent, part)
    return parent, parts[-1]
