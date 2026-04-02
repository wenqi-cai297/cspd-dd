from __future__ import annotations

"""Generic Stage 2 generative-backbone helpers.

This module stays honest about scope:
- resolves backbone-family assumptions from a name,
- provides a lightweight loader hook contract,
- applies include/exclude targeting rules to a real torch module tree when available,
- offers a small inspection surface for candidate module names,
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
    loader_name: str | None = None
    notes: list[str] | None = None


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


def infer_backbone_family(backbone_name: str) -> str:
    lowered = backbone_name.lower()
    if "flux" in lowered and "kontext" in lowered:
        return "flux_kontext"
    return "generic_diffusion_backbone"


def load_generative_backbone(
    backbone_name: str,
    *,
    loader: str | None = None,
    allow_unimplemented: bool = True,
) -> BackboneLoadResult:
    """Resolve a conservative loader hook.

    This function intentionally avoids implicit network downloads or claiming that
    a specific FLUX Kontext runtime is already wired. It exposes an honest load
    contract so later code can plug in a real loader without changing the Stage 2
    planning surface.
    """

    family = infer_backbone_family(backbone_name)
    loader_name = loader or ("diffusers_flux_kontext" if family == "flux_kontext" else "generic_python_loader")

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

    if loader_name == "diffusers_flux_kontext":
        has_torch = importlib.util.find_spec("torch") is not None
        has_diffusers = importlib.util.find_spec("diffusers") is not None
        if not has_torch or not has_diffusers:
            status = "dependency_missing"
            notes = [
                "FLUX Kontext loader hook selected, but required dependencies are missing.",
                f"torch_installed={has_torch}",
                f"diffusers_installed={has_diffusers}",
                "No fake model was created.",
            ]
        else:
            status = "not_implemented"
            notes = [
                "Dependency probes passed, but a concrete FLUX Kontext backbone loader is still not implemented in this repo.",
                "This guard is intentional: Stage 2 should not pretend full backbone loading exists when it does not.",
            ]
        if not allow_unimplemented and status != "loaded":
            raise NotImplementedError("Concrete generative-backbone loading is not implemented for this backbone yet.")
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


def _named_modules(module: Any) -> list[tuple[str, Any]]:
    _ensure_torch_module(module)
    return list(module.named_modules())


def _ensure_torch_module(module: Any) -> None:
    if not hasattr(module, "named_modules") or not hasattr(module, "named_parameters"):
        raise TypeError("Expected an object compatible with torch.nn.Module for module inspection/targeting")


def _matched_patterns(name: str, patterns: list[str]) -> list[str]:
    return [pattern for pattern in patterns if fnmatch.fnmatchcase(name, pattern)]
