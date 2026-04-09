from __future__ import annotations

"""FLUX-family Stage 2 backbone loading helpers."""

from typing import Any


def infer_flux_family(backbone_name: str) -> str:
    """Infer the FLUX family label from a backbone identifier."""
    lowered = backbone_name.lower()
    if "flux" in lowered and "kontext" in lowered:
        return "flux_kontext"
    return "flux"


def resolve_flux_pipeline_class_name(*, family: str) -> str:
    """Return the diffusers pipeline class name for a FLUX family label."""
    if family == "flux_kontext":
        return "FluxKontextPipeline"
    if family == "flux":
        return "FluxPipeline"
    raise RuntimeError(f"Unsupported FLUX backbone family: {family}")


def default_flux_loader_name(*, family: str) -> str:
    """Return the default shared-loader label for a FLUX family label."""
    if family == "flux_kontext":
        return "diffusers_flux_kontext"
    if family == "flux":
        return "diffusers_flux"
    raise RuntimeError(f"Unsupported FLUX backbone family: {family}")


def load_flux_pipeline(
    backbone_name: str,
    *,
    family: str,
    pipeline_class: Any,
    load_kwargs: dict[str, Any],
) -> Any:
    """Load a FLUX-family diffusers pipeline."""
    del family
    return pipeline_class.from_pretrained(backbone_name, **load_kwargs)
