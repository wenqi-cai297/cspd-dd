from __future__ import annotations

"""SDXL-family backbone helpers."""

from typing import Any


def infer_sdxl_family(backbone_name: str) -> str:
    """Infer the Stage 2 family label for SDXL-style backbones."""
    lowered = backbone_name.lower()
    if "sdxl" in lowered or "stable-diffusion-xl" in lowered:
        return "sdxl"
    return "sdxl"


def resolve_sdxl_pipeline_class_name(family: str) -> str:
    """Return the diffusers pipeline class name for an SDXL family label."""
    if family != "sdxl":
        raise RuntimeError(f"Unsupported SDXL backbone family: {family}")
    return "StableDiffusionXLPipeline"


def default_sdxl_loader_name(family: str) -> str:
    """Return the default loader label for an SDXL family label."""
    if family != "sdxl":
        raise RuntimeError(f"Unsupported SDXL backbone family: {family}")
    return "diffusers_sdxl"


def load_sdxl_pipeline(backbone_name: str, *, family: str, pipeline_class: Any, load_kwargs: dict[str, Any]) -> Any:
    """Load an SDXL-family diffusers pipeline."""
    if family != "sdxl":
        raise RuntimeError(f"Unsupported SDXL backbone family: {family}")
    return pipeline_class.from_pretrained(backbone_name, **load_kwargs)
