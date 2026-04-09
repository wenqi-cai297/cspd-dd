from __future__ import annotations

"""PixArt-family Stage 2 backbone loading helpers."""

from typing import Any


def resolve_pixart_conditioning_transformer_patterns() -> list[str]:
    """Return the PixArt-specific narrowed conditioning-target patterns."""
    return [
        "caption_projection",
        "caption_projection.*",
        "adaln_single",
        "adaln_single.*",
        "transformer_blocks.*.attn1.*",
        "transformer_blocks.*.attn2.*",
        "transformer_blocks.*.ff.*",
    ]


def infer_pixart_family(backbone_name: str) -> str:
    """Infer the PixArt family label from a backbone identifier."""
    lowered = backbone_name.lower()
    if "pixart" in lowered and ("sigma" in lowered or "pixart-sigma" in lowered):
        return "pixart_sigma"
    return "pixart"


def resolve_pixart_pipeline_class_name(*, family: str) -> str:
    """Return the diffusers pipeline class name for a PixArt family label."""
    if family == "pixart_sigma":
        return "PixArtSigmaPipeline"
    if family == "pixart":
        return "PixArtAlphaPipeline"
    raise RuntimeError(f"Unsupported PixArt backbone family: {family}")


def default_pixart_loader_name(*, family: str) -> str:
    """Return the default shared-loader label for a PixArt family label."""
    if family == "pixart_sigma":
        return "diffusers_pixart_sigma"
    if family == "pixart":
        return "diffusers_pixart"
    raise RuntimeError(f"Unsupported PixArt backbone family: {family}")


def load_pixart_pipeline(
    backbone_name: str,
    *,
    family: str,
    pipeline_class: Any,
    load_kwargs: dict[str, Any],
    resolved_dtype: Any,
    diffusers_module: Any,
    local_files_only: bool,
) -> Any:
    """Load a PixArt-family diffusers pipeline, including Sigma fallback assembly."""
    if family != "pixart_sigma":
        return pipeline_class.from_pretrained(backbone_name, **load_kwargs)

    transformer_class = getattr(diffusers_module, "Transformer2DModel")
    base_repo = "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers"
    try:
        return pipeline_class.from_pretrained(backbone_name, **load_kwargs)
    except Exception:
        transformer = transformer_class.from_pretrained(
            backbone_name,
            subfolder="transformer",
            torch_dtype=resolved_dtype,
            local_files_only=local_files_only,
        )
        return pipeline_class.from_pretrained(base_repo, transformer=transformer, **load_kwargs)
