from __future__ import annotations

"""Stage 2 backbone-family detection (SDXL-only as of 2026-04-18).

The module inspection / adapter-injection / real-loader helpers that lived
here were all for the self-built FLUX/PixArt training loop, which has been
removed. SDXL training is delegated to the official diffusers trainer and
does its own loading internally.
"""

from cspd_stage2.families.sdxl.backbone import infer_sdxl_family


def infer_backbone_family(backbone_name: str) -> str:
    lowered = backbone_name.lower()
    if "sdxl" in lowered or "stable-diffusion-xl" in lowered:
        return infer_sdxl_family(backbone_name)
    return "generic_diffusion_backbone"
