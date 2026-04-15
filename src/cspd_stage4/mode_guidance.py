"""Mode guidance for SDXL denoising — steers generation toward VAE latent centroids.

Adapts MGD³ (ICML 2025) mode guidance to the SDXL pipeline. During each denoising
step, a guidance term pushes the predicted x0 toward the target mode centroid in
VAE latent space, ensuring each generated image belongs to a distinct visual mode.

Combined with our structured caption conditioning (from Stage 1), this provides
dual control: text controls WHAT to generate (semantic attributes), mode guidance
controls HOW it should look (visual layout/composition).

The guidance formula per step:
    guidance = -(pred_x0 - mode_centroid) * mode_guidance_scale * sigma_t
    latent = latent + guidance

Where sigma_t decays with the noise schedule, so guidance is strong in early
steps (coarse structure) and weak in late steps (fine details).
"""

from __future__ import annotations

from typing import Any

import torch


def apply_mode_guidance(
    *,
    latents: torch.Tensor,
    pred_original_sample: torch.Tensor,
    mode_centroid: torch.Tensor,
    sigma: float,
    mode_guidance_scale: float,
    timestep: int,
    stop_timestep: int,
) -> torch.Tensor:
    """Apply mode guidance to latents during one denoising step.

    Args:
        latents: Current noisy latents (B, C, H, W).
        pred_original_sample: Model's prediction of clean x0 (B, C, H, W).
        mode_centroid: Target VAE latent centroid (1, C, H, W) or (C, H, W).
        sigma: Current noise level (sigma_t from scheduler).
        mode_guidance_scale: Guidance strength (default 0.1 in MGD³).
        timestep: Current timestep.
        stop_timestep: Stop guidance below this timestep to avoid over-conditioning.

    Returns:
        Adjusted latents with mode guidance applied.
    """
    if timestep <= stop_timestep:
        return latents

    # Ensure centroid has batch dimension
    if mode_centroid.dim() == 3:
        mode_centroid = mode_centroid.unsqueeze(0)

    # Match shapes: centroid might be (1, 4, 64, 64) but latents could be different size
    # If resolution mismatch, interpolate centroid
    if mode_centroid.shape[-2:] != pred_original_sample.shape[-2:]:
        mode_centroid = torch.nn.functional.interpolate(
            mode_centroid.float(),
            size=pred_original_sample.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).to(pred_original_sample.dtype)

    mode_centroid = mode_centroid.to(device=latents.device, dtype=latents.dtype)

    # MGD³ formula: guidance = -(pred_x0 - centroid) * scale * sigma
    guidance = -(pred_original_sample - mode_centroid) * mode_guidance_scale * sigma

    return latents + guidance
