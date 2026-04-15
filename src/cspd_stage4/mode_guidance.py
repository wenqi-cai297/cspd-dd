"""Mode guidance scheduler for SDXL — steers generation toward VAE latent centroids.

Subclasses EulerDiscreteScheduler to inject mode guidance inside step(),
exactly matching MGD³ (ICML 2025) but adapted for Euler ODE sampling.

MGD³ original (DDPM):
    guidance = -(pred_x0 - centroid) * scale * sqrt(beta_t)
    prev_sample = predicted_mean + guidance + noise

Ours (Euler):
    guidance = -(pred_x0 - centroid) * scale * dt_ratio
    prev_sample = euler_step_result + guidance

dt_ratio normalizes guidance by the step size to keep it scale-invariant
across different numbers of sampling steps.
"""

from __future__ import annotations

import torch
from diffusers import EulerDiscreteScheduler
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteSchedulerOutput


class EulerModeGuidanceScheduler(EulerDiscreteScheduler):
    """EulerDiscreteScheduler with mode guidance injection in step().

    Usage:
        scheduler = EulerModeGuidanceScheduler.from_config(pipe.scheduler.config)
        scheduler.set_mode_guidance(centroid, scale=0.1, stop_step=25)
        pipe.scheduler = scheduler
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mode_centroid: torch.Tensor | None = None
        self._mode_guidance_scale: float = 0.0
        self._mode_guidance_stop_step: int = 25

    def set_mode_guidance(
        self,
        centroid: torch.Tensor | None,
        scale: float = 0.1,
        stop_step: int = 25,
    ) -> None:
        """Set the target mode centroid and guidance parameters.

        Args:
            centroid: VAE latent centroid (4, H, W) or (1, 4, H, W). None to disable.
            scale: Guidance strength (0.1 = MGD³ default).
            stop_step: Apply guidance for the first N steps only (25 = 50% of 50 steps).
        """
        self._mode_centroid = centroid
        self._mode_guidance_scale = scale
        self._mode_guidance_stop_step = stop_step

    def step(
        self,
        model_output: torch.Tensor,
        timestep: float | torch.Tensor,
        sample: torch.Tensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: torch.Generator | None = None,
        return_dict: bool = True,
    ) -> EulerDiscreteSchedulerOutput | tuple:
        """Euler step with optional mode guidance injection."""

        # Run the standard Euler step
        output = super().step(
            model_output=model_output,
            timestep=timestep,
            sample=sample,
            s_churn=s_churn,
            s_tmin=s_tmin,
            s_tmax=s_tmax,
            s_noise=s_noise,
            generator=generator,
            return_dict=True,
        )

        prev_sample = output.prev_sample
        pred_original_sample = output.pred_original_sample

        # Apply mode guidance if active
        # step_index was already incremented by super().step(), so current step = step_index - 1
        current_step = self._step_index - 1 if self._step_index is not None else 0

        if (
            self._mode_centroid is not None
            and self._mode_guidance_scale > 0
            and pred_original_sample is not None
            and current_step < self._mode_guidance_stop_step
        ):
            centroid = self._mode_centroid
            if centroid.dim() == 3:
                centroid = centroid.unsqueeze(0)

            # Match spatial shape if needed
            if centroid.shape[-2:] != pred_original_sample.shape[-2:]:
                centroid = torch.nn.functional.interpolate(
                    centroid.float(),
                    size=pred_original_sample.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            centroid = centroid.to(device=prev_sample.device, dtype=prev_sample.dtype)
            pred_x0 = pred_original_sample.to(prev_sample.dtype)

            # Guidance with linear decay: strong at step 0, zero at stop_step
            decay = 1.0 - current_step / self._mode_guidance_stop_step
            guidance = -(pred_x0 - centroid) * self._mode_guidance_scale * decay

            prev_sample = prev_sample + guidance

        if not return_dict:
            return (prev_sample, pred_original_sample)

        return EulerDiscreteSchedulerOutput(
            prev_sample=prev_sample,
            pred_original_sample=pred_original_sample,
        )
