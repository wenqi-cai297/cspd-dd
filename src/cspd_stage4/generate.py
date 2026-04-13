"""Stage 4 — Dual-anchor conditioned distilled dataset generation.

Uses visual modes (latent centroids) and semantic modes (text embedding means)
from Stage 3 as dual anchors, combined with the Stage 2 LoRA-finetuned SDXL
backbone, to generate the final distilled dataset.

Generation flow per mode:
  1. Load visual mode latent as the initial denoising starting point
  2. Add noise at a controlled strength level
  3. Use semantic mode embedding as text conditioning
  4. Run SDXL UNet (with Stage 2 LoRA) to denoise
  5. VAE decode → distilled image
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from cspd_stage1.io_utils import write_json


@dataclass(slots=True)
class GenerateResult:
    """Result of Stage 4 distilled dataset generation."""

    output_dir: str
    num_images: int
    num_classes: int
    ipc: int
    images_dir: str
    metadata_path: str
    summary_path: str


def _load_modes(modes_dir: str | Path) -> dict[str, Any]:
    """Load Stage 3 mode tensors and metadata."""
    modes_dir = Path(modes_dir)

    visual_modes = torch.load(modes_dir / "visual_modes.pt", weights_only=True)
    semantic_modes = torch.load(modes_dir / "semantic_modes.pt", weights_only=True)
    pooled_modes = torch.load(modes_dir / "pooled_modes.pt", weights_only=True)

    with open(modes_dir / "modes_index.json", encoding="utf-8") as f:
        modes_index = json.load(f)

    modes_list = modes_index.get("modes", [])

    return {
        "visual_modes": visual_modes,
        "semantic_modes": semantic_modes,
        "pooled_modes": pooled_modes,
        "modes_list": modes_list,
        "ipc": modes_index.get("ipc", 0),
        "num_classes": modes_index.get("num_classes", 0),
        "total_modes": modes_index.get("total_modes", 0),
    }


def _load_pipeline(
    model_name: str,
    lora_weights: str | None,
    device: str,
    dtype: str,
) -> Any:
    """Load SDXL pipeline with optional Stage 2 LoRA weights."""
    from diffusers import StableDiffusionXLPipeline

    torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16

    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        use_safetensors=True,
    )

    if lora_weights:
        lora_path = Path(lora_weights)
        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA weights not found: {lora_path}")
        pipe.load_lora_weights(str(lora_path.parent), weight_name=lora_path.name)

    pipe = pipe.to(device)
    return pipe


@torch.no_grad()
def generate_distilled_dataset(
    *,
    modes_dir: str | Path,
    output_dir: str | Path,
    lora_weights: str | None = None,
    model_name: str = "stabilityai/stable-diffusion-xl-base-1.0",
    strength: float = 0.5,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    seed: int = 42,
    device: str = "cuda",
    dtype: str = "float16",
    resolution: int = 512,
) -> GenerateResult:
    """Generate the distilled dataset using dual-anchor conditioning.

    Args:
        modes_dir: Directory with Stage 3 mode outputs (visual_modes.pt, semantic_modes.pt, etc.).
        output_dir: Directory for distilled dataset output.
        lora_weights: Path to Stage 2 LoRA weights (.safetensors). None for baseline.
        model_name: SDXL model identifier.
        strength: Noise strength for img2img. 0=pure visual mode, 1=pure text-to-image.
        num_inference_steps: Diffusion sampling steps.
        guidance_scale: Classifier-free guidance scale.
        seed: RNG seed.
        device: Torch device.
        dtype: Weight dtype.
        resolution: Output image resolution.

    Returns:
        GenerateResult with paths to generated images and metadata.
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16

    # Load Stage 3 modes
    print("[Stage 4] Loading Stage 3 modes...")
    modes = _load_modes(modes_dir)
    visual_modes = modes["visual_modes"]       # (M, 4, H, W)
    semantic_modes = modes["semantic_modes"]     # (M, seq_len, dim)
    pooled_modes = modes["pooled_modes"]         # (M, pooled_dim)
    modes_list = modes["modes_list"]
    total_modes = modes["total_modes"]

    print(f"[Stage 4] Loaded {total_modes} modes ({modes['num_classes']} classes × IPC {modes['ipc']})")

    # Load SDXL pipeline
    print(f"[Stage 4] Loading SDXL pipeline...")
    if lora_weights:
        print(f"[Stage 4] LoRA weights: {lora_weights}")
    pipe = _load_pipeline(model_name, lora_weights, device, dtype)

    # Get VAE and scheduler from pipeline
    vae = pipe.vae
    scheduler = pipe.scheduler
    unet = pipe.unet
    vae_scaling_factor = vae.config.scaling_factor

    # Prepare time_ids for SDXL conditioning (original_size + crop_coords + target_size)
    add_time_ids = torch.tensor(
        [[resolution, resolution, 0, 0, resolution, resolution]],
        dtype=torch_dtype,
        device=device,
    )

    # Generate one image per mode
    metadata_rows = []
    print(f"[Stage 4] Generating {total_modes} distilled images (strength={strength}, steps={num_inference_steps})...")

    for mode_idx in tqdm(range(total_modes), desc="Generating"):
        mode_meta = modes_list[mode_idx] if mode_idx < len(modes_list) else {}
        class_name = mode_meta.get("class_name", "unknown")
        class_name_raw = mode_meta.get("class_name_raw", "unknown")
        archetype = mode_meta.get("archetype", "unknown")
        cluster_id = mode_meta.get("cluster_id", mode_idx)
        representative_caption = mode_meta.get("representative_caption", "")

        # Get mode tensors
        visual_latent = visual_modes[mode_idx].unsqueeze(0).to(device, dtype=torch_dtype)  # (1, 4, H, W)
        prompt_embeds = semantic_modes[mode_idx].unsqueeze(0).to(device, dtype=torch_dtype)  # (1, seq_len, dim)
        pooled_prompt_embeds = pooled_modes[mode_idx].unsqueeze(0).to(device, dtype=torch_dtype)  # (1, pooled_dim)

        # Prepare negative prompt embeddings (unconditional = zeros)
        negative_prompt_embeds = torch.zeros_like(prompt_embeds)
        negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)

        # Set up scheduler
        scheduler.set_timesteps(num_inference_steps, device=device)

        # Determine the starting timestep based on strength
        # strength=0.5 means start denoising from halfway
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = scheduler.timesteps[t_start:]

        # Add noise to the visual mode latent
        # Generate noise on CPU (supports generator) then move to device
        cpu_generator = torch.Generator(device="cpu").manual_seed(seed + mode_idx)
        noise = torch.randn(visual_latent.shape, generator=cpu_generator, dtype=visual_latent.dtype).to(device)

        if len(timesteps) > 0:
            # Scale the latent (already scaled by vae_scaling_factor from Stage 3)
            latents = scheduler.add_noise(visual_latent, noise, timesteps[:1])
        else:
            # strength=0: no noise, just decode the centroid directly
            latents = visual_latent

        # Concatenate conditional and unconditional for CFG
        prompt_embeds_cfg = torch.cat([negative_prompt_embeds, prompt_embeds])
        pooled_embeds_cfg = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds])
        add_time_ids_cfg = torch.cat([add_time_ids, add_time_ids])
        added_cond_kwargs = {
            "text_embeds": pooled_embeds_cfg,
            "time_ids": add_time_ids_cfg,
        }

        # Denoising loop
        for t in timesteps:
            latent_model_input = torch.cat([latents] * 2)  # for CFG
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds_cfg,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            # CFG: noise_pred = uncond + guidance_scale * (cond - uncond)
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # Step
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # VAE decode
        latents_decoded = latents / vae_scaling_factor
        image_tensor = vae.decode(latents_decoded, return_dict=False)[0]

        # Tensor to PIL
        image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
        image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
        image_np = (image_np * 255).round().astype(np.uint8)
        image = Image.fromarray(image_np)

        # Save image: organize by class
        class_dir = images_dir / class_name_raw
        class_dir.mkdir(parents=True, exist_ok=True)
        image_filename = f"{class_name_raw}_mode{cluster_id:03d}.png"
        image_path = class_dir / image_filename
        image.save(image_path)

        metadata_rows.append({
            "mode_index": mode_idx,
            "class_name": class_name,
            "class_name_raw": class_name_raw,
            "archetype": archetype,
            "cluster_id": cluster_id,
            "representative_caption": representative_caption,
            "num_cluster_members": mode_meta.get("num_members", 0),
            "image_path": str(image_path),
            "relative_image_path": f"{class_name_raw}/{image_filename}",
        })

    # Save metadata
    metadata_path = output_dir / "distilled_metadata.json"
    write_json(metadata_path, {
        "num_images": len(metadata_rows),
        "num_classes": modes["num_classes"],
        "ipc": modes["ipc"],
        "model_name": model_name,
        "lora_weights": str(lora_weights) if lora_weights else None,
        "strength": strength,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "resolution": resolution,
        "modes_dir": str(Path(modes_dir).resolve()),
        "images": metadata_rows,
    })

    # Save summary
    class_counts = {}
    for row in metadata_rows:
        cn = row["class_name_raw"]
        class_counts[cn] = class_counts.get(cn, 0) + 1

    summary = {
        "num_images": len(metadata_rows),
        "num_classes": len(class_counts),
        "ipc": modes["ipc"],
        "strength": strength,
        "lora_loaded": lora_weights is not None,
        "class_counts": class_counts,
    }
    summary_path = output_dir / "stage4_summary.json"
    write_json(summary_path, summary)

    print(f"[Stage 4] Generated {len(metadata_rows)} distilled images")
    print(f"[Stage 4] Output: {output_dir}")

    return GenerateResult(
        output_dir=str(output_dir),
        num_images=len(metadata_rows),
        num_classes=len(class_counts),
        ipc=modes["ipc"],
        images_dir=str(images_dir),
        metadata_path=str(metadata_path),
        summary_path=str(summary_path),
    )
