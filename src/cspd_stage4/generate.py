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
) -> tuple[Any, Any]:
    """Load SDXL img2img pipeline + VAE with optional Stage 2 LoRA weights.

    Returns (img2img_pipe, vae) where VAE is in float32 for stable decoding.
    """
    from diffusers import StableDiffusionXLImg2ImgPipeline, AutoencoderKL

    torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16

    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
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
    semantic_mode: str = "caption",
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
        semantic_mode: "caption" uses the representative caption text as prompt (recommended).
            "embedding" uses the mean text embedding from Stage 3 (baseline, may produce blurry results).

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

    # Load SDXL img2img pipeline
    print(f"[Stage 4] Loading SDXL img2img pipeline...")
    if lora_weights:
        print(f"[Stage 4] LoRA weights: {lora_weights}")
    pipe = _load_pipeline(model_name, lora_weights, device, dtype)

    # We need VAE in float32 for stable decoding of visual mode centroids
    vae = pipe.vae
    vae_scaling_factor = vae.config.scaling_factor

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

        # Get visual mode latent and decode to PIL image for img2img input
        visual_latent = visual_modes[mode_idx].unsqueeze(0).to(device, dtype=torch.float32)  # (1, 4, H, W)
        latent_for_decode = visual_latent / vae_scaling_factor

        # Temporarily cast VAE to float32 for decoding
        vae_orig_dtype = vae.dtype
        vae.to(dtype=torch.float32)
        decoded = vae.decode(latent_for_decode, return_dict=False)[0]
        vae.to(dtype=vae_orig_dtype)

        # Convert decoded tensor to PIL Image
        decoded = (decoded / 2 + 0.5).clamp(0, 1)
        decoded_np = decoded.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
        decoded_np = (decoded_np * 255).round().astype(np.uint8)
        init_image = Image.fromarray(decoded_np).resize((resolution, resolution), Image.LANCZOS)

        # Generate via img2img pipeline
        generator = torch.Generator(device="cpu").manual_seed(seed + mode_idx)

        if semantic_mode == "embedding":
            # Baseline: use mean text embedding from Stage 3 clustering.
            # May produce blurry results because the averaged embedding
            # doesn't correspond to any real caption the model has seen.
            prompt_embeds = semantic_modes[mode_idx].unsqueeze(0).to(device, dtype=torch_dtype)
            pooled_prompt_embeds = pooled_modes[mode_idx].unsqueeze(0).to(device, dtype=torch_dtype)
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
            output = pipe(
                image=init_image,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                output_type="pil",
            )
        else:
            # Recommended: use the representative caption (medoid caption) as
            # text prompt. This produces a real, coherent text embedding that
            # the Stage 2 LoRA model has seen during training.
            prompt = representative_caption if representative_caption else class_name
            output = pipe(
                image=init_image,
                prompt=prompt,
                negative_prompt="",
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                output_type="pil",
            )

        image = output.images[0]

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
