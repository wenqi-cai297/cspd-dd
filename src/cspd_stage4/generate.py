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

    # Try to load encode_index for medoid image paths
    # modes_dir is typically .../modes or .../modes_hdbscan, encode_dir is sibling .../encoded
    encode_dir = modes_dir.parent / "encoded"
    encode_samples = []
    encode_index_path = encode_dir / "encode_index.json"
    if encode_index_path.exists():
        with open(encode_index_path, encoding="utf-8") as f:
            encode_index = json.load(f)
        encode_samples = encode_index.get("samples", [])

    return {
        "visual_modes": visual_modes,
        "semantic_modes": semantic_modes,
        "pooled_modes": pooled_modes,
        "modes_list": modes_list,
        "ipc": modes_index.get("ipc", 0),
        "num_classes": modes_index.get("num_classes", 0),
        "total_modes": modes_index.get("total_modes", 0),
        "encode_samples": encode_samples,
    }


def _load_text2img_pipeline(
    model_name: str,
    lora_weights: str | None,
    device: str,
    dtype: str,
) -> Any:
    """Load SDXL text2img pipeline — identical to scripts/inference/sample_sdxl_lora.py."""
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
    pipe.set_progress_bar_config(disable=False)
    return pipe


def _load_img2img_pipeline(
    model_name: str,
    lora_weights: str | None,
    device: str,
    dtype: str,
) -> Any:
    """Load SDXL img2img pipeline for visual-anchor modes."""
    from diffusers import StableDiffusionXLImg2ImgPipeline

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
    visual_mode: str = "centroid",
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
        visual_mode: "centroid" uses the cluster centroid latent decoded to image.
            "medoid" uses the real image closest to centroid (sharper init image).
            "none" skips visual anchor entirely — pure text-to-image (recommended).
            When "none", the generation uses StableDiffusionXLPipeline (text2img) instead of img2img.
            Stage 3 visual clustering is still used to SELECT which captions to generate,
            but the generation itself is driven purely by text conditioning.

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

    encode_samples = modes["encode_samples"]

    print(f"[Stage 4] Loaded {total_modes} modes ({modes['num_classes']} classes × IPC {modes['ipc']})")
    print(f"[Stage 4] Visual mode: {visual_mode}, Semantic mode: {semantic_mode}")

    # Load SDXL pipeline
    if lora_weights:
        print(f"[Stage 4] LoRA weights: {lora_weights}")
    if visual_mode == "none":
        print(f"[Stage 4] Loading SDXL text2img pipeline (same as inference script)...")
        pipe = _load_text2img_pipeline(model_name, lora_weights, device, dtype)
    else:
        print(f"[Stage 4] Loading SDXL img2img pipeline...")
        pipe = _load_img2img_pipeline(model_name, lora_weights, device, dtype)

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

        # --- Text-to-image path (visual_mode="none") ---
        # Mirrors scripts/inference/sample_sdxl_lora.py exactly:
        #   generator = torch.Generator(device=device).manual_seed(seed + idx)
        #   image = pipe(prompt=prompt, height=res, width=res,
        #                num_inference_steps=steps, guidance_scale=gs,
        #                generator=generator).images[0]
        if visual_mode == "none":
            prompt = representative_caption if representative_caption else class_name
            generator = torch.Generator(device=device).manual_seed(seed + mode_idx)
            image = pipe(
                prompt=prompt,
                height=resolution,
                width=resolution,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images[0]

            # Save image
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
                "generation_mode": "text2img",
            })
            continue

        # --- Img2img path (visual_mode="centroid" or "medoid") ---
        use_centroid = visual_mode == "centroid"

        if visual_mode == "medoid":
            # Use the real image closest to the cluster centroid (sharp, no decode artifacts)
            medoid_index = mode_meta.get("visual_medoid_index", mode_idx)
            medoid_image_path = ""
            if encode_samples and medoid_index < len(encode_samples):
                medoid_image_path = encode_samples[medoid_index].get("image_path", "")
            if not medoid_image_path or not Path(medoid_image_path).exists():
                # Fallback: try to find image path from record_id
                medoid_record_id = mode_meta.get("visual_medoid_record_id", "")
                for s in encode_samples:
                    if s.get("record_id") == medoid_record_id:
                        medoid_image_path = s.get("image_path", "")
                        break
            if medoid_image_path and Path(medoid_image_path).exists():
                init_image = Image.open(medoid_image_path).convert("RGB")
                init_image = init_image.resize((resolution, resolution), Image.LANCZOS)
            else:
                print(f"  [WARN] Medoid image not found for mode {mode_idx}, falling back to centroid")
                use_centroid = True

        if use_centroid:
            # Decode cluster centroid latent to PIL image (may be blurry due to averaging)
            visual_latent = visual_modes[mode_idx].unsqueeze(0).to(device, dtype=torch.float32)
            latent_for_decode = visual_latent / vae_scaling_factor

            vae_orig_dtype = vae.dtype
            vae.to(dtype=torch.float32)
            decoded = vae.decode(latent_for_decode, return_dict=False)[0]
            vae.to(dtype=vae_orig_dtype)

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
