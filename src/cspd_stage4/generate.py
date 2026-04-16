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
    """Load Stage 3 mode metadata (modes_index.json only).

    No tensor files are needed — Stage 4 text2img uses representative captions
    as plain text strings from modes_index.json.
    """
    modes_dir = Path(modes_dir)

    with open(modes_dir / "modes_index.json", encoding="utf-8") as f:
        modes_index = json.load(f)

    modes_list = modes_index.get("modes", [])

    # Try to load encode_index for medoid image paths (used by img2img path)
    encode_dir = modes_dir.parent / "encoded"
    encode_samples = []
    encode_index_path = encode_dir / "encode_index.json"
    if encode_index_path.exists():
        with open(encode_index_path, encoding="utf-8") as f:
            encode_index = json.load(f)
        encode_samples = encode_index.get("samples", [])

    # Try to load mode centroids (for mode guidance)
    mode_centroids = None
    centroids_path = modes_dir / "mode_centroids.pt"
    if centroids_path.exists():
        mode_centroids = torch.load(centroids_path, weights_only=True)

    return {
        "modes_list": modes_list,
        "ipc": modes_index.get("ipc", 0),
        "num_classes": modes_index.get("num_classes", 0),
        "total_modes": modes_index.get("total_modes", 0),
        "encode_samples": encode_samples,
        "mode_centroids": mode_centroids,
    }


def _is_sd15_model(model_name: str) -> bool:
    """Check if model_name refers to SD v1.5 (not SDXL)."""
    lowered = model_name.lower()
    return "stable-diffusion" in lowered and "xl" not in lowered


def _load_text2img_pipeline(
    model_name: str,
    lora_weights: str | None,
    device: str,
    dtype: str,
) -> Any:
    """Load text2img pipeline — auto-detects SD v1.5 vs SDXL.

    For full fine-tuned models, pass the checkpoint dir as model_name (lora_weights=None).
    For LoRA fine-tuned models, pass base model as model_name and LoRA as lora_weights.
    """
    torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16

    # If lora_weights points to a directory (full fine-tuned checkpoint), load from there directly
    if lora_weights and Path(lora_weights).is_dir():
        print(f"[Stage 4] Loading full fine-tuned model from {lora_weights}")
        if _is_sd15_model(model_name):
            from diffusers import StableDiffusionPipeline
            pipe = StableDiffusionPipeline.from_pretrained(
                lora_weights,
                torch_dtype=torch_dtype,
                safety_checker=None,
            )
        else:
            from diffusers import StableDiffusionXLPipeline
            pipe = StableDiffusionXLPipeline.from_pretrained(
                lora_weights,
                torch_dtype=torch_dtype,
                use_safetensors=True,
            )
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=False)
        return pipe

    # Load base model
    if _is_sd15_model(model_name):
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            safety_checker=None,
        )
    else:
        from diffusers import StableDiffusionXLPipeline
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            use_safetensors=True,
        )

    # Load LoRA weights if provided (file path, not directory)
    if lora_weights:
        lora_path = Path(lora_weights)
        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA weights not found: {lora_path}")
        pipe.load_lora_weights(str(lora_path.parent), weight_name=lora_path.name)

    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=False)
    return pipe


def _load_refiner_pipeline(
    refiner_model: str,
    device: str,
    dtype: str,
) -> Any:
    """Load SDXL refiner pipeline for two-stage generation."""
    from diffusers import StableDiffusionXLImg2ImgPipeline

    torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16

    refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        refiner_model,
        torch_dtype=torch_dtype,
        use_safetensors=True,
    )
    refiner = refiner.to(device)
    refiner.set_progress_bar_config(disable=False)
    return refiner


def _load_img2img_pipeline(
    model_name: str,
    lora_weights: str | None,
    device: str,
    dtype: str,
) -> Any:
    """Load img2img pipeline — auto-detects SD v1.5 vs SDXL."""
    torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16

    if _is_sd15_model(model_name):
        from diffusers import StableDiffusionImg2ImgPipeline
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            safety_checker=None,
        )
    else:
        from diffusers import StableDiffusionXLImg2ImgPipeline
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
    model_name: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
    strength: float = 0.8,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    seed: int = 42,
    device: str = "cuda",
    dtype: str = "float16",
    resolution: int = 512,
    visual_mode: str = "none",
    refiner_model: str | None = None,
    refiner_strength: float = 0.3,
    mode_guidance_scale: float = 0.0,
    mode_guidance_stop_step: int = 25,
    num_candidates: int = 1,
    candidate_beta: float = 0.5,
    candidate_probe_dir: str | None = None,
) -> GenerateResult:
    """Generate the distilled dataset using dual-anchor conditioning.

    Args:
        modes_dir: Directory with Stage 3 mode outputs (modes_index.json).
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
        visual_mode: "none" for pure text-to-image (recommended).
            "medoid" for img2img from real medoid image.
            Stage 3 DINOv2 clustering selects which captions to generate (one per mode).
        refiner_model: Optional SDXL refiner model identifier (e.g. "stabilityai/stable-diffusion-xl-refiner-1.0").
            When provided, runs a two-stage pipeline: base generates at high_noise_frac, refiner
            refines the result. Adds detail and sharpness.
        refiner_strength: Denoising strength for refiner pass (0-1). Lower = less change, more detail.

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
    modes_list = modes["modes_list"]
    total_modes = modes["total_modes"]
    mode_centroids = modes.get("mode_centroids")

    encode_samples = modes["encode_samples"]

    use_mode_guidance = mode_guidance_scale > 0 and mode_centroids is not None
    if use_mode_guidance:
        print(f"[Stage 4] Mode guidance enabled: scale={mode_guidance_scale}, stop_step={mode_guidance_stop_step}")
        print(f"[Stage 4] Mode centroids shape: {list(mode_centroids.shape)}")
    elif mode_guidance_scale > 0:
        print("[Stage 4] WARNING: mode_guidance_scale > 0 but no mode_centroids.pt found. Running without guidance.")

    print(f"[Stage 4] Loaded {total_modes} modes ({modes['num_classes']} classes × IPC {modes['ipc']})")
    print(f"[Stage 4] Visual mode: {visual_mode}")

    # Load SDXL pipeline
    if lora_weights:
        print(f"[Stage 4] LoRA weights: {lora_weights}")
    if visual_mode == "none":
        print(f"[Stage 4] Loading SDXL text2img pipeline (same as inference script)...")
        pipe = _load_text2img_pipeline(model_name, lora_weights, device, dtype)
    else:
        print(f"[Stage 4] Loading SDXL img2img pipeline...")
        pipe = _load_img2img_pipeline(model_name, lora_weights, device, dtype)

    # Load optional refiner
    refiner = None
    if refiner_model:
        print(f"[Stage 4] Loading SDXL refiner: {refiner_model}")
        refiner = _load_refiner_pipeline(refiner_model, device, dtype)

    # Initialize candidate selector if multi-candidate mode
    selector = None
    if num_candidates > 1:
        from cspd_stage4.candidate_selection import CandidateSelector

        # Collect all class names from modes
        all_class_names_raw = sorted(set(m.get("class_name_raw", "") for m in modes_list))
        selector = CandidateSelector(
            class_names_raw=all_class_names_raw,
            device=device,
            beta=candidate_beta,
        )

        # Train linear probe on real data DINOv2 features if encode_dir has them
        modes_dir_path = Path(modes_dir)
        encode_dir = candidate_probe_dir or str(modes_dir_path.parent / "encoded")
        dino_path = Path(encode_dir) / "dino_embeds.pt"
        index_path = Path(encode_dir) / "encode_index.json"
        if dino_path.exists() and index_path.exists():
            import json as _json
            dino_features = torch.load(dino_path, weights_only=True)
            with open(index_path, encoding="utf-8") as _f:
                _encode_index = _json.load(_f)
            _samples = _encode_index.get("samples", [])
            _labels = torch.tensor([
                selector.class_to_id.get(s.get("class_name_raw", ""), 0)
                for s in _samples
            ])
            print(f"[Stage 4] Training linear probe on {len(_samples)} real DINOv2 features...")
            selector.train_probe(dino_features, _labels)
        else:
            print(f"[Stage 4] WARNING: No DINOv2 features at {dino_path}, running without discriminative scoring")

        print(f"[Stage 4] Multi-candidate mode: {num_candidates} candidates/mode, beta={candidate_beta}")

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

            if use_mode_guidance:
                # Swap scheduler to mode-guided version for this image,
                # then set the target centroid. The custom scheduler injects
                # guidance inside step() with access to pred_x0.
                from cspd_stage4.mode_guidance import EulerModeGuidanceScheduler

                centroid = mode_centroids[mode_idx]

                # Replace scheduler (only once, reuse across images)
                if not isinstance(pipe.scheduler, EulerModeGuidanceScheduler):
                    pipe.scheduler = EulerModeGuidanceScheduler.from_config(pipe.scheduler.config)

                pipe.scheduler.set_mode_guidance(
                    centroid=centroid,
                    scale=mode_guidance_scale,
                    stop_step=mode_guidance_stop_step,
                )

                image = pipe(
                    prompt=prompt,
                    height=resolution,
                    width=resolution,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                ).images[0]
            else:
                if selector is not None and num_candidates > 1:
                    # Multi-candidate: generate N, score, select best
                    best_image = None
                    best_score = -float("inf")
                    best_embedding = None
                    for cand_idx in range(num_candidates):
                        cand_seed = seed + mode_idx * num_candidates + cand_idx
                        cand_gen = torch.Generator(device=device).manual_seed(cand_seed)
                        cand_image = pipe(
                            prompt=prompt,
                            height=resolution,
                            width=resolution,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            generator=cand_gen,
                        ).images[0]
                        total, disc, div, emb = selector.score_candidate(cand_image, class_name_raw)
                        if total > best_score:
                            best_score = total
                            best_image = cand_image
                            best_embedding = emb
                    image = best_image
                    selector.accept_candidate(class_name_raw, best_embedding)
                else:
                    # Standard: single candidate
                    image = pipe(
                        prompt=prompt,
                        height=resolution,
                        width=resolution,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                    ).images[0]

            # Optional refiner pass
            if refiner is not None:
                refiner_gen = torch.Generator(device=device).manual_seed(seed + mode_idx)
                image = refiner(
                    prompt=prompt,
                    image=image,
                    strength=refiner_strength,
                    generator=refiner_gen,
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
                "generation_mode": "text2img" + ("+mode_guidance" if use_mode_guidance else ""),
            })
            continue

        # --- Img2img path (visual_mode="medoid") ---
        # Load medoid real image as img2img init
        medoid_record_id = mode_meta.get("medoid_record_id", mode_meta.get("visual_medoid_record_id", ""))
        medoid_image_path = ""
        for s in encode_samples:
            if s.get("record_id") == medoid_record_id:
                medoid_image_path = s.get("image_path", "")
                break
        if not medoid_image_path or not Path(medoid_image_path).exists():
            print(f"  [WARN] Medoid image not found for mode {mode_idx}, skipping")
            continue

        init_image = Image.open(medoid_image_path).convert("RGB")
        init_image = init_image.resize((resolution, resolution), Image.LANCZOS)

        prompt = representative_caption if representative_caption else class_name
        generator = torch.Generator(device=device).manual_seed(seed + mode_idx)
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

        # Optional refiner pass
        if refiner is not None:
            refiner_gen = torch.Generator(device=device).manual_seed(seed + mode_idx)
            image = refiner(
                prompt=prompt,
                image=image,
                strength=refiner_strength,
                generator=refiner_gen,
            ).images[0]

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
        "refiner_model": refiner_model,
        "refiner_strength": refiner_strength if refiner_model else None,
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
