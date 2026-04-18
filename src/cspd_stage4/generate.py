"""Stage 4 — Distilled dataset generation.

Two generation paths are kept:

- **text2img** (`visual_mode="none"`, default): SDXL + Stage 2 LoRA generates
  one image per Stage 3 mode from the mode's representative caption.
- **img2img** (`visual_mode="medoid"`): same but starts from the real medoid
  image with noise strength `strength`. Worse eval accuracy than text2img but
  kept available for ablation.

An optional SDXL refiner pass (`--refiner-model`) can be appended to either
path.

Removed 2026-04-18: multi-candidate selection (Phase 2), set-level
representativeness selection (Phase 3), MGD³-style mode guidance. All three
were tested and regressed vs the single-medoid baseline.
"""

from __future__ import annotations

import json
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
    """Load Stage 3 mode metadata (modes_index.json)."""
    modes_dir = Path(modes_dir)

    with open(modes_dir / "modes_index.json", encoding="utf-8") as f:
        modes_index = json.load(f)

    modes_list = modes_index.get("modes", [])

    # Load encode_index for medoid image paths (used by the img2img path)
    encode_dir = modes_dir.parent / "encoded"
    encode_samples = []
    encode_index_path = encode_dir / "encode_index.json"
    if encode_index_path.exists():
        with open(encode_index_path, encoding="utf-8") as f:
            encode_index = json.load(f)
        encode_samples = encode_index.get("samples", [])

    return {
        "modes_list": modes_list,
        "ipc": modes_index.get("ipc", 0),
        "num_classes": modes_index.get("num_classes", 0),
        "total_modes": modes_index.get("total_modes", 0),
        "encode_samples": encode_samples,
    }


def _is_sd15_model(model_name: str) -> bool:
    """Heuristic: SD v1.5-style model identifier vs SDXL."""
    lowered = model_name.lower()
    return "stable-diffusion" in lowered and "xl" not in lowered


def _load_text2img_pipeline(model_name: str, lora_weights: str | None, device: str, dtype: str) -> Any:
    torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16

    # If `lora_weights` points to a directory, treat as a full fine-tuned checkpoint.
    if lora_weights and Path(lora_weights).is_dir():
        print(f"[Stage 4] Loading full fine-tuned model from {lora_weights}")
        if _is_sd15_model(model_name):
            from diffusers import StableDiffusionPipeline
            pipe = StableDiffusionPipeline.from_pretrained(
                lora_weights, torch_dtype=torch_dtype, safety_checker=None,
            )
        else:
            from diffusers import StableDiffusionXLPipeline
            pipe = StableDiffusionXLPipeline.from_pretrained(
                lora_weights, torch_dtype=torch_dtype, use_safetensors=True,
            )
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=False)
        return pipe

    if _is_sd15_model(model_name):
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            model_name, torch_dtype=torch_dtype, safety_checker=None,
        )
    else:
        from diffusers import StableDiffusionXLPipeline
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_name, torch_dtype=torch_dtype, use_safetensors=True,
        )

    if lora_weights:
        lora_path = Path(lora_weights)
        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA weights not found: {lora_path}")
        pipe.load_lora_weights(str(lora_path.parent), weight_name=lora_path.name)

    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=False)
    return pipe


def _load_img2img_pipeline(model_name: str, lora_weights: str | None, device: str, dtype: str) -> Any:
    torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16

    if _is_sd15_model(model_name):
        from diffusers import StableDiffusionImg2ImgPipeline
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_name, torch_dtype=torch_dtype, safety_checker=None,
        )
    else:
        from diffusers import StableDiffusionXLImg2ImgPipeline
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            model_name, torch_dtype=torch_dtype, use_safetensors=True,
        )

    if lora_weights:
        lora_path = Path(lora_weights)
        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA weights not found: {lora_path}")
        pipe.load_lora_weights(str(lora_path.parent), weight_name=lora_path.name)

    pipe = pipe.to(device)
    return pipe


def _load_refiner_pipeline(refiner_model: str, device: str, dtype: str) -> Any:
    from diffusers import StableDiffusionXLImg2ImgPipeline

    torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
    refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        refiner_model, torch_dtype=torch_dtype, use_safetensors=True,
    )
    refiner = refiner.to(device)
    refiner.set_progress_bar_config(disable=False)
    return refiner


@torch.no_grad()
def generate_distilled_dataset(
    *,
    modes_dir: str | Path,
    output_dir: str | Path,
    lora_weights: str | None = None,
    model_name: str = "stabilityai/stable-diffusion-xl-base-1.0",
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
) -> GenerateResult:
    """Generate one distilled image per Stage 3 mode via SDXL LoRA.

    Args:
        modes_dir: Stage 3 output directory with modes_index.json.
        output_dir: Where the distilled dataset is written.
        lora_weights: Stage 2 LoRA weights (.safetensors), or a full fine-tuned
            checkpoint directory, or None for baseline SDXL.
        model_name: Base SDXL model identifier.
        strength: Img2img denoising strength (ignored when visual_mode="none").
        num_inference_steps: Diffusion sampling steps.
        guidance_scale: Classifier-free guidance scale.
        seed: RNG seed (per-image seed is `seed + mode_idx`).
        device / dtype / resolution: standard knobs.
        visual_mode: "none" for text2img (default, baseline) or "medoid" for
            img2img starting from the real medoid image.
        refiner_model: Optional SDXL refiner model id. When set, runs a second
            refiner pass at `refiner_strength` for added detail / sharpness.
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print("[Stage 4] Loading Stage 3 modes...")
    modes = _load_modes(modes_dir)
    modes_list = modes["modes_list"]
    total_modes = modes["total_modes"]
    encode_samples = modes["encode_samples"]

    print(f"[Stage 4] Loaded {total_modes} modes ({modes['num_classes']} classes x IPC {modes['ipc']})")
    print(f"[Stage 4] Visual mode: {visual_mode}")

    if lora_weights:
        print(f"[Stage 4] LoRA weights: {lora_weights}")

    if visual_mode == "none":
        print("[Stage 4] Loading SDXL text2img pipeline...")
        pipe = _load_text2img_pipeline(model_name, lora_weights, device, dtype)
    elif visual_mode == "medoid":
        print("[Stage 4] Loading SDXL img2img pipeline...")
        pipe = _load_img2img_pipeline(model_name, lora_weights, device, dtype)
    else:
        raise ValueError(f"visual_mode must be 'none' or 'medoid', got {visual_mode!r}")

    refiner = None
    if refiner_model:
        print(f"[Stage 4] Loading SDXL refiner: {refiner_model}")
        refiner = _load_refiner_pipeline(refiner_model, device, dtype)

    metadata_rows: list[dict[str, Any]] = []
    print(f"[Stage 4] Generating {total_modes} distilled images "
          f"(steps={num_inference_steps}, guidance={guidance_scale}"
          f"{', strength=' + str(strength) if visual_mode == 'medoid' else ''})...")

    for mode_idx in tqdm(range(total_modes), desc="Generating"):
        mode_meta = modes_list[mode_idx] if mode_idx < len(modes_list) else {}
        class_name = mode_meta.get("class_name", "unknown")
        class_name_raw = mode_meta.get("class_name_raw", "unknown")
        archetype = mode_meta.get("archetype", "unknown")
        cluster_id = mode_meta.get("cluster_id", mode_idx)
        representative_caption = mode_meta.get("representative_caption", "")
        prompt = representative_caption if representative_caption else class_name
        generator = torch.Generator(device=device).manual_seed(seed + mode_idx)

        if visual_mode == "none":
            image = pipe(
                prompt=prompt,
                height=resolution,
                width=resolution,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images[0]
            generation_mode = "text2img"
        else:  # medoid img2img
            medoid_record_id = mode_meta.get("medoid_record_id", mode_meta.get("visual_medoid_record_id", ""))
            medoid_image_path = ""
            for sample in encode_samples:
                if sample.get("record_id") == medoid_record_id:
                    medoid_image_path = sample.get("image_path", "")
                    break
            if not medoid_image_path or not Path(medoid_image_path).exists():
                print(f"  [WARN] Medoid image not found for mode {mode_idx}, skipping")
                continue
            init_image = Image.open(medoid_image_path).convert("RGB")
            init_image = init_image.resize((resolution, resolution), Image.LANCZOS)
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
            generation_mode = "img2img+medoid"

        if refiner is not None:
            refiner_gen = torch.Generator(device=device).manual_seed(seed + mode_idx)
            image = refiner(
                prompt=prompt,
                image=image,
                strength=refiner_strength,
                generator=refiner_gen,
            ).images[0]
            generation_mode += "+refiner"

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
            "generation_mode": generation_mode,
        })

    metadata_path = output_dir / "distilled_metadata.json"
    write_json(metadata_path, {
        "num_images": len(metadata_rows),
        "num_classes": modes["num_classes"],
        "ipc": modes["ipc"],
        "model_name": model_name,
        "lora_weights": str(lora_weights) if lora_weights else None,
        "visual_mode": visual_mode,
        "strength": strength if visual_mode == "medoid" else None,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "resolution": resolution,
        "refiner_model": refiner_model,
        "refiner_strength": refiner_strength if refiner_model else None,
        "modes_dir": str(Path(modes_dir).resolve()),
        "images": metadata_rows,
    })

    class_counts: dict[str, int] = {}
    for row in metadata_rows:
        cn = row["class_name_raw"]
        class_counts[cn] = class_counts.get(cn, 0) + 1

    summary_path = output_dir / "stage4_summary.json"
    write_json(summary_path, {
        "num_images": len(metadata_rows),
        "num_classes": len(class_counts),
        "ipc": modes["ipc"],
        "visual_mode": visual_mode,
        "strength": strength if visual_mode == "medoid" else None,
        "lora_loaded": lora_weights is not None,
        "class_counts": class_counts,
    })

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
