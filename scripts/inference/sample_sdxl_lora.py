#!/usr/bin/env python3
"""Sample images from SDXL + Stage 2 LoRA weights using canonical captions.

Usage:
    python scripts/inference/sample_sdxl_lora.py \
        --lora-weights runs/stage2/train/.../official_output/pytorch_lora_weights.safetensors \
        --output-dir runs/stage2/samples/my_test

    # Compare with baseline (no LoRA):
    python scripts/inference/sample_sdxl_lora.py \
        --output-dir runs/stage2/samples/baseline --no-lora

    # Use custom prompts from file:
    python scripts/inference/sample_sdxl_lora.py \
        --lora-weights ... --prompt-file prompts.txt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL


# Representative canonical captions covering all ImageNette archetypes.
# Updated 2026-04-13 to match the current per-slot-guided render style.
DEFAULT_PROMPTS = [
    # animal - tench
    "a brown large and flat body tench being held in outdoor, natural setting with grass and trees front view",
    # animal - springer spaniel
    "a black and white long floppy ears english springer standing in grass side view with ears",
    # device_or_appliance - cassette player
    "a black and silver plastic rectangular with rounded edges cassette player idle with display off in white surface front view",
    # device_or_appliance - gas pump
    "an orange metal rectangular with rounded edges gas pump idle in park-like area side view",
    # tool - chain saw
    "a black and silver metal long handle with blade chain saw in use in grass field side view",
    # structure_or_building - church
    "a carpet and stone and wood gothic large church in interior front view with stained glass windows",
    # instrument - french horn
    "a golden yellow metal curved tubing french horn resting in white surface side view",
    # vehicle - garbage truck
    "an orange dump truck body and open rear garbage truck driving on road in urban street side view",
    # sports_or_toy - golf ball
    "a white with dimples rubber spherical golf ball resting in green grass close-up view",
    # sports_or_toy - parachute
    "a rainbow striped fabric elliptical with lines parachute in flight in sky with clouds low angle view",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample from SDXL with optional Stage 2 LoRA")
    parser.add_argument(
        "--lora-weights",
        default=None,
        help="Path to pytorch_lora_weights.safetensors from Stage 2 training",
    )
    parser.add_argument("--no-lora", action="store_true", help="Run baseline SDXL without LoRA for comparison")
    parser.add_argument(
        "--model-name",
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Base SDXL model identifier",
    )
    parser.add_argument("--output-dir", required=True, help="Directory to save generated images and metadata")
    parser.add_argument("--prompt", action="append", dest="prompts", default=None, help="Custom prompt; may be repeated")
    parser.add_argument("--prompt-file", default=None, help="Text file with one prompt per line")
    parser.add_argument("--resolution", type=int, default=512, help="Output image resolution")
    parser.add_argument("--num-inference-steps", type=int, default=50, help="Diffusion sampling steps")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="Classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    parser.add_argument("--device", default="cuda", help="Device to run on")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"], help="Model dtype")
    return parser.parse_args()


def load_prompts(args: argparse.Namespace) -> list[str]:
    """Resolve prompt list from args, file, or defaults."""
    if args.prompts:
        return args.prompts
    if args.prompt_file:
        lines = Path(args.prompt_file).read_text(encoding="utf-8").strip().splitlines()
        return [line.strip() for line in lines if line.strip()]
    return list(DEFAULT_PROMPTS)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(args)
    torch_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    print(f"[INFO] Loading base model: {args.model_name}")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        use_safetensors=True,
    )

    lora_loaded = False
    if args.lora_weights and not args.no_lora:
        lora_path = Path(args.lora_weights)
        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA weights not found: {lora_path}")
        print(f"[INFO] Loading LoRA weights: {lora_path}")
        pipe.load_lora_weights(str(lora_path.parent), weight_name=lora_path.name)
        lora_loaded = True
    elif args.no_lora:
        print("[INFO] Running baseline (no LoRA)")
    else:
        print("[WARN] No --lora-weights provided and --no-lora not set; running baseline")

    pipe = pipe.to(args.device)
    pipe.set_progress_bar_config(disable=False)

    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    results = []
    print(f"[INFO] Generating {len(prompts)} images at {args.resolution}x{args.resolution}")
    print(f"[INFO] steps={args.num_inference_steps}, guidance={args.guidance_scale}, seed={args.seed}")
    print()

    for idx, prompt in enumerate(prompts):
        print(f"[{idx + 1}/{len(prompts)}] {prompt}")
        # Reset generator per image for reproducibility
        generator = torch.Generator(device=args.device).manual_seed(args.seed + idx)

        t0 = time.time()
        image = pipe(
            prompt=prompt,
            height=args.resolution,
            width=args.resolution,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        ).images[0]
        elapsed = time.time() - t0

        fname = f"{idx:03d}.png"
        image.save(output_dir / fname)
        results.append({
            "index": idx,
            "prompt": prompt,
            "file": fname,
            "elapsed_seconds": round(elapsed, 2),
        })
        print(f"  -> {fname} ({elapsed:.1f}s)")

    # Save metadata
    meta = {
        "model_name": args.model_name,
        "lora_weights": str(args.lora_weights) if args.lora_weights else None,
        "lora_loaded": lora_loaded,
        "resolution": args.resolution,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "dtype": args.dtype,
        "num_images": len(results),
        "results": results,
    }
    meta_path = output_dir / "sample_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[DONE] {len(results)} images saved to {output_dir}")
    print(f"[DONE] Metadata: {meta_path}")


if __name__ == "__main__":
    main()
