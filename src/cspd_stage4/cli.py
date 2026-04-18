"""CLI entrypoint for CSPD Stage 4 — distilled dataset generation."""

from __future__ import annotations

import argparse
import json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CSPD Stage 4 CLI for distilled dataset generation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen_parser = subparsers.add_parser(
        "generate",
        help="Generate the distilled dataset from Stage 3 modes using the Stage 2 LoRA",
    )
    gen_parser.add_argument("--modes-dir", required=True, help="Stage 3 output dir with modes_index.json")
    gen_parser.add_argument("--output-dir", required=True, help="Output dir for the distilled dataset")
    gen_parser.add_argument("--lora-weights", default=None,
                            help="Stage 2 LoRA weights (.safetensors) or a full fine-tuned checkpoint dir. Omit for baseline SDXL.")
    gen_parser.add_argument("--model-name", default="stabilityai/stable-diffusion-xl-base-1.0",
                            help="Base SDXL model identifier")
    gen_parser.add_argument("--num-inference-steps", type=int, default=50)
    gen_parser.add_argument("--guidance-scale", type=float, default=7.5)
    gen_parser.add_argument("--seed", type=int, default=42)
    gen_parser.add_argument("--device", default="cuda")
    gen_parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    gen_parser.add_argument("--resolution", type=int, default=512)
    gen_parser.add_argument("--visual-mode", default="none", choices=["none", "medoid"],
                            help="'none' for text2img (baseline) or 'medoid' for img2img from the real medoid image")
    gen_parser.add_argument("--strength", type=float, default=0.8,
                            help="Img2img denoising strength; only used when --visual-mode medoid")
    gen_parser.add_argument("--refiner-model", default=None,
                            help="Optional SDXL refiner model id (e.g. stabilityai/stable-diffusion-xl-refiner-1.0)")
    gen_parser.add_argument("--refiner-strength", type=float, default=0.3,
                            help="Refiner denoising strength (0-1); lower = less change, more detail")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "generate":
        from cspd_stage4.generate import generate_distilled_dataset

        result = generate_distilled_dataset(
            modes_dir=args.modes_dir,
            output_dir=args.output_dir,
            lora_weights=args.lora_weights,
            model_name=args.model_name,
            strength=args.strength,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            device=args.device,
            dtype=args.dtype,
            resolution=args.resolution,
            visual_mode=args.visual_mode,
            refiner_model=args.refiner_model,
            refiner_strength=args.refiner_strength,
        )
        print(json.dumps({
            "output_dir": result.output_dir,
            "num_images": result.num_images,
            "num_classes": result.num_classes,
            "ipc": result.ipc,
            "images_dir": result.images_dir,
        }, indent=2))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
