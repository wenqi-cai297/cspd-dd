"""CLI entrypoint for CSPD Stage 4 — distilled dataset generation."""

from __future__ import annotations

import argparse
import json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CSPD Stage 4 CLI for distilled dataset generation"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- generate ---
    gen_parser = subparsers.add_parser(
        "generate",
        help="Generate distilled dataset from Stage 3 modes using Stage 2 LoRA backbone",
    )
    gen_parser.add_argument("--modes-dir", required=True, help="Directory with Stage 3 mode outputs (modes_index.json)")
    gen_parser.add_argument("--output-dir", required=True, help="Directory for distilled dataset output")
    gen_parser.add_argument("--lora-weights", default=None, help="Path to Stage 2 LoRA weights (.safetensors). Omit for baseline SDXL.")
    gen_parser.add_argument("--model-name", default="stabilityai/stable-diffusion-xl-base-1.0", help="SDXL model identifier")
    gen_parser.add_argument("--num-inference-steps", type=int, default=50, help="Diffusion sampling steps")
    gen_parser.add_argument("--guidance-scale", type=float, default=7.5, help="Classifier-free guidance scale")
    gen_parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    gen_parser.add_argument("--device", default="cuda", help="Torch device")
    gen_parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"], help="Weight dtype")
    gen_parser.add_argument("--resolution", type=int, default=512, help="Output image resolution")
    gen_parser.add_argument("--refiner-model", default=None, help="SDXL refiner model ID (e.g. stabilityai/stable-diffusion-xl-refiner-1.0). Adds detail/sharpness.")
    gen_parser.add_argument("--refiner-strength", type=float, default=0.3, help="Refiner denoising strength (0-1). Lower=more detail, less change.")
    gen_parser.add_argument("--visual-mode", default="none", choices=["none", "medoid"],
                            help="'none' for text2img (recommended), 'medoid' for img2img from real medoid image")
    gen_parser.add_argument("--strength", type=float, default=0.8,
                            help="Img2img denoising strength (0-1). Ignored when visual-mode=none.")

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
