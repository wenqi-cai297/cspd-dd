from __future__ import annotations

"""CLI entrypoint for CSPD Stage 2 (SDXL LoRA only)."""

import argparse
import json

from cspd_stage2.training import (
    AdapterPlan,
    Stage2TrainConfig,
    derive_stage2_output_dir,
    run_stage2_training,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CSPD Stage 2 CLI for SDXL LoRA fine-tuning on Stage 1 canonical captions"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train",
        help="Build the Stage 2 paired manifest and launch SDXL LoRA training via the official diffusers trainer",
    )
    train_parser.add_argument("--dataset-root", required=True, help="ImageFolder-style dataset root")
    train_parser.add_argument("--render-input", required=True, help="Stage 1 render records.jsonl path")
    train_parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional run directory. Defaults to "
            "runs/stage2/train/<dataset_label>/<backbone_slug>/<timestamp>."
        ),
    )
    train_parser.add_argument(
        "--backbone-name",
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="SDXL backbone identifier (the only supported family)",
    )

    # Training knobs consumed by the SDXL wrapper or upstream pairing logic
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--learning-rate", type=float, default=2e-5)
    train_parser.add_argument("--epochs", type=int, default=9)
    train_parser.add_argument("--max-steps", type=int, default=None, help="Optional hard cap on training steps")
    train_parser.add_argument("--resolution", type=int, default=512)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--save-every", type=int, default=200, help="Forwarded as --checkpointing_steps")
    train_parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count inside the official trainer")
    train_parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    train_parser.add_argument("--adapter-rank", type=int, default=64, help="LoRA rank forwarded as --rank to the trainer")

    # Pairing + dry-run controls
    train_parser.add_argument("--class-name-map", default=None)
    train_parser.add_argument("--class-archetype-map", default=None)
    train_parser.add_argument("--verify-images", action="store_true")
    train_parser.add_argument("--strict-pairing", action="store_true")
    train_parser.add_argument("--dry-run", action="store_true", help="Only prepare the Stage 2 run directory and manifest")
    train_parser.add_argument("--generate-manifest-only", action="store_true", help="Skip any training and only write manifest artifacts")
    train_parser.add_argument("--max-train-samples", type=int, default=None, help="Cap pair count for quick smoke runs")

    # accelerate / SDXL-specific knobs
    train_parser.add_argument("--disable-accelerate", action="store_true", help="Run the official trainer directly without accelerate launch")
    train_parser.add_argument("--sdxl-official-script", default=None, help="Explicit path to diffusers' train_text_to_image_lora_sdxl.py")
    train_parser.add_argument("--sdxl-num-processes", type=int, default=None, help="Override accelerate --num_processes")
    train_parser.add_argument("--sdxl-accelerate-extra-arg", action="append", dest="sdxl_accelerate_extra_args", default=None)
    train_parser.add_argument("--sdxl-mixed-precision", default="fp16", choices=["fp16", "bf16", "no"])
    train_parser.add_argument("--sdxl-lr-scheduler", default="cosine")
    train_parser.add_argument("--sdxl-lr-warmup-steps", type=int, default=500)
    train_parser.add_argument("--sdxl-validation-epochs", type=int, default=1)
    train_parser.add_argument("--sdxl-validation-prompt", default=None)
    train_parser.add_argument("--sdxl-report-to", default="none")
    train_parser.add_argument("--sdxl-use-8bit-adam", action="store_true")
    train_parser.add_argument("--sdxl-enable-xformers", action="store_true")
    train_parser.add_argument("--sdxl-disable-gradient-checkpointing", action="store_true")
    train_parser.add_argument("--sdxl-train-text-encoder", action="store_true")
    train_parser.add_argument("--sdxl-caption-dropout-probability", type=float, default=None)
    train_parser.add_argument("--sdxl-noise-offset", type=float, default=0.05)
    train_parser.add_argument("--sdxl-snr-gamma", type=float, default=5.0)
    train_parser.add_argument("--sdxl-extra-arg", action="append", dest="sdxl_extra_args", default=None)

    return parser


def config_from_args(args: argparse.Namespace) -> Stage2TrainConfig:
    output_dir = args.output_dir or derive_stage2_output_dir(args.dataset_root, args.backbone_name)
    return Stage2TrainConfig(
        dataset_root=args.dataset_root,
        render_input=args.render_input,
        output_dir=output_dir,
        backbone_name=args.backbone_name,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        max_steps=args.max_steps,
        num_workers=args.num_workers,
        resolution=args.resolution,
        seed=args.seed,
        save_every=args.save_every,
        max_train_samples=args.max_train_samples,
        class_name_map=args.class_name_map,
        class_archetype_map=args.class_archetype_map,
        verify_images=args.verify_images,
        strict_pairing=args.strict_pairing,
        dry_run=args.dry_run,
        generate_manifest_only=args.generate_manifest_only,
        use_accelerate=not args.disable_accelerate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        adapter_plan=AdapterPlan(rank=args.adapter_rank, alpha=float(args.adapter_rank)),
        sdxl_official_script=args.sdxl_official_script,
        sdxl_num_processes=args.sdxl_num_processes,
        sdxl_accelerate_extra_args=args.sdxl_accelerate_extra_args or [],
        sdxl_mixed_precision=args.sdxl_mixed_precision,
        sdxl_lr_scheduler=args.sdxl_lr_scheduler,
        sdxl_lr_warmup_steps=args.sdxl_lr_warmup_steps,
        sdxl_validation_epochs=args.sdxl_validation_epochs,
        sdxl_validation_prompt=args.sdxl_validation_prompt,
        sdxl_report_to=args.sdxl_report_to,
        sdxl_use_8bit_adam=args.sdxl_use_8bit_adam,
        sdxl_enable_xformers=args.sdxl_enable_xformers,
        sdxl_gradient_checkpointing=not args.sdxl_disable_gradient_checkpointing,
        sdxl_train_text_encoder=args.sdxl_train_text_encoder,
        sdxl_caption_dropout_probability=args.sdxl_caption_dropout_probability,
        sdxl_noise_offset=args.sdxl_noise_offset,
        sdxl_snr_gamma=args.sdxl_snr_gamma,
        sdxl_extra_args=args.sdxl_extra_args or [],
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        summary = run_stage2_training(config_from_args(args))
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        training_result = summary.get("training_result") if isinstance(summary, dict) else None
        failed = bool(summary.get("top_level_failure")) if isinstance(summary, dict) else True
        if isinstance(training_result, dict):
            failed = failed or training_result.get("status") in {
                "failed",
                "failed_before_training",
                "failed_before_training_setup_complete",
                "unsupported_backbone",
            }
            failed = failed or int(training_result.get("returncode", 0) or 0) != 0
        if failed:
            raise SystemExit(1)
        return

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
