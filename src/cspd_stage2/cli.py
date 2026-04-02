from __future__ import annotations

"""CLI entrypoint for CSPD Stage 2."""

import argparse
import json

from cspd_stage2.training import Stage2TrainConfig, run_stage2_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CSPD Stage 2 CLI for generative-backbone adaptation / canonical-semantic-space familiarization"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train",
        help="Build a Stage 2 paired manifest and optionally run a conservative transformer-core training scaffold",
    )
    train_parser.add_argument("--dataset-root", required=True, help="ImageFolder-style dataset root used as visual input")
    train_parser.add_argument(
        "--render-input",
        required=True,
        help="Stage 1 render records.jsonl path used as canonical text-conditioning source",
    )
    train_parser.add_argument("--output-dir", required=True, help="Run directory for Stage 2 artifacts")
    train_parser.add_argument(
        "--backbone-name",
        default="black-forest-labs/FLUX.1-Kontext-dev",
        help="Backbone identifier for the intended Stage 2 generative-backbone adaptation target",
    )
    train_parser.add_argument("--batch-size", type=int, default=4, help="Logical training batch size")
    train_parser.add_argument("--learning-rate", type=float, default=1e-4, help="Optimizer learning rate")
    train_parser.add_argument("--epochs", type=int, default=1, help="Number of epochs for the training scaffold")
    train_parser.add_argument("--max-steps", type=int, default=None, help="Optional maximum number of optimization steps")
    train_parser.add_argument("--num-workers", type=int, default=0, help="Data loader worker count placeholder")
    train_parser.add_argument("--resolution", type=int, default=512, help="Target image resolution placeholder")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    train_parser.add_argument("--weight-dtype", default="float16", help="Requested training weight dtype label")
    train_parser.add_argument("--optimizer-name", default="adamw", help="Optimizer name label")
    train_parser.add_argument("--log-every", type=int, default=10, help="Logging interval placeholder")
    train_parser.add_argument("--save-every", type=int, default=200, help="Checkpoint interval placeholder")
    train_parser.add_argument("--max-train-samples", type=int, default=None, help="Optional cap for quick dry runs")
    train_parser.add_argument("--class-name-map", default=None, help="Optional raw-folder -> readable class-name JSON")
    train_parser.add_argument("--class-archetype-map", default=None, help="Optional raw-folder -> archetype JSON")
    train_parser.add_argument("--verify-images", action="store_true", help="Probe image sizes while building the manifest")
    train_parser.add_argument("--strict-pairing", action="store_true", help="Fail when any image or render row is unmatched")
    train_parser.add_argument("--dry-run", action="store_true", help="Only prepare the Stage 2 run directory and manifest")
    train_parser.add_argument(
        "--generate-manifest-only",
        action="store_true",
        help="Alias-like explicit mode that skips any training attempt and only writes manifest artifacts",
    )
    train_parser.add_argument(
        "--allow-placeholder-loop",
        action="store_true",
        help="Run a tiny optional PyTorch placeholder loop if torch is installed; still not real FLUX training",
    )
    train_parser.add_argument(
        "--unfreeze-text-encoder",
        action="store_true",
        help="Override the default transformer-core-only plan and mark text encoder as trainable",
    )
    train_parser.add_argument(
        "--unfreeze-vae",
        action="store_true",
        help="Override the default transformer-core-only plan and mark VAE as trainable",
    )
    train_parser.add_argument(
        "--disable-train-transformer-core-only",
        action="store_true",
        help="Override the default transformer-core-only plan in config metadata",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> Stage2TrainConfig:
    return Stage2TrainConfig(
        dataset_root=args.dataset_root,
        render_input=args.render_input,
        output_dir=args.output_dir,
        backbone_name=args.backbone_name,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        max_steps=args.max_steps,
        num_workers=args.num_workers,
        resolution=args.resolution,
        seed=args.seed,
        weight_dtype=args.weight_dtype,
        optimizer_name=args.optimizer_name,
        log_every=args.log_every,
        save_every=args.save_every,
        max_train_samples=args.max_train_samples,
        class_name_map=args.class_name_map,
        class_archetype_map=args.class_archetype_map,
        verify_images=args.verify_images,
        strict_pairing=args.strict_pairing,
        dry_run=args.dry_run,
        generate_manifest_only=args.generate_manifest_only,
        allow_placeholder_loop=args.allow_placeholder_loop,
        freeze_text_encoder=not args.unfreeze_text_encoder,
        freeze_vae=not args.unfreeze_vae,
        train_transformer_core_only=not args.disable_train_transformer_core_only,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        summary = run_stage2_training(config_from_args(args))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
