from __future__ import annotations

"""CLI entrypoint for CSPD Stage 2."""

import argparse
import json

from cspd_stage2.training import (
    AdapterPlan,
    Stage2TrainConfig,
    inspect_stage2_backbone_targets,
    run_stage2_training,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CSPD Stage 2 CLI for generative-backbone adaptation / canonical-semantic-space familiarization"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train",
        help="Build a Stage 2 paired manifest and optionally run a conservative text-conditioning adaptation scaffold",
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
    train_parser.add_argument(
        "--stage2-focus",
        default="text_conditioning_adaptation",
        help="High-level Stage 2 focus label; default keeps the implementation honest about text-conditioning adaptation",
    )
    train_parser.add_argument(
        "--conditioning-objective",
        default="align_stage1_canonical_captions_with_backbone_text_conditioning_path",
        help="Short objective label describing what part of conditioning adaptation this run targets",
    )
    train_parser.add_argument(
        "--conditioning-text-field",
        default="canonical_caption",
        help="Manifest field treated as the canonical text-conditioning source",
    )
    train_parser.add_argument(
        "--trainable-component-group",
        action="append",
        dest="trainable_component_groups",
        default=None,
        help="Trainable component-group label to record in the plan; may be repeated",
    )
    train_parser.add_argument(
        "--module-include-pattern",
        action="append",
        dest="module_include_patterns",
        default=None,
        help="Module-name include pattern placeholder for future backbone-specific selection; may be repeated",
    )
    train_parser.add_argument(
        "--module-exclude-pattern",
        action="append",
        dest="module_exclude_patterns",
        default=None,
        help="Module-name exclude pattern placeholder for future backbone-specific selection; may be repeated",
    )
    train_parser.add_argument(
        "--adapter-type",
        default="lora",
        help="Adapter strategy label recorded in the Stage 2 plan metadata",
    )
    train_parser.add_argument("--adapter-rank", type=int, default=16, help="Adapter rank placeholder")
    train_parser.add_argument("--adapter-alpha", type=float, default=16.0, help="Adapter alpha placeholder")
    train_parser.add_argument("--adapter-dropout", type=float, default=0.0, help="Adapter dropout placeholder")
    train_parser.add_argument(
        "--adapter-bias",
        default="none",
        help="Adapter bias mode placeholder",
    )
    train_parser.add_argument(
        "--inspect-module-reference",
        default=None,
        help="Optional Python object reference in the form package.module:object_or_factory used for real module inspection",
    )
    train_parser.add_argument(
        "--backbone-torch-dtype",
        default="bfloat16",
        help="Torch dtype label used when attempting a real diffusers backbone load",
    )
    train_parser.add_argument(
        "--backbone-device",
        default=None,
        help="Optional device passed to pipeline.to(...) after a real load, e.g. cuda or cpu",
    )
    train_parser.add_argument(
        "--backbone-device-map",
        default=None,
        help="Optional device_map forwarded to diffusers from_pretrained when attempting a real load",
    )
    train_parser.add_argument(
        "--backbone-local-files-only",
        action="store_true",
        help="Require real backbone loads to use only the local Hugging Face cache",
    )
    train_parser.add_argument(
        "--backbone-component",
        default=None,
        help="Optional component name to inspect from a real loaded backbone, e.g. transformer or text_encoder",
    )
    train_parser.add_argument(
        "--inspect-limit",
        type=int,
        default=200,
        help="Maximum matched modules to emit during optional module inspection",
    )
    train_parser.add_argument(
        "--apply-real-module-selection",
        action="store_true",
        help="If --inspect-module-reference is provided, apply include/exclude rules to the real module tree via requires_grad",
    )
    train_parser.add_argument(
        "--inject-adapters-on-real-module",
        action="store_true",
        help="If --inspect-module-reference is provided, inject lightweight LoRA adapters into matching real torch modules",
    )

    inspect_parser = subparsers.add_parser(
        "inspect-targets",
        help="Inspect candidate trainable module names on an explicitly provided torch module tree",
    )
    inspect_parser.add_argument("--backbone-name", default="black-forest-labs/FLUX.1-Kontext-dev")
    inspect_parser.add_argument(
        "--module-reference",
        default=None,
        help="Python object reference in the form package.module:object_or_factory",
    )
    inspect_parser.add_argument(
        "--load-backbone",
        action="store_true",
        help="Load the real backbone through the Stage 2 loader instead of requiring --module-reference",
    )
    inspect_parser.add_argument(
        "--torch-dtype",
        default="bfloat16",
        help="Torch dtype label used when attempting a real diffusers backbone load",
    )
    inspect_parser.add_argument(
        "--device",
        default=None,
        help="Optional device passed to pipeline.to(...) after a real load, e.g. cuda or cpu",
    )
    inspect_parser.add_argument(
        "--device-map",
        default=None,
        help="Optional device_map forwarded to diffusers from_pretrained when attempting a real load",
    )
    inspect_parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Require real backbone loads to use only the local Hugging Face cache",
    )
    inspect_parser.add_argument(
        "--component",
        default=None,
        help="Optional component name to inspect from a real loaded backbone, e.g. transformer or text_encoder",
    )
    inspect_parser.add_argument(
        "--module-include-pattern",
        action="append",
        dest="module_include_patterns",
        default=None,
        help="Include pattern to match candidate module names; may be repeated",
    )
    inspect_parser.add_argument(
        "--module-exclude-pattern",
        action="append",
        dest="module_exclude_patterns",
        default=None,
        help="Exclude pattern to filter module names; may be repeated",
    )
    inspect_parser.add_argument("--limit", type=int, default=200)
    inspect_parser.add_argument(
        "--apply-selection",
        action="store_true",
        help="Apply include/exclude rules to requires_grad on the provided module tree before reporting",
    )
    inspect_parser.add_argument(
        "--inject-adapters",
        action="store_true",
        help="Inject lightweight LoRA adapters into matching real torch modules before reporting",
    )
    inspect_parser.add_argument("--adapter-rank", type=int, default=16, help="Adapter rank for optional injection")
    inspect_parser.add_argument("--adapter-alpha", type=float, default=16.0, help="Adapter alpha for optional injection")
    inspect_parser.add_argument("--adapter-dropout", type=float, default=0.0, help="Adapter dropout for optional injection")
    inspect_parser.add_argument("--adapter-bias", default="none", help="Adapter bias label for metadata")
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
        stage2_focus=args.stage2_focus,
        conditioning_objective=args.conditioning_objective,
        conditioning_text_field=args.conditioning_text_field,
        trainable_component_groups=args.trainable_component_groups or [
            "conditioning_bridge",
            "cross_attention",
            "transformer_text_conditioning",
        ],
        module_include_patterns=args.module_include_patterns or [
            "transformer.*attn",
            "transformer.*cross_attn",
            "transformer.*context",
            "transformer.*txt",
            "context_embedder",
            "conditioning_bridge",
        ],
        module_exclude_patterns=args.module_exclude_patterns or [
            "vae",
            "autoencoder",
            "decoder",
            "image_encoder",
        ],
        adapter_plan=AdapterPlan(
            adapter_type=args.adapter_type,
            rank=args.adapter_rank,
            alpha=args.adapter_alpha,
            dropout=args.adapter_dropout,
            bias=args.adapter_bias,
            target_module_patterns=args.module_include_patterns or [
                "transformer.*attn",
                "transformer.*cross_attn",
                "transformer.*context",
                "transformer.*txt",
                "context_embedder",
                "conditioning_bridge",
            ],
            exclude_module_patterns=args.module_exclude_patterns or [
                "vae",
                "autoencoder",
                "decoder",
                "image_encoder",
            ],
        ),
        inspect_module_reference=args.inspect_module_reference,
        inspect_limit=args.inspect_limit,
        apply_real_module_selection=args.apply_real_module_selection,
        inject_adapters_on_real_module=args.inject_adapters_on_real_module,
        backbone_torch_dtype=args.backbone_torch_dtype,
        backbone_device=args.backbone_device,
        backbone_device_map=args.backbone_device_map,
        backbone_local_files_only=args.backbone_local_files_only,
        backbone_component=args.backbone_component,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        summary = run_stage2_training(config_from_args(args))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if args.command == "inspect-targets":
        include_patterns = args.module_include_patterns or [
            "transformer.*attn",
            "transformer.*cross_attn",
            "transformer.*context",
            "transformer.*txt",
            "context_embedder",
            "conditioning_bridge",
        ]
        exclude_patterns = args.module_exclude_patterns or [
            "vae",
            "autoencoder",
            "decoder",
            "image_encoder",
        ]
        summary = inspect_stage2_backbone_targets(
            backbone_name=args.backbone_name,
            module_reference=args.module_reference,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            limit=args.limit,
            apply_selection=args.apply_selection,
            inject_adapters=args.inject_adapters,
            adapter_plan=AdapterPlan(
                adapter_type="lora",
                rank=args.adapter_rank,
                alpha=args.adapter_alpha,
                dropout=args.adapter_dropout,
                bias=args.adapter_bias,
                target_module_patterns=include_patterns,
                exclude_module_patterns=exclude_patterns,
            ),
            load_backbone=args.load_backbone,
            torch_dtype=args.torch_dtype,
            device=args.device,
            device_map=args.device_map,
            local_files_only=args.local_files_only,
            component=args.component,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
