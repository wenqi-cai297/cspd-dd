from __future__ import annotations

"""CLI entrypoint for CSPD Stage 2."""

import argparse
import json
from pathlib import Path
from typing import Any

from cspd_stage2.backbone import infer_backbone_family, load_module_from_reference, load_real_backbone_module
from cspd_stage2.training import (
    AdapterPlan,
    CONDITIONING_RELATED_GROUP_PATTERNS,
    Stage2TrainConfig,
    derive_stage2_output_dir,
    inspect_stage2_backbone_targets,
    resolve_effective_module_selection,
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
    train_parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional run directory for Stage 2 artifacts. Defaults to "
            "runs/stage2/train/<dataset_label>/<backbone_slug>/<timestamp>, where split-only dataset roots "
            "like train/val/valid/validation/test/testing become <parent>_<split>."
        ),
    )
    train_parser.add_argument(
        "--backbone-name",
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Backbone identifier for the intended Stage 2 generative-backbone adaptation target",
    )
    train_parser.add_argument("--batch-size", type=int, default=4, help="Logical training batch size")
    train_parser.add_argument("--learning-rate", type=float, default=2e-5, help="Optimizer learning rate")
    train_parser.add_argument("--lr-scheduler", default="constant_with_warmup", choices=["constant", "constant_with_warmup"], help="Learning-rate scheduler for the real Stage 2 path")
    train_parser.add_argument("--lr-warmup-steps", type=int, default=1000, help="Warmup steps for the real Stage 2 learning-rate scheduler")
    train_parser.add_argument("--max-grad-norm", type=float, default=0.01, help="Gradient clipping norm for the real Stage 2 path")
    train_parser.add_argument("--adam-beta1", type=float, default=0.9, help="AdamW beta1")
    train_parser.add_argument("--adam-beta2", type=float, default=0.999, help="AdamW beta2")
    train_parser.add_argument("--adam-weight-decay", type=float, default=0.0, help="AdamW weight decay")
    train_parser.add_argument("--adam-epsilon", type=float, default=1e-8, help="AdamW epsilon")
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
        default="transformer_finetuning",
        help="High-level Stage 2 focus label; default records full-transformer fine-tuning intent",
    )
    train_parser.add_argument(
        "--conditioning-objective",
        default="finetune_sdxl_unet_lora_on_real_image_and_stage1_canonical_caption_pairs",
        help="Short objective label describing the SDXL LoRA fine-tuning target for this run",
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
        help=(
            "Trainable component-group label to record and resolve into real transformer selection; may be repeated. "
            f"Known groups: {', '.join(sorted(CONDITIONING_RELATED_GROUP_PATTERNS))}"
        ),
    )
    train_parser.add_argument(
        "--module-include-pattern",
        action="append",
        dest="module_include_patterns",
        default=None,
        help="Additional module-name include pattern; may be repeated and is combined with any selected component groups",
    )
    train_parser.add_argument(
        "--module-exclude-pattern",
        action="append",
        dest="module_exclude_patterns",
        default=None,
        help="Module-name exclude pattern; may be repeated",
    )
    train_parser.add_argument(
        "--training-parameterization",
        choices=["full", "lora"],
        default="full",
        help="Choose full real-parameter transformer updates or injected LoRA-only adapter training",
    )
    train_parser.add_argument(
        "--adapter-type",
        default="lora",
        help="Adapter strategy label recorded in the Stage 2 plan metadata; real training currently supports only LoRA",
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
        "--disable-accelerate",
        action="store_true",
        help="Disable accelerate-managed preparation and run the minimal single-process device path instead",
    )
    train_parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps forwarded to accelerate during real Stage 2 training",
    )
    train_parser.add_argument(
        "--dataloader-drop-last",
        action="store_true",
        help="Drop the last incomplete batch when building the Stage 2 dataloader",
    )
    train_parser.add_argument(
        "--disable-gradient-checkpointing",
        action="store_true",
        help="Disable the default attempt to enable gradient checkpointing on the loaded transformer when supported",
    )
    train_parser.add_argument("--wandb", action="store_true", help="Enable optional Weights & Biases logging when wandb is installed")
    train_parser.add_argument("--wandb-project", default="cspd-stage2", help="W&B project name")
    train_parser.add_argument("--wandb-entity", default=None, help="Optional W&B entity/team")
    train_parser.add_argument("--wandb-run-name", default=None, help="Optional W&B run name override")
    train_parser.add_argument("--wandb-tag", action="append", dest="wandb_tags", default=None, help="Optional W&B tag; may be repeated")
    train_parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online", help="W&B mode passed to wandb.init")
    train_parser.add_argument("--wandb-dir", default=None, help="Optional local W&B run directory")
    train_parser.add_argument("--wandb-resume", default=None, help="Optional W&B resume mode, e.g. allow or must")
    train_parser.add_argument("--wandb-run-id", default=None, help="Optional explicit W&B run id for resume/reuse")
    train_parser.add_argument(
        "--inspect-limit",
        type=int,
        default=200,
        help="Maximum matched modules to emit during optional module inspection",
    )
    train_parser.add_argument(
        "--apply-real-module-selection",
        action="store_true",
        help="Force resolved include/exclude rules onto the real module tree via requires_grad even during inspection-oriented runs",
    )
    train_parser.add_argument(
        "--inject-adapters-on-real-module",
        action="store_true",
        help="If --inspect-module-reference is provided, inject lightweight LoRA adapters into matching real torch modules",
    )

    train_parser.add_argument('--sdxl-official-script', default=None, help='Path to the official diffusers SDXL LoRA training script; defaults to CSPD_STAGE2_SDXL_SCRIPT or train_text_to_image_lora_sdxl.py on PATH')
    train_parser.add_argument('--sdxl-num-processes', type=int, default=None, help='Optional accelerate --num_processes override for the official SDXL launch path')
    train_parser.add_argument('--sdxl-accelerate-extra-arg', action='append', dest='sdxl_accelerate_extra_args', default=None, help='Extra argument forwarded to accelerate launch for the SDXL official path; may be repeated')
    train_parser.add_argument('--sdxl-mixed-precision', default='fp16', help='mixed_precision value for the official diffusers SDXL launch path')
    train_parser.add_argument('--sdxl-lr-scheduler', default='cosine', help='Learning-rate scheduler forwarded to the official diffusers SDXL script')
    train_parser.add_argument('--sdxl-lr-warmup-steps', type=int, default=500, help='Warmup steps forwarded to the official diffusers SDXL script')
    train_parser.add_argument('--sdxl-validation-epochs', type=int, default=1, help='validation_epochs forwarded to the official diffusers SDXL script')
    train_parser.add_argument('--sdxl-validation-prompt', default=None, help='Optional validation prompt forwarded to the official diffusers SDXL script')
    train_parser.add_argument('--sdxl-report-to', default='none', help='report_to backend for the official diffusers SDXL script, e.g. none or wandb')
    train_parser.add_argument('--sdxl-use-8bit-adam', action='store_true', help='Forward --use_8bit_adam to the official diffusers SDXL script')
    train_parser.add_argument('--sdxl-enable-xformers', action='store_true', help='Forward --enable_xformers_memory_efficient_attention to the official diffusers SDXL script')
    train_parser.add_argument('--sdxl-disable-gradient-checkpointing', action='store_true', help='Disable --gradient_checkpointing on the official diffusers SDXL script path')
    train_parser.add_argument('--sdxl-train-text-encoder', action='store_true', help='Forward --train_text_encoder to the official diffusers SDXL script')
    train_parser.add_argument('--sdxl-caption-dropout-probability', type=float, default=None, help='Optional caption dropout probability forwarded to the official diffusers SDXL script')
    train_parser.add_argument('--sdxl-noise-offset', type=float, default=0.05, help='Noise offset for improved contrast/brightness range (default 0.05)')
    train_parser.add_argument('--sdxl-snr-gamma', type=float, default=5.0, help='Min-SNR gamma for balanced timestep loss weighting (default 5.0, None to disable)')
    train_parser.add_argument('--sdxl-extra-arg', action='append', dest='sdxl_extra_args', default=None, help='Extra raw argument appended to the official diffusers SDXL script command; may be repeated')


    inspect_parser = subparsers.add_parser(
        "inspect-targets",
        help="Inspect candidate trainable module names on an explicitly provided torch module tree",
    )
    inspect_parser.add_argument("--backbone-name", default="stabilityai/stable-diffusion-xl-base-1.0")
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

    dump_parser = subparsers.add_parser(
        "dump-modules",
        help="Dump real loaded backbone module names to text files for later trainable-component review",
    )
    dump_parser.add_argument("--backbone-name", default="stabilityai/stable-diffusion-xl-base-1.0")
    dump_parser.add_argument(
        "--module-reference",
        default=None,
        help="Optional Python object reference in the form package.module:object_or_factory instead of loading a real backbone",
    )
    dump_parser.add_argument(
        "--load-backbone",
        action="store_true",
        help="Load the real backbone through the Stage 2 loader instead of requiring --module-reference",
    )
    dump_parser.add_argument("--output-dir", required=True, help="Directory where dump text/json artifacts should be written")
    dump_parser.add_argument(
        "--torch-dtype",
        default="bfloat16",
        help="Torch dtype label used when attempting a real diffusers backbone load",
    )
    dump_parser.add_argument(
        "--device",
        default=None,
        help="Optional device passed to pipeline.to(...) after a real load, e.g. cuda or cpu",
    )
    dump_parser.add_argument(
        "--device-map",
        default=None,
        help="Optional device_map forwarded to diffusers from_pretrained when attempting a real load",
    )
    dump_parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Require real backbone loads to use only the local Hugging Face cache",
    )
    dump_parser.add_argument(
        "--component",
        default=None,
        help="Optional component name to inspect from a real loaded backbone, e.g. transformer or text_encoder",
    )
    dump_parser.add_argument(
        "--keyword",
        action="append",
        dest="keywords",
        default=None,
        help="Keyword filter for a focused module-name text dump; may be repeated",
    )
    dump_parser.add_argument(
        "--module-limit",
        type=int,
        default=None,
        help="Optional limit for the full named_modules dump; defaults to all modules",
    )
    dump_parser.add_argument(
        "--child-limit",
        type=int,
        default=None,
        help="Optional limit for named_children dumps; defaults to all children",
    )
    return parser


def _resolve_conditioning_objective(backbone_name: str, requested_objective: str) -> str:
    # SDXL is the only supported family; the helper stays for future expansion.
    return requested_objective



def config_from_args(args: argparse.Namespace) -> Stage2TrainConfig:
    output_dir = args.output_dir or derive_stage2_output_dir(args.dataset_root, args.backbone_name)
    config = Stage2TrainConfig(
        dataset_root=args.dataset_root,
        render_input=args.render_input,
        output_dir=output_dir,
        backbone_name=args.backbone_name,
        wandb_enabled=args.wandb and args.wandb_mode != "disabled",
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_tags=args.wandb_tags or [],
        wandb_mode=args.wandb_mode,
        wandb_dir=args.wandb_dir,
        wandb_resume=args.wandb_resume,
        wandb_run_id=args.wandb_run_id,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler=args.lr_scheduler,
        lr_warmup_steps=args.lr_warmup_steps,
        max_grad_norm=args.max_grad_norm,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_weight_decay=args.adam_weight_decay,
        adam_epsilon=args.adam_epsilon,
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
        freeze_text_encoder=not args.unfreeze_text_encoder,
        freeze_vae=not args.unfreeze_vae,
        train_transformer_core_only=not args.disable_train_transformer_core_only,
        stage2_focus=args.stage2_focus,
        conditioning_objective=_resolve_conditioning_objective(args.backbone_name, args.conditioning_objective),
        conditioning_text_field=args.conditioning_text_field,
        trainable_component_groups=args.trainable_component_groups or (["conditioning_transformer"] if args.training_parameterization == "lora" else ["full_transformer"]),
        module_include_patterns=args.module_include_patterns or [],
        module_exclude_patterns=args.module_exclude_patterns or [
            "vae",
            "autoencoder",
            "decoder",
            "image_encoder",
        ],
        training_parameterization=args.training_parameterization,
        adapter_plan=AdapterPlan(
            adapter_type=args.adapter_type,
            rank=args.adapter_rank,
            alpha=args.adapter_alpha,
            dropout=args.adapter_dropout,
            bias=args.adapter_bias,
            master_weight_dtype=None,
            target_module_patterns=args.module_include_patterns or [],
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
        use_accelerate=not args.disable_accelerate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_drop_last=args.dataloader_drop_last,
        enable_gradient_checkpointing=not args.disable_gradient_checkpointing,
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
    config.adapter_plan.target_module_patterns = (
        config.adapter_plan.target_module_patterns or resolve_effective_module_selection(config)["effective_include_patterns"]
    )
    return config


def _named_children_lines(module: Any, *, limit: int | None = None) -> list[str]:
    if not hasattr(module, "named_children"):
        return []
    lines: list[str] = []
    for index, (name, child) in enumerate(module.named_children()):
        if limit is not None and index >= limit:
            break
        lines.append(f"{name}\t{type(child).__name__}")
    return lines


def _top_level_component_lines(module: Any, *, limit: int | None = None) -> list[str]:
    common_component_names = [
        "transformer",
        "transformer_2d",
        "unet",
        "vae",
        "text_encoder",
        "text_encoder_2",
        "image_encoder",
        "safety_checker",
        "scheduler",
        "tokenizer",
        "tokenizer_2",
        "feature_extractor",
    ]

    seen: set[str] = set()
    lines: list[str] = []

    def maybe_add(name: str, value: Any) -> None:
        if name in seen or value is None:
            return
        seen.add(name)
        lines.append(f"{name}\t{type(value).__name__}")

    for name in common_component_names:
        maybe_add(name, getattr(module, name, None))
        if limit is not None and len(lines) >= limit:
            return lines[:limit]

    children = _named_children_lines(module, limit=None)
    for line in children:
        name = line.split("\t", 1)[0]
        if name in seen:
            continue
        lines.append(line)
        seen.add(name)
        if limit is not None and len(lines) >= limit:
            return lines[:limit]

    if not lines and hasattr(module, "components"):
        components = getattr(module, "components")
        if isinstance(components, dict):
            for name, value in components.items():
                maybe_add(str(name), value)
                if limit is not None and len(lines) >= limit:
                    return lines[:limit]

    return lines


def _named_modules_lines(module: Any, *, limit: int | None = None) -> list[str]:
    if not hasattr(module, "named_modules"):
        return []
    lines: list[str] = []
    for index, (name, child) in enumerate(module.named_modules()):
        if limit is not None and index >= limit:
            break
        qualified_name = name or "<root>"
        lines.append(f"{qualified_name}\t{type(child).__name__}")
    return lines


def _keyword_filtered_lines(lines: list[str], keywords: list[str]) -> dict[str, list[str]]:
    lowered_pairs = [(keyword, keyword.lower()) for keyword in keywords]
    result: dict[str, list[str]] = {}
    for original_keyword, lowered_keyword in lowered_pairs:
        result[original_keyword] = [line for line in lines if lowered_keyword in line.lower()]
    return result


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def _run_dump_modules(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    keywords = args.keywords or ["context", "embed", "cond", "attn", "cross", "proj", "txt", "block"]

    if args.load_backbone:
        load_result = load_real_backbone_module(
            args.backbone_name,
            torch_dtype=args.torch_dtype,
            device=args.device,
            device_map=args.device_map,
            local_files_only=args.local_files_only,
            component=args.component,
            allow_unimplemented=False,
        )
        root_module = load_result.root_module or load_result.module
        focus_module = load_result.module
        focus_module_name = load_result.resolved_module_name or args.component or "resolved_module"
        load_summary = {
            "loader": load_result.loader_name,
            "loader_status": load_result.implementation_status,
            "loader_notes": list(load_result.notes or []),
            "resolved_module_name": load_result.resolved_module_name,
            "resolved_module_type": load_result.resolved_module_type,
            "local_files_only": args.local_files_only,
            "torch_dtype": args.torch_dtype,
            "device": args.device,
            "device_map": args.device_map,
            "component": args.component,
        }
        module_reference = None
    else:
        if not args.module_reference:
            raise ValueError("module_reference is required unless load_backbone=True")
        focus_module = load_module_from_reference(args.module_reference)
        root_module = focus_module
        focus_module_name = args.component or "module_reference"
        load_summary = None
        module_reference = args.module_reference

    pipeline_top_level_lines = _top_level_component_lines(root_module, limit=args.child_limit)
    root_children_lines = _named_children_lines(root_module, limit=args.child_limit)
    focus_children_lines = _named_children_lines(focus_module, limit=args.child_limit)
    focus_module_lines = _named_modules_lines(focus_module, limit=args.module_limit)
    filtered = _keyword_filtered_lines(focus_module_lines, keywords)

    _write_lines(output_dir / "pipeline_top_level_components.txt", pipeline_top_level_lines)
    _write_lines(output_dir / "pipeline_named_children.txt", root_children_lines)
    _write_lines(output_dir / f"{focus_module_name}_named_children.txt", focus_children_lines)
    _write_lines(output_dir / f"{focus_module_name}_named_modules.txt", focus_module_lines)
    for keyword, matched_lines in filtered.items():
        safe_keyword = keyword.replace("/", "_").replace("\\", "_").replace(" ", "_")
        _write_lines(output_dir / "filtered" / f"keyword_{safe_keyword}.txt", matched_lines)

    summary = {
        "backbone_name": args.backbone_name,
        "module_reference": module_reference,
        "loaded_backbone": bool(args.load_backbone),
        "load_summary": load_summary,
        "output_dir": str(output_dir.resolve()),
        "focus_module_name": focus_module_name,
        "focus_module_type": type(focus_module).__name__,
        "root_module_type": type(root_module).__name__,
        "keywords": list(keywords),
        "artifacts": {
            "pipeline_top_level_components": str((output_dir / "pipeline_top_level_components.txt").resolve()),
            "pipeline_named_children": str((output_dir / "pipeline_named_children.txt").resolve()),
            "focus_named_children": str((output_dir / f"{focus_module_name}_named_children.txt").resolve()),
            "focus_named_modules": str((output_dir / f"{focus_module_name}_named_modules.txt").resolve()),
            "filtered_dir": str((output_dir / "filtered").resolve()),
        },
        "counts": {
            "pipeline_top_level_components": len(pipeline_top_level_lines),
            "pipeline_named_children": len(root_children_lines),
            "focus_named_children": len(focus_children_lines),
            "focus_named_modules": len(focus_module_lines),
            "filtered_matches": {keyword: len(lines) for keyword, lines in filtered.items()},
        },
        "notes": [
            "Use pipeline_top_level_components.txt to see large functional components exposed by the loaded pipeline.",
            "Use pipeline_named_children.txt as the raw named_children view when the pipeline exposes it.",
            "Use the focus-module named_modules dump plus filtered keyword files to decide which conditioning-related modules to tune.",
        ],
    }
    (output_dir / "dump_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        summary = run_stage2_training(config_from_args(args))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if args.command == "inspect-targets":
        include_patterns = args.module_include_patterns or ["*"]
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

    if args.command == "dump-modules":
        summary = _run_dump_modules(args)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
