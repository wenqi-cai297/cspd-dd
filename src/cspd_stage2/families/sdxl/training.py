from __future__ import annotations

"""Thin SDXL Stage 2 orchestration around the official diffusers LoRA trainer."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from cspd_stage1.io_utils import write_json, write_jsonl


def _sanitize_copy_name(value: str) -> str:
    """Build a filesystem-safe relative filename for a copied training image."""
    safe = value.replace('\\', '__').replace('/', '__').replace(':', '_')
    safe = ''.join(ch if ch.isalnum() or ch in '._-' else '_' for ch in safe)
    return safe or 'image'


def materialize_sdxl_training_dataset(*, pairs: list[Any], output_dir: str | Path, image_subdir: str = 'images') -> dict[str, Any]:
    """Copy paired images and emit metadata.jsonl for official diffusers imagefolder training.

    Args:
        pairs: Stage 2 pair records carrying image_path and canonical_caption.
        output_dir: Target directory for the diffusers-style dataset.
        image_subdir: Relative subdirectory under output_dir that will hold copied images.

    Returns:
        Summary dict describing the materialized dataset layout and artifacts.
    """
    dataset_dir = Path(output_dir)
    images_dir = dataset_dir / image_subdir
    images_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows: list[dict[str, Any]] = []
    copied_files: list[str] = []

    for index, pair in enumerate(pairs):
        source_path = Path(str(getattr(pair, 'image_path'))).expanduser().resolve()
        if not source_path.exists():
            raise FileNotFoundError(f'SDXL materialization source image not found: {source_path}')
        relative_key = str(getattr(pair, 'relative_image_path', '') or source_path.name)
        suffix = source_path.suffix or '.png'
        copy_name = f"{index:06d}_{_sanitize_copy_name(relative_key)}"
        if not copy_name.lower().endswith(suffix.lower()):
            copy_name += suffix
        target_path = images_dir / copy_name
        shutil.copy2(source_path, target_path)
        copied_files.append(str(target_path.resolve()))
        metadata_rows.append({
            'file_name': f'{image_subdir}/{copy_name}'.replace('\\', '/'),
            'text': str(getattr(pair, 'canonical_caption', '')),
            'pair_id': str(getattr(pair, 'pair_id', '')),
            'record_id': str(getattr(pair, 'record_id', '')),
            'sample_id': str(getattr(pair, 'sample_id', '')),
            'relative_image_path': str(getattr(pair, 'relative_image_path', '')),
            'class_name': str(getattr(pair, 'class_name', '')),
            'class_name_raw': str(getattr(pair, 'class_name_raw', '')),
            'archetype': str(getattr(pair, 'archetype', '')),
        })

    metadata_path = dataset_dir / 'metadata.jsonl'
    write_jsonl(metadata_path, metadata_rows)
    write_json(dataset_dir / 'dataset_summary.json', {
        'num_examples': len(metadata_rows),
        'image_subdir': image_subdir,
        'metadata_path': str(metadata_path.resolve()),
        'images_dir': str(images_dir.resolve()),
        'copied_file_count': len(copied_files),
        'copied_file_samples': copied_files[:20],
        'truncated_copied_file_samples': len(copied_files) > 20,
    })
    return {
        'dataset_dir': str(dataset_dir.resolve()),
        'images_dir': str(images_dir.resolve()),
        'metadata_path': str(metadata_path.resolve()),
        'num_examples': len(metadata_rows),
        'image_subdir': image_subdir,
    }


def _resolve_official_sdxl_script(config: Any) -> str:
    """Resolve the official diffusers SDXL LoRA training script path."""
    explicit = str(getattr(config, 'sdxl_official_script', '') or '').strip()
    if explicit:
        return explicit
    env_value = str(os.environ.get('CSPD_STAGE2_SDXL_SCRIPT', '')).strip()
    if env_value:
        return env_value
    return 'train_text_to_image_lora_sdxl.py'


def _build_accelerate_prefix(config: Any) -> list[str]:
    """Build the command prefix used to launch the official diffusers script."""
    if not bool(getattr(config, 'use_accelerate', True)):
        return [sys.executable]
    prefix = ['accelerate', 'launch']
    num_processes = getattr(config, 'sdxl_num_processes', None)
    if num_processes is not None and int(num_processes) > 0:
        prefix.extend(['--num_processes', str(int(num_processes))])
    extra_args = list(getattr(config, 'sdxl_accelerate_extra_args', []) or [])
    prefix.extend(extra_args)
    return prefix


def _build_sdxl_official_command(*, config: Any, script_path: str, materialized: dict[str, Any], run_dir: Path) -> list[str]:
    """Translate Stage 2 config into the official diffusers SDXL LoRA CLI."""
    command = _build_accelerate_prefix(config)
    if command == [sys.executable]:
        command.append(script_path)
    else:
        command.append(script_path)
    train_batch_size = max(int(getattr(config, 'batch_size', 1)), 1)
    gradient_accumulation_steps = max(int(getattr(config, 'gradient_accumulation_steps', 1)), 1)
    learning_rate = float(getattr(config, 'learning_rate', 1e-4))
    resolution = int(getattr(config, 'resolution', 1024))
    command.extend([
        '--pretrained_model_name_or_path', str(getattr(config, 'backbone_name')),
        '--train_data_dir', materialized['dataset_dir'],
        '--caption_column', 'text',
        '--image_column', 'image',
        '--output_dir', str((run_dir / 'official_output').resolve()),
        '--resolution', str(resolution),
        '--train_batch_size', str(train_batch_size),
        '--gradient_accumulation_steps', str(gradient_accumulation_steps),
        '--learning_rate', str(learning_rate),
        '--lr_scheduler', str(getattr(config, 'sdxl_lr_scheduler', 'constant')),
        '--lr_warmup_steps', str(int(getattr(config, 'sdxl_lr_warmup_steps', 0))),
        '--checkpointing_steps', str(max(int(getattr(config, 'save_every', 500)), 1)),
        '--validation_epochs', str(max(int(getattr(config, 'sdxl_validation_epochs', 1)), 1)),
        '--rank', str(int(getattr(config, 'adapter_plan').rank)),
        '--seed', str(int(getattr(config, 'seed', 42))),
        '--report_to', str(getattr(config, 'sdxl_report_to', 'none')),
    ])
    if getattr(config, 'max_steps', None) is not None:
        command.extend(['--max_train_steps', str(int(getattr(config, 'max_steps')))])
    else:
        command.extend(['--num_train_epochs', str(max(int(getattr(config, 'epochs', 1)), 1))])
    if bool(getattr(config, 'sdxl_use_8bit_adam', False)):
        command.append('--use_8bit_adam')
    if bool(getattr(config, 'sdxl_enable_xformers', False)):
        command.append('--enable_xformers_memory_efficient_attention')
    if bool(getattr(config, 'sdxl_gradient_checkpointing', True)):
        command.append('--gradient_checkpointing')
    if bool(getattr(config, 'sdxl_train_text_encoder', False)):
        command.append('--train_text_encoder')
    mixed_precision = str(getattr(config, 'sdxl_mixed_precision', 'fp16') or 'fp16').strip()
    if mixed_precision and mixed_precision.lower() != 'no':
        command.extend(['--mixed_precision', mixed_precision])
    dataloader_workers = int(getattr(config, 'num_workers', 0))
    if dataloader_workers > 0:
        command.extend(['--dataloader_num_workers', str(dataloader_workers)])
    if getattr(config, 'sdxl_caption_dropout_probability', None) is not None:
        command.extend(['--caption_dropout_probability', str(float(getattr(config, 'sdxl_caption_dropout_probability')))])
    if getattr(config, 'sdxl_noise_offset', None) is not None:
        command.extend(['--noise_offset', str(float(getattr(config, 'sdxl_noise_offset')))])
    validation_prompt = str(getattr(config, 'sdxl_validation_prompt', '') or '').strip()
    if validation_prompt:
        command.extend(['--validation_prompt', validation_prompt])
    extra_args = list(getattr(config, 'sdxl_extra_args', []) or [])
    command.extend(extra_args)
    return command


def run_stage2_sdxl_official_training(*, config: Any, pairs: list[Any], run_dir: Path, manifest_path: str) -> dict[str, Any]:
    """Materialize SDXL training data and launch the official diffusers SDXL LoRA trainer."""
    materialized = materialize_sdxl_training_dataset(pairs=pairs, output_dir=run_dir / 'sdxl_materialized_dataset')
    script_path = _resolve_official_sdxl_script(config)
    command = _build_sdxl_official_command(config=config, script_path=script_path, materialized=materialized, run_dir=run_dir)
    launch_plan = {
        'family': 'sdxl',
        'mode': 'official_diffusers_lora_wrapper',
        'script_path': script_path,
        'manifest_path': str(Path(manifest_path).resolve()),
        'materialized_dataset': materialized,
        'command': command,
        'command_string': subprocess.list2cmdline(command),
        'output_dir': str((run_dir / 'official_output').resolve()),
        'assumptions': [
            'The official diffusers SDXL LoRA training script is available either on PATH, via --sdxl-official-script, or via CSPD_STAGE2_SDXL_SCRIPT.',
            'This Stage 2 path delegates SDXL training internals to diffusers and only owns pairing, materialization, run structure, and launch orchestration.',
        ],
    }
    write_json(run_dir / 'sdxl_official_launch_plan.json', launch_plan)

    if bool(getattr(config, 'dry_run', False)) or bool(getattr(config, 'generate_manifest_only', False)):
        return {
            'status': 'sdxl_launch_prepared_only',
            'implemented_training': False,
            'placeholder_training': False,
            'family': 'sdxl',
            'message': 'Prepared SDXL materialized dataset and official diffusers launch plan without executing training.',
            'materialized_dataset': materialized,
            'official_launch_plan_path': str((run_dir / 'sdxl_official_launch_plan.json').resolve()),
            'launch_command': command,
            'last_known_phase': 'sdxl_launch_plan_ready',
        }

    completed = subprocess.run(command, cwd=str(run_dir), check=False, capture_output=True, text=True)
    stdout_path = run_dir / 'sdxl_official_stdout.txt'
    stderr_path = run_dir / 'sdxl_official_stderr.txt'
    stdout_path.write_text(completed.stdout or '', encoding='utf-8')
    stderr_path.write_text(completed.stderr or '', encoding='utf-8')
    return {
        'status': 'completed' if completed.returncode == 0 else 'failed',
        'implemented_training': completed.returncode == 0,
        'placeholder_training': False,
        'family': 'sdxl',
        'message': 'Official diffusers SDXL LoRA training completed.' if completed.returncode == 0 else 'Official diffusers SDXL LoRA training exited with a non-zero status.',
        'materialized_dataset': materialized,
        'official_launch_plan_path': str((run_dir / 'sdxl_official_launch_plan.json').resolve()),
        'launch_command': command,
        'returncode': int(completed.returncode),
        'stdout_path': str(stdout_path.resolve()),
        'stderr_path': str(stderr_path.resolve()),
        'last_known_phase': 'sdxl_training_finished' if completed.returncode == 0 else 'sdxl_training_failed',
    }
