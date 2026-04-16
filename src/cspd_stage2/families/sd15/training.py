"""SD v1.5 Stage 2 orchestration around the official diffusers trainer.

Uses `train_text_to_image.py` from diffusers for full fine-tuning (not LoRA).
SD v1.5 UNet is only ~860M params, making full fine-tuning feasible and
providing better distribution capture than LoRA.

Shares dataset materialization with the SDXL wrapper since the metadata.jsonl
format is identical.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from cspd_stage1.io_utils import write_json
from cspd_stage2.families.sdxl.training import materialize_sdxl_training_dataset


# Reuse the SDXL materialization — format is identical for SD v1.5
materialize_sd15_training_dataset = materialize_sdxl_training_dataset


def _candidate_sd15_script_paths(config: Any) -> list[str]:
    """Candidate paths for the official diffusers SD v1.5 full fine-tuning script."""
    candidates: list[str] = []
    explicit = str(getattr(config, 'sd15_official_script', '') or '').strip()
    env_value = str(os.environ.get('CSPD_STAGE2_SD15_SCRIPT', '')).strip()
    diffusers_roots = [
        str(os.environ.get('DIFFUSERS_REPO_ROOT', '')).strip(),
        str(os.environ.get('DIFFUSERS_HOME', '')).strip(),
    ]

    for value in [explicit, env_value]:
        if value:
            candidates.append(value)

    for root in diffusers_roots:
        if root:
            candidates.append(str(Path(root).expanduser() / 'examples' / 'text_to_image' / 'train_text_to_image.py'))

    candidates.append('train_text_to_image.py')
    return candidates


def _resolve_sd15_script(config: Any) -> str:
    """Resolve the official diffusers SD v1.5 full fine-tuning script path."""
    for candidate in _candidate_sd15_script_paths(config):
        candidate_path = Path(candidate).expanduser()
        if candidate_path.exists():
            return str(candidate_path.resolve())
        resolved_on_path = shutil.which(candidate)
        if resolved_on_path:
            return str(Path(resolved_on_path).resolve())
    return 'train_text_to_image_lora.py'


def _build_accelerate_prefix(config: Any) -> list[str]:
    """Build accelerate launch prefix."""
    if not bool(getattr(config, 'use_accelerate', True)):
        return [sys.executable]
    prefix = ['accelerate', 'launch']
    num_processes = getattr(config, 'sdxl_num_processes', None)
    if num_processes is not None and int(num_processes) > 0:
        prefix.extend(['--num_processes', str(int(num_processes))])
    extra_args = list(getattr(config, 'sdxl_accelerate_extra_args', []) or [])
    prefix.extend(extra_args)
    return prefix


def _build_sd15_command(*, config: Any, script_path: str, materialized: dict[str, Any], run_dir: Path) -> list[str]:
    """Translate Stage 2 config into the official diffusers SD v1.5 LoRA CLI."""
    command = _build_accelerate_prefix(config)
    command.append(script_path)

    train_batch_size = max(int(getattr(config, 'batch_size', 1)), 1)
    gradient_accumulation_steps = max(int(getattr(config, 'gradient_accumulation_steps', 1)), 1)
    learning_rate = float(getattr(config, 'learning_rate', 1e-4))
    resolution = int(getattr(config, 'resolution', 512))

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
        '--lr_scheduler', str(getattr(config, 'sdxl_lr_scheduler', 'cosine')),
        '--lr_warmup_steps', str(int(getattr(config, 'sdxl_lr_warmup_steps', 500))),
        '--checkpointing_steps', str(max(int(getattr(config, 'save_every', 500)), 1)),
        '--seed', str(int(getattr(config, 'seed', 42))),
    ])

    # report_to
    report_to = str(getattr(config, 'sdxl_report_to', 'none') or 'none').strip().lower()
    if report_to and report_to != 'none':
        command.extend(['--report_to', report_to])

    # epochs or max steps
    if getattr(config, 'max_steps', None) is not None:
        command.extend(['--max_train_steps', str(int(getattr(config, 'max_steps')))])
    else:
        command.extend(['--num_train_epochs', str(max(int(getattr(config, 'epochs', 1)), 1))])

    # Optional flags
    if bool(getattr(config, 'sdxl_use_8bit_adam', False)):
        command.append('--use_8bit_adam')
    if bool(getattr(config, 'sdxl_enable_xformers', False)):
        command.append('--enable_xformers_memory_efficient_attention')
    if bool(getattr(config, 'sdxl_gradient_checkpointing', True)):
        command.append('--gradient_checkpointing')

    mixed_precision = str(getattr(config, 'sdxl_mixed_precision', 'fp16') or 'fp16').strip()
    if mixed_precision and mixed_precision.lower() != 'no':
        command.extend(['--mixed_precision', mixed_precision])

    dataloader_workers = int(getattr(config, 'num_workers', 0))
    if dataloader_workers > 0:
        command.extend(['--dataloader_num_workers', str(dataloader_workers)])

    if getattr(config, 'sdxl_noise_offset', None) is not None:
        command.extend(['--noise_offset', str(float(getattr(config, 'sdxl_noise_offset')))])
    if getattr(config, 'sdxl_snr_gamma', None) is not None:
        command.extend(['--snr_gamma', str(float(getattr(config, 'sdxl_snr_gamma')))])

    validation_prompt = str(getattr(config, 'sdxl_validation_prompt', '') or '').strip()
    if validation_prompt:
        command.extend(['--validation_prompt', validation_prompt])

    extra_args = list(getattr(config, 'sdxl_extra_args', []) or [])
    command.extend(extra_args)

    return command


def run_stage2_sd15_official_training(*, config: Any, pairs: list[Any], run_dir: Path, manifest_path: str) -> dict[str, Any]:
    """Materialize SD v1.5 training data and launch the official diffusers LoRA trainer."""
    materialized = materialize_sd15_training_dataset(pairs=pairs, output_dir=run_dir / 'sd15_materialized_dataset')
    script_path = _resolve_sd15_script(config)
    command = _build_sd15_command(config=config, script_path=script_path, materialized=materialized, run_dir=run_dir)

    launch_plan = {
        'family': 'sd15',
        'mode': 'official_diffusers_lora_wrapper',
        'script_path': script_path,
        'command': command,
        'command_string': subprocess.list2cmdline(command),
        'output_dir': str((run_dir / 'official_output').resolve()),
    }
    write_json(run_dir / 'sd15_official_launch_plan.json', launch_plan)

    # Preflight check
    script_exists = Path(script_path).expanduser().exists() or shutil.which(script_path) is not None
    if not script_exists:
        return {
            'status': 'failed_before_training',
            'message': f'SD v1.5 LoRA training script not found: {script_path}. '
                       'Set DIFFUSERS_REPO_ROOT or CSPD_STAGE2_SD15_SCRIPT.',
        }

    if bool(getattr(config, 'dry_run', False)):
        return {
            'status': 'dry_run',
            'command': command,
            'launch_plan_path': str((run_dir / 'sd15_official_launch_plan.json').resolve()),
        }

    # Run training
    print(f"[Stage 2] Launching SD v1.5 LoRA training...")
    print(f"[Stage 2] Command: {subprocess.list2cmdline(command)}")

    stdout_path = run_dir / 'sd15_official_stdout.txt'
    stderr_path = run_dir / 'sd15_official_stderr.txt'

    with open(stdout_path, 'w') as stdout_f, open(stderr_path, 'w') as stderr_f:
        proc = subprocess.run(
            command,
            stdout=stdout_f,
            stderr=stderr_f,
            text=True,
        )

    if proc.returncode != 0:
        stderr_content = stderr_path.read_text(encoding='utf-8', errors='replace')[-2000:]
        return {
            'status': 'training_failed',
            'returncode': proc.returncode,
            'stderr_tail': stderr_content,
        }

    return {
        'status': 'completed',
        'output_dir': str((run_dir / 'official_output').resolve()),
        'launch_plan_path': str((run_dir / 'sd15_official_launch_plan.json').resolve()),
    }
