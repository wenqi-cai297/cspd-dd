from pathlib import Path

from cspd_stage2.families.sdxl.training import _build_sdxl_launch_preflight, _resolve_official_sdxl_script
from cspd_stage2.training import Stage2TrainConfig


def test_resolve_official_sdxl_script_prefers_explicit_existing_file(tmp_path: Path) -> None:
    script_path = tmp_path / 'train_text_to_image_lora_sdxl.py'
    script_path.write_text('#!/usr/bin/env python\n', encoding='utf-8')
    config = Stage2TrainConfig(dataset_root='data', render_input='records.jsonl', sdxl_official_script=str(script_path))

    assert _resolve_official_sdxl_script(config) == str(script_path.resolve())


def test_build_sdxl_launch_preflight_reports_missing_script(tmp_path: Path) -> None:
    dataset_dir = tmp_path / 'materialized'
    dataset_dir.mkdir(parents=True)
    metadata_path = dataset_dir / 'metadata.jsonl'
    metadata_path.write_text('{"file_name":"images/x.png","text":"caption"}\n', encoding='utf-8')
    config = Stage2TrainConfig(dataset_root='data', render_input='records.jsonl', sdxl_official_script=str(tmp_path / 'missing.py'))

    summary = _build_sdxl_launch_preflight(
        config=config,
        script_path=str(tmp_path / 'missing.py'),
        materialized={
            'dataset_dir': str(dataset_dir),
            'metadata_path': str(metadata_path),
            'num_examples': 1,
        },
    )

    assert summary['ok'] is False
    assert any('Could not resolve the official diffusers SDXL LoRA script' in message for message in summary['errors'])
