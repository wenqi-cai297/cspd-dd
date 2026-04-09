from __future__ import annotations

from cspd_stage2.cli import build_parser
from cspd_stage2.training import Stage2TrainConfig, derive_stage2_baseline_sample_output_dir


def test_stage2_train_config_pixart_sampling_defaults_are_eval_like() -> None:
    config = Stage2TrainConfig(dataset_root="/tmp/dataset", render_input="/tmp/render.jsonl")
    assert config.sample_num_inference_steps == 50
    assert config.sample_guidance_scale == 7.0


def test_stage2_cli_train_sampling_defaults_match_config() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "train",
            "--dataset-root",
            "/tmp/dataset",
            "--render-input",
            "/tmp/render.jsonl",
        ]
    )
    assert args.sample_num_inference_steps == 50
    assert args.sample_guidance_scale == 7.0


def test_stage2_cli_baseline_sampling_defaults_and_output_root() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "sample-baseline",
            "--dataset-root",
            "/tmp/ImageNette/train",
        ]
    )
    assert args.sample_num_inference_steps == 50
    assert args.sample_guidance_scale == 7.0
    output_dir = derive_stage2_baseline_sample_output_dir(args.dataset_root, args.backbone_name, timestamp="2026-04-09_160000")
    normalized = output_dir.replace("\\", "/")
    assert normalized == "runs/stage2/baseline_samples/ImageNette_train/PixArt-alpha_PixArt-Sigma-XL-2-512-MS/2026-04-09_160000"
