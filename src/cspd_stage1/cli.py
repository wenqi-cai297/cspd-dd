from __future__ import annotations

"""Command-line entrypoint for CSPD Stage 1."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from cspd_stage1.pipeline import config_from_args, run_stage1
from cspd_stage1.render_pipeline import config_from_args as render_config_from_args, run_stage1_render


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description="CSPD workflow CLI for Prep + Stage 1")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run Stage 1 attribute extraction")
    run_parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path to an ImageFolder-style dataset root. Each class must be a subdirectory.",
    )
    run_parser.add_argument("--output-dir", required=True, help="Directory for output artifacts")
    run_parser.add_argument("--backend", default="mock", help="VLM backend name")
    run_parser.add_argument("--max-retries", type=int, default=2, help="Retries after initial attempt")
    run_parser.add_argument("--no-raw-response", action="store_true", help="Do not persist raw backend text")
    run_parser.add_argument(
        "--model-name",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Model identifier used by local real backends such as qwen_local",
    )
    run_parser.add_argument(
        "--torch-dtype",
        default="float16",
        help="Torch dtype for local model loading: float16, bfloat16, or float32",
    )
    run_parser.add_argument(
        "--device-map",
        default="auto",
        help="Transformers device_map for local model loading",
    )
    run_parser.add_argument(
        "--disable-fast-processor",
        action="store_true",
        help="Use the slow processor implementation instead of the fast default",
    )
    run_parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Generation cap for local VLM backends",
    )
    run_parser.add_argument(
        "--class-name-map",
        default=None,
        help="Optional JSON file that maps raw folder labels (e.g. synset ids) to readable class names.",
    )
    run_parser.add_argument(
        "--class-archetype-map",
        default=None,
        help="Optional JSON file that maps raw folder labels to frozen archetypes.",
    )
    run_parser.add_argument(
        "--flush-every",
        type=int,
        default=10,
        help="Write partial JSONL results to disk every N samples.",
    )
    run_parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume and overwrite prior JSONL outputs in the output directory.",
    )

    normalize_parser = subparsers.add_parser("normalize", help="Run deterministic normalization with inline constrained VLM review by default")
    normalize_parser.add_argument("--input", required=True, help="Path to Stage 1 attributes.jsonl")
    normalize_parser.add_argument("--output-dir", required=True, help="Directory for normalization artifacts")
    normalize_parser.add_argument("--rules", default="configs/stage1/normalization/stage1_attribute_normalization_rules.json", help="Path to normalization rules JSON")
    normalize_parser.add_argument("--disable-vlm-review", action="store_true", help="Disable inline VLM review and keep outputs purely deterministic")
    normalize_parser.add_argument("--review-backend", default="qwen_local", help="Inline review backend; use mock for plumbing-only tests")
    normalize_parser.add_argument("--review-model-name", default="Qwen/Qwen2.5-VL-7B-Instruct", help="Model name for inline review")
    normalize_parser.add_argument("--review-torch-dtype", default="float16", help="Torch dtype for inline review")
    normalize_parser.add_argument("--review-device-map", default="auto", help="Device map for inline review")
    normalize_parser.add_argument("--disable-review-fast-processor", action="store_true", help="Use the slow processor for inline review")
    normalize_parser.add_argument("--review-max-new-tokens", type=int, default=256, help="Generation cap for inline review")
    normalize_parser.add_argument("--review-limit", type=int, default=None, help="Optional maximum number of ambiguous slots to review")

    render_parser = subparsers.add_parser("render", help="Run Stage 1 canonical rendering from normalized attributes")
    render_parser.add_argument("--input", required=True, help="Path to attributes_normalized.jsonl")
    render_parser.add_argument("--output-dir", required=True, help="Directory for Stage 1 render artifacts")
    render_parser.add_argument("--renderer-version", default="v1", help="Renderer version label")
    render_parser.add_argument("--flush-every", type=int, default=100, help="Flush partial render outputs every N rows")
    render_parser.add_argument("--fallback-anchor-token", default=None, help="Fallback anchor token used when anchor slot is missing")
    render_parser.add_argument("--fallback-to-raw", action="store_true", help="Use raw attributes when normalized_attributes is missing")
    render_parser.add_argument("--fail-on-missing-anchor", action="store_true", help="Fail rows whose anchor slot cannot be rendered")
    render_parser.add_argument("--no-resume", action="store_true", help="Disable resume and overwrite prior render outputs")

    enrich_parser = subparsers.add_parser("enrich", help="Enrich template captions with VLM visual details (Stage 1D)")
    enrich_parser.add_argument("--input", required=True, help="Path to Stage 1C records.jsonl")
    enrich_parser.add_argument("--dataset-root", required=True, help="ImageFolder dataset root (for loading images)")
    enrich_parser.add_argument("--output", required=True, help="Path for enriched output JSONL (e.g. records_enriched.jsonl)")
    enrich_parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct", help="VLM model name")
    enrich_parser.add_argument("--device", default="cuda", help="Torch device")
    enrich_parser.add_argument("--max-new-tokens", type=int, default=100, help="Max tokens for VLM generation")
    enrich_parser.add_argument("--no-resume", action="store_true", help="Disable resume, overwrite existing output")

    return parser


def main() -> None:
    """Parse CLI args and dispatch the requested command."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        stats = run_stage1(config_from_args(args))
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return

    if args.command == "normalize":
        script_path = Path(__file__).resolve().parents[2] / "scripts" / "data" / "normalize_stage1_attributes.py"
        cmd = [
            sys.executable,
            str(script_path),
            "--input",
            args.input,
            "--output-dir",
            args.output_dir,
            "--rules",
            args.rules,
            "--review-backend",
            args.review_backend,
            "--review-model-name",
            args.review_model_name,
            "--review-torch-dtype",
            args.review_torch_dtype,
            "--review-device-map",
            args.review_device_map,
            "--review-max-new-tokens",
            str(args.review_max_new_tokens),
        ]
        if args.disable_vlm_review:
            cmd.append("--disable-vlm-review")
        if args.disable_review_fast_processor:
            cmd.append("--disable-review-fast-processor")
        if args.review_limit is not None:
            cmd.extend(["--review-limit", str(args.review_limit)])
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        print(result.stdout.strip())
        return

    if args.command == "render":
        summary = run_stage1_render(render_config_from_args(args))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if args.command == "enrich":
        from cspd_stage1.enrich import enrich_captions

        result = enrich_captions(
            render_input=args.input,
            dataset_root=args.dataset_root,
            output_path=args.output,
            model_name=args.model_name,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            resume=not args.no_resume,
        )
        print(json.dumps({
            "output_path": result.output_path,
            "num_enriched": result.num_enriched,
            "num_skipped": result.num_skipped,
            "num_failed": result.num_failed,
        }, indent=2))
        return

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
