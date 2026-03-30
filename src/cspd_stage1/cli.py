from __future__ import annotations

"""Command-line entrypoint for CSPD Stage 1."""

import argparse
import json

from cspd_stage1.pipeline import config_from_args, run_stage1
from cspd_stage2.pipeline import config_from_args as render_config_from_args, run_stage2


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

    render_parser = subparsers.add_parser("render", help="Run Stage 1 canonical rendering from normalized attributes")
    render_parser.add_argument("--input", required=True, help="Path to attributes_normalized.jsonl")
    render_parser.add_argument("--output-dir", required=True, help="Directory for Stage 1 render artifacts")
    render_parser.add_argument("--renderer-version", default="v1", help="Renderer version label")
    render_parser.add_argument("--flush-every", type=int, default=100, help="Flush partial render outputs every N rows")
    render_parser.add_argument("--fallback-anchor-token", default=None, help="Fallback anchor token used when anchor slot is missing")
    render_parser.add_argument("--fallback-to-raw", action="store_true", help="Use raw attributes when normalized_attributes is missing")
    render_parser.add_argument("--fail-on-missing-anchor", action="store_true", help="Fail rows whose anchor slot cannot be rendered")
    render_parser.add_argument("--no-resume", action="store_true", help="Disable resume and overwrite prior render outputs")
    return parser


def main() -> None:
    """Parse CLI args and dispatch the requested command."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        stats = run_stage1(config_from_args(args))
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return

    if args.command == "render":
        summary = run_stage2(render_config_from_args(args))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
