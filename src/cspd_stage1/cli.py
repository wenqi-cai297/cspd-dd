from __future__ import annotations

"""Command-line entrypoint for CSPD Stage 1."""

import argparse
import json

from cspd_stage1.pipeline import config_from_args, run_stage1


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description="CSPD Stage 1 attribute extraction")
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
    return parser


def main() -> None:
    """Parse CLI args and dispatch the requested command."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        stats = run_stage1(config_from_args(args))
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
