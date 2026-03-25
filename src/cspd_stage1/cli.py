from __future__ import annotations

"""Command-line entrypoint for CSPD Stage 1."""

import argparse
import json

from cspd_stage1.pipeline import config_from_args, run_stage1


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser.

    We use a subcommand structure even though there is only `run` today,
    because future stages / utilities will probably want to live under the same
    executable without turning the argument surface into soup.
    """
    parser = argparse.ArgumentParser(description="CSPD Stage 1 attribute extraction")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run Stage 1 attribute extraction")
    run_parser.add_argument("--input", required=True, help="Path to input JSONL")
    run_parser.add_argument("--output-dir", required=True, help="Directory for output artifacts")
    run_parser.add_argument("--backend", default="mock", help="VLM backend name")
    run_parser.add_argument("--max-retries", type=int, default=2, help="Retries after initial attempt")
    run_parser.add_argument("--no-raw-response", action="store_true", help="Do not persist raw backend text")
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
