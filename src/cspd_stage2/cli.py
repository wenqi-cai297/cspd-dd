from __future__ import annotations

import argparse
import json

from cspd_stage2.pipeline import config_from_args, run_stage2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CSPD Stage 2 canonical semantic rendering")
    subparsers = parser.add_subparsers(dest="command", required=True)

    render_parser = subparsers.add_parser("render", help="Render normalized Stage 1 records into canonical captions")
    render_parser.add_argument("--input", required=True, help="Path to attributes_normalized.jsonl")
    render_parser.add_argument("--output-dir", required=True, help="Directory for Stage 2 render artifacts")
    render_parser.add_argument("--renderer-version", default="v1", help="Renderer version label")
    render_parser.add_argument("--flush-every", type=int, default=100, help="Flush partial render outputs every N rows")
    render_parser.add_argument("--fallback-anchor-token", default=None, help="Fallback anchor token used when anchor slot is missing")
    render_parser.add_argument("--fallback-to-raw", action="store_true", help="Use raw attributes when normalized_attributes is missing")
    render_parser.add_argument("--fail-on-missing-anchor", action="store_true", help="Fail rows whose anchor slot cannot be rendered")
    render_parser.add_argument("--no-resume", action="store_true", help="Disable resume and overwrite prior Stage 2 outputs")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "render":
        summary = run_stage2(config_from_args(args))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
