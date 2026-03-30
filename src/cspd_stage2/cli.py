from __future__ import annotations

import json

from cspd_stage1.cli import build_parser
from cspd_stage2.pipeline import config_from_args, run_stage2


def main() -> None:
    parser = build_parser()
    args = parser.parse_args(["render", *(__import__("sys").argv[1:])])
    summary = run_stage2(config_from_args(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
