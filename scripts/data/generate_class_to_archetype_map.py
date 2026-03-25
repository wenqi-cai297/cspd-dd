"""Generate a raw-class-label -> archetype mapping from a classes.json file.

Input format:
    {
      "n01440764": "tench, Tinca tinca",
      ...
    }

Output format:
    {
      "n01440764": "animal",
      ...
    }

The current implementation uses the Stage 1 `infer_archetype` heuristic over
readable class names. This is useful for bootstrapping a frozen mapping file
that can later be manually reviewed and edited.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from cspd_stage1.schema import infer_archetype


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate class->archetype mapping from classes.json")
    parser.add_argument("--input", required=True, help="Path to classes.json")
    parser.add_argument("--output", required=True, help="Path to output mapping JSON")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    classes = json.loads(input_path.read_text(encoding="utf-8-sig"))
    if not isinstance(classes, dict):
        raise ValueError("Input classes file must be a JSON object")

    mapping = {str(raw_label): infer_archetype(str(readable_name)) for raw_label, readable_name in classes.items()}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")

    counts = Counter(mapping.values())
    print(f"[OK] Wrote {len(mapping)} entries to {output_path}")
    print("[INFO] Archetype counts:")
    for archetype, count in sorted(counts.items()):
        print(f"  - {archetype}: {count}")


if __name__ == "__main__":
    main()
