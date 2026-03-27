from __future__ import annotations

"""Analyze Stage 1 attribute JSONL outputs for normalization work.

This script summarizes slot/value distributions from `attributes.jsonl` so we can
inspect high-frequency values, unknown rates, and potentially noisy values
without manually reading the full file.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize slot/value distributions from a Stage 1 attributes.jsonl file."
    )
    parser.add_argument("--input", required=True, help="Path to attributes.jsonl")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON path for the full summary report. Defaults to <input>_slot_value_summary.json",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="How many top values to keep per archetype/slot in the report.",
    )
    parser.add_argument(
        "--print-top-k",
        type=int,
        default=10,
        help="How many top values to print per archetype/slot to stdout.",
    )
    return parser


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield line_number, json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc


def canonicalize_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, str):
        cleaned = " ".join(value.strip().split())
        return cleaned if cleaned else "unknown"
    return str(value)


def build_summary(path: Path, top_k: int) -> dict[str, Any]:
    total_rows = 0
    total_success_rows = 0
    archetype_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    slot_presence_by_archetype: dict[str, Counter[str]] = defaultdict(Counter)
    slot_unknown_by_archetype: dict[str, Counter[str]] = defaultdict(Counter)
    slot_value_counts_by_archetype: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))

    for _, row in iter_jsonl(path):
        total_rows += 1
        if row.get("extraction_status") != "success":
            continue
        total_success_rows += 1

        archetype = str(row.get("archetype", "unknown_archetype"))
        class_name = str(row.get("class_name_raw") or row.get("class_name") or "unknown_class")
        archetype_counts[archetype] += 1
        class_counts[class_name] += 1

        attributes = row.get("attributes", {})
        if not isinstance(attributes, dict):
            continue

        for slot, raw_value in attributes.items():
            slot_presence_by_archetype[archetype][slot] += 1
            value = canonicalize_value(raw_value)
            if value.lower() in {"unknown", "not_applicable", "null", "none", "n/a"}:
                slot_unknown_by_archetype[archetype][slot] += 1
            slot_value_counts_by_archetype[archetype][slot][value] += 1

    archetypes: dict[str, Any] = {}
    for archetype in sorted(slot_value_counts_by_archetype):
        slots: dict[str, Any] = {}
        for slot in sorted(slot_value_counts_by_archetype[archetype]):
            counter = slot_value_counts_by_archetype[archetype][slot]
            total_slot_values = slot_presence_by_archetype[archetype][slot]
            unknown_count = slot_unknown_by_archetype[archetype][slot]
            slots[slot] = {
                "num_values": total_slot_values,
                "num_unique_values": len(counter),
                "num_unknown_like_values": unknown_count,
                "unknown_like_ratio": (unknown_count / total_slot_values) if total_slot_values else 0.0,
                "top_values": [
                    {"value": value, "count": count}
                    for value, count in counter.most_common(top_k)
                ],
            }
        archetypes[archetype] = {
            "num_samples": archetype_counts[archetype],
            "slots": slots,
        }

    return {
        "input_path": str(path.resolve()),
        "num_rows_total": total_rows,
        "num_success_rows": total_success_rows,
        "num_archetypes": len(archetypes),
        "archetype_counts": dict(archetype_counts.most_common()),
        "class_counts": dict(class_counts.most_common()),
        "archetypes": archetypes,
    }


def print_summary(summary: dict[str, Any], print_top_k: int) -> None:
    print(f"[summary] input={summary['input_path']}")
    print(
        f"[summary] rows_total={summary['num_rows_total']} success_rows={summary['num_success_rows']} archetypes={summary['num_archetypes']}"
    )

    for archetype, archetype_data in summary["archetypes"].items():
        print()
        print(f"[archetype] {archetype} | samples={archetype_data['num_samples']}")
        for slot, slot_data in archetype_data["slots"].items():
            print(
                f"  [slot] {slot} | values={slot_data['num_values']} | unique={slot_data['num_unique_values']} | unknown_like={slot_data['num_unknown_like_values']} ({slot_data['unknown_like_ratio']:.1%})"
            )
            for item in slot_data["top_values"][: max(print_top_k, 0)]:
                print(f"    - {item['value']}: {item['count']}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file does not exist: {input_path}")

    output_path = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}_slot_value_summary.json")
    summary = build_summary(input_path, top_k=max(args.top_k, 1))
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print_summary(summary, print_top_k=max(args.print_top_k, 0))
    print()
    print(f"[ok] Wrote summary report to: {output_path}")


if __name__ == "__main__":
    main()
