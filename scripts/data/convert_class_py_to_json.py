"""Convert a Python class-mapping file into a JSON mapping file.

Supported input pattern:
- a Python file that defines one or more top-level variables containing a dict
  or OrderedDict mapping class ids to readable class names.

Typical example:
    IMAGENET2012_CLASSES = OrderedDict({
        "n01440764": "tench, Tinca tinca",
        "n01443537": "goldfish, Carassius auratus",
    })

Usage:
    python scripts/data/convert_class_py_to_json.py \
        --input /path/to/class.py \
        --output /path/to/classes.json

Optional:
    python scripts/data/convert_class_py_to_json.py \
        --input /path/to/class.py \
        --output /path/to/classes.json \
        --var-name IMAGENET2012_CLASSES
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from types import ModuleType
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    """Build CLI arguments for the conversion script."""
    parser = argparse.ArgumentParser(description="Convert a Python class mapping file to JSON.")
    parser.add_argument("--input", required=True, help="Path to the source Python file")
    parser.add_argument("--output", required=True, help="Path to the output JSON file")
    parser.add_argument(
        "--var-name",
        default=None,
        help="Optional variable name to extract. If omitted, the script auto-detects a mapping variable.",
    )
    return parser


def load_python_namespace(file_path: str) -> dict[str, Any]:
    """Execute the source Python file in a restricted namespace and return globals.

    This is acceptable here because the mapping file is expected to be a small,
    user-controlled config-like Python file rather than untrusted code.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    source = path.read_text(encoding="utf-8")
    namespace: dict[str, Any] = {
        "OrderedDict": OrderedDict,
        "__builtins__": __builtins__,
        "__name__": "__class_mapping_loader__",
        "__file__": str(path),
    }
    exec(compile(source, str(path), "exec"), namespace)  # noqa: S102
    return namespace


def pick_mapping(namespace: dict[str, Any], var_name: str | None) -> dict[str, str]:
    """Pick the target mapping from the executed namespace.

    If `var_name` is provided, we require that exact variable. Otherwise we
    auto-detect the first top-level dict-like object whose keys and values are
    all strings.
    """
    if var_name is not None:
        if var_name not in namespace:
            raise KeyError(f"Variable '{var_name}' not found in source file")
        candidate = namespace[var_name]
        return normalize_mapping(candidate, var_name)

    candidates: list[tuple[str, dict[str, str]]] = []
    for name, value in namespace.items():
        if name.startswith("__"):
            continue
        if isinstance(value, (dict, OrderedDict)):
            try:
                normalized = normalize_mapping(value, name)
                candidates.append((name, normalized))
            except ValueError:
                continue

    if not candidates:
        raise ValueError("No suitable string-to-string mapping variable found in source file")
    if len(candidates) > 1:
        names = ", ".join(name for name, _ in candidates)
        raise ValueError(
            "Multiple mapping variables found. Please specify one with --var-name. "
            f"Candidates: {names}"
        )
    return candidates[0][1]


def normalize_mapping(value: Any, name: str) -> dict[str, str]:
    """Validate and normalize a candidate mapping into a plain dict[str, str]."""
    if not isinstance(value, (dict, OrderedDict)):
        raise ValueError(f"Variable '{name}' is not a dict-like mapping")

    normalized: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise ValueError(f"Variable '{name}' is not a string-to-string mapping")
        normalized[key] = item
    return normalized


def write_json(mapping: dict[str, str], output_path: str) -> None:
    """Write the final mapping as pretty JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    """CLI entrypoint."""
    args = build_parser().parse_args()
    namespace = load_python_namespace(args.input)
    mapping = pick_mapping(namespace, args.var_name)
    write_json(mapping, args.output)
    print(f"[OK] Wrote {len(mapping)} entries to {args.output}")


if __name__ == "__main__":
    main()
