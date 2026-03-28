from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_class_dirs(dataset_root: Path) -> list[Path]:
    return sorted([path for path in dataset_root.iterdir() if path.is_dir()], key=lambda p: p.name)


def iter_images(class_dir: Path) -> list[Path]:
    return sorted(
        [path for path in class_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda p: str(p.relative_to(class_dir)).replace("\\", "/"),
    )


def copy_subset(dataset_root: Path, output_root: Path, max_per_class: int, flatten: bool) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "dataset_root": str(dataset_root.resolve()),
        "output_root": str(output_root.resolve()),
        "max_per_class": max_per_class,
        "num_classes": 0,
        "copied_images": 0,
        "classes": {},
    }

    for class_dir in iter_class_dirs(dataset_root):
        images = iter_images(class_dir)
        selected = images[:max_per_class]
        target_class_dir = output_root / class_dir.name
        target_class_dir.mkdir(parents=True, exist_ok=True)

        for idx, src_path in enumerate(selected, start=1):
            if flatten:
                extension = src_path.suffix.lower()
                dst_path = target_class_dir / f"{class_dir.name}_{idx:04d}{extension}"
            else:
                rel_path = src_path.relative_to(class_dir)
                dst_path = target_class_dir / rel_path
                dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path)

        summary["num_classes"] += 1
        summary["copied_images"] += len(selected)
        summary["classes"][class_dir.name] = {
            "available_images": len(images),
            "copied_images": len(selected),
        }

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a small ImageFolder subset with at most N images per class.")
    parser.add_argument("--input", required=True, help="Path to the source ImageFolder dataset root")
    parser.add_argument("--output", required=True, help="Path to the output subset root")
    parser.add_argument("--max-per-class", type=int, default=20, help="Maximum number of images to copy per class")
    parser.add_argument(
        "--flatten",
        action="store_true",
        help="Flatten nested files inside each class into the class root with deterministic renamed filenames",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dataset_root = Path(args.input)
    output_root = Path(args.output)
    if not dataset_root.exists():
        raise SystemExit(f"Input dataset root does not exist: {dataset_root}")
    if not dataset_root.is_dir():
        raise SystemExit(f"Input dataset root is not a directory: {dataset_root}")
    if args.max_per_class <= 0:
        raise SystemExit("--max-per-class must be > 0")

    summary = copy_subset(
        dataset_root=dataset_root,
        output_root=output_root,
        max_per_class=args.max_per_class,
        flatten=bool(args.flatten),
    )
    summary_path = output_root / "subset_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
