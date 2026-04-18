"""Stage 3A — Encode images to DINOv2 features for clustering.

Only DINOv2 CLS features are needed: Stage 3B clusters on them and Stage 4
text2img consumes representative captions as plain strings. SDXL VAE encoding
was removed 2026-04-18 along with the set-level selection code that used it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm.auto import tqdm

from cspd_stage1.io_utils import write_json


@dataclass(slots=True)
class EncodeResult:
    """Result of encoding a dataset split."""

    dino_embeds_path: str      # .pt file with stacked DINOv2 features
    index_path: str            # .json mapping index → record metadata
    num_samples: int
    dino_embed_dim: int


def _load_image(path: str | Path, resolution: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img = img.resize((resolution, resolution), Image.LANCZOS)
    return img


def _load_render_records(render_input: str | Path) -> dict[str, dict[str, Any]]:
    """Load Stage 1C render records.jsonl, keyed by record_id."""
    records = {}
    with open(render_input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rid = row.get("record_id", "")
            if rid:
                records[rid] = row
    return records


def _load_pairs_index(render_input: str | Path, dataset_root: str | Path) -> list[dict[str, Any]]:
    """Build the (image_path, caption, class, archetype, record_id) list from render records.

    Reads Stage 1C render records.jsonl. Each record has a record_id of the
    form ``class_name_raw::relative_path``, which is used to reconstruct the
    full image path under dataset_root.
    """
    dataset_root = Path(dataset_root)
    render_records = _load_render_records(render_input)

    pairs = []
    for record_id, row in render_records.items():
        if row.get("render_status") != "success":
            continue
        parts = record_id.split("::", 1)
        if len(parts) != 2:
            continue
        class_name_raw = parts[0]
        relative_path = parts[1]
        image_path = dataset_root / relative_path
        if not image_path.exists():
            sample_id = row.get("sample_id", "")
            if sample_id:
                image_path = dataset_root / sample_id
        pairs.append({
            "index": len(pairs),
            "record_id": record_id,
            "image_path": str(image_path),
            "canonical_caption": row.get("canonical_caption", ""),
            "class_name": row.get("class_name", class_name_raw),
            "class_name_raw": class_name_raw,
            "class_id": int(row.get("class_id", 0) or 0),
            "archetype": row.get("archetype", ""),
        })
    pairs.sort(key=lambda p: p["record_id"])
    for i, p in enumerate(pairs):
        p["index"] = i
    return pairs


@torch.no_grad()
def encode_dataset(
    *,
    dataset_root: str | Path,
    render_input: str | Path,
    output_dir: str | Path,
    resolution: int = 512,
    batch_size: int = 8,
    device: str = "cuda",
) -> EncodeResult:
    """Encode images to DINOv2 CLS features for Stage 3B clustering.

    Args:
        dataset_root: ImageFolder dataset root (same as Stage 2).
        render_input: Path to Stage 1C records.jsonl.
        output_dir: Directory for output .pt and index files.
        resolution: Image resolution for loading.
        batch_size: Encoding batch size.
        device: Torch device.

    Returns:
        EncodeResult with paths to features and metadata.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[Stage 3A] Building pairs index...")
    pairs = _load_pairs_index(render_input, dataset_root)
    if not pairs:
        raise ValueError("No pairs found. Check dataset_root and render_input paths.")
    print(f"[Stage 3A] Found {len(pairs)} paired samples")

    print("[Stage 3A] Loading DINOv2 (dinov2_vitb14)...")
    dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    dino_model = dino_model.to(device)
    dino_model.eval()

    from torchvision import transforms as T
    dino_transform = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print(f"[Stage 3A] Encoding {len(pairs)} images to DINOv2 features...")
    all_dino_embeds = []
    for i in tqdm(range(0, len(pairs), batch_size), desc="DINOv2 encode"):
        batch_pairs = pairs[i : i + batch_size]
        images = [_load_image(p["image_path"], resolution) for p in batch_pairs]
        pixel_values = torch.stack([dino_transform(img) for img in images]).to(device)
        features = dino_model(pixel_values)  # (B, 768) CLS token
        all_dino_embeds.append(features.cpu().float())

    all_dino_embeds = torch.cat(all_dino_embeds, dim=0)  # (N, 768)
    print(f"[Stage 3A] DINOv2 embedding shape: {list(all_dino_embeds.shape)}")

    del dino_model
    torch.cuda.empty_cache()

    dino_embeds_path = output_dir / "dino_embeds.pt"
    index_path = output_dir / "encode_index.json"
    torch.save(all_dino_embeds, dino_embeds_path)

    index_data = {
        "num_samples": len(pairs),
        "encoding": {
            "feature_extractor": "DINOv2 (dinov2_vitb14)",
            "dino_model": "facebookresearch/dinov2:dinov2_vitb14",
            "dino_input_resolution": 224,
            "dino_normalization": "ImageNet (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])",
            "dino_output_dim": int(all_dino_embeds.shape[1]),
            "dino_embed_shape": list(all_dino_embeds.shape),
            "image_load_resolution": resolution,
        },
        "source": {
            "dataset_root": str(Path(dataset_root).resolve()),
            "render_input": str(Path(render_input).resolve()),
        },
        "samples": [{k: v for k, v in p.items()} for p in pairs],
    }
    write_json(index_path, index_data)

    print(f"[Stage 3A] Encoding complete. Saved to {output_dir}")
    return EncodeResult(
        dino_embeds_path=str(dino_embeds_path),
        index_path=str(index_path),
        num_samples=len(pairs),
        dino_embed_dim=all_dino_embeds.shape[-1],
    )
