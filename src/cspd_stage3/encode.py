"""Stage 3A — Encode captions to text embeddings and images to DINOv2 features.

Text embeddings are encoded via SDXL's dual CLIP text encoders (loaded standalone,
without the full pipeline/VAE/UNet). DINOv2 CLS features are used for clustering.

VAE encoding is no longer performed since Stage 4 uses text2img generation,
which does not require VAE latents.
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

    text_embeds_path: str      # .pt file with stacked text embeddings
    pooled_embeds_path: str    # .pt file with stacked pooled text embeddings
    dino_embeds_path: str      # .pt file with stacked DINOv2 features
    index_path: str            # .json mapping index → record metadata
    num_samples: int
    text_embed_dim: int
    pooled_embed_dim: int
    dino_embed_dim: int


def _load_image(path: str | Path, resolution: int) -> Image.Image:
    """Load an image as PIL RGB at target resolution."""
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
    """Build a list of (image_path, caption, class, archetype, record_id) from render records.

    Reads directly from Stage 1C render records.jsonl. Each record has a record_id
    of the form class_name_raw::relative_path, which is used to reconstruct the
    full image path under dataset_root.
    """
    dataset_root = Path(dataset_root)
    render_records = _load_render_records(render_input)

    pairs = []
    for record_id, row in render_records.items():
        if row.get("render_status") != "success":
            continue
        # record_id = "class_name_raw::relative_path" e.g. "n01440764::n01440764/img.JPEG"
        parts = record_id.split("::", 1)
        if len(parts) != 2:
            continue
        class_name_raw = parts[0]
        relative_path = parts[1]
        image_path = dataset_root / relative_path
        if not image_path.exists():
            # Try sample_id as fallback
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
    # Sort by record_id for deterministic ordering
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
    model_name: str = "stabilityai/stable-diffusion-xl-base-1.0",
    resolution: int = 512,
    batch_size: int = 8,
    device: str = "cuda",
    dtype: str = "float16",
) -> EncodeResult:
    """Encode captions to text embeddings and images to DINOv2 features.

    VAE encoding is skipped — Stage 4 uses text2img and does not need VAE latents.
    Text encoders are loaded standalone (without UNet/VAE) to save memory.

    Args:
        dataset_root: ImageFolder dataset root (same as Stage 2).
        render_input: Path to Stage 1C records.jsonl.
        output_dir: Directory for output .pt and index files.
        model_name: SDXL model identifier (for text encoders).
        resolution: Image resolution for DINOv2 encoding.
        batch_size: Encoding batch size.
        device: Torch device.
        dtype: Weight dtype (float16 or bfloat16).

    Returns:
        EncodeResult with paths to saved tensors and metadata.
    """
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, AutoTokenizer

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16

    # Build pairs index
    print("[Stage 3A] Building pairs index...")
    pairs = _load_pairs_index(render_input, dataset_root)
    if not pairs:
        raise ValueError("No pairs found. Check dataset_root and render_input paths.")
    print(f"[Stage 3A] Found {len(pairs)} paired samples")

    # Load SDXL text encoders standalone (no VAE, no UNet — saves ~6GB VRAM)
    print(f"[Stage 3A] Loading SDXL text encoders from {model_name}...")
    tokenizer_1 = AutoTokenizer.from_pretrained(model_name, subfolder="tokenizer", use_fast=False)
    tokenizer_2 = AutoTokenizer.from_pretrained(model_name, subfolder="tokenizer_2", use_fast=False)
    text_encoder_1 = CLIPTextModel.from_pretrained(
        model_name, subfolder="text_encoder", torch_dtype=torch_dtype,
    ).to(device)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        model_name, subfolder="text_encoder_2", torch_dtype=torch_dtype,
    ).to(device)
    text_encoder_1.eval()
    text_encoder_2.eval()

    # Encode captions → text embeddings
    print(f"[Stage 3A] Encoding {len(pairs)} captions to text embeddings...")
    all_prompt_embeds = []
    all_pooled_embeds = []
    for i in tqdm(range(0, len(pairs), batch_size), desc="Text encode"):
        batch_pairs = pairs[i : i + batch_size]
        captions = [p["canonical_caption"] for p in batch_pairs]

        # Tokenize for both encoders
        tokens_1 = tokenizer_1(
            captions, padding="max_length", max_length=tokenizer_1.model_max_length,
            truncation=True, return_tensors="pt",
        ).input_ids.to(device)
        tokens_2 = tokenizer_2(
            captions, padding="max_length", max_length=tokenizer_2.model_max_length,
            truncation=True, return_tensors="pt",
        ).input_ids.to(device)

        # Encode with both text encoders
        enc1_out = text_encoder_1(tokens_1, output_hidden_states=True)
        enc2_out = text_encoder_2(tokens_2, output_hidden_states=True)

        # SDXL concatenates penultimate hidden states from both encoders
        prompt_embeds_1 = enc1_out.hidden_states[-2]  # (B, seq_len, 768)
        prompt_embeds_2 = enc2_out.hidden_states[-2]  # (B, seq_len, 1280)
        prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)  # (B, seq_len, 2048)

        # Pooled from text_encoder_2
        pooled_prompt_embeds = enc2_out.text_embeds  # (B, 1280)

        all_prompt_embeds.append(prompt_embeds.cpu().float())
        all_pooled_embeds.append(pooled_prompt_embeds.cpu().float())

    all_prompt_embeds = torch.cat(all_prompt_embeds, dim=0)    # (N, seq_len, 2048)
    all_pooled_embeds = torch.cat(all_pooled_embeds, dim=0)    # (N, 1280)
    print(f"[Stage 3A] Text embedding shape: {list(all_prompt_embeds.shape)}")
    print(f"[Stage 3A] Pooled embedding shape: {list(all_pooled_embeds.shape)}")

    # Free text encoders before loading DINOv2
    del text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2
    torch.cuda.empty_cache()

    # Encode images → DINOv2 features (for clustering)
    print(f"[Stage 3A] Loading DINOv2 (dinov2_vitb14)...")
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

    # Save tensors
    text_embeds_path = output_dir / "text_embeds.pt"
    pooled_embeds_path = output_dir / "pooled_embeds.pt"
    dino_embeds_path = output_dir / "dino_embeds.pt"
    index_path = output_dir / "encode_index.json"

    torch.save(all_prompt_embeds, text_embeds_path)
    torch.save(all_pooled_embeds, pooled_embeds_path)
    torch.save(all_dino_embeds, dino_embeds_path)

    # Save index (lightweight metadata without tensors)
    index_data = {
        "num_samples": len(pairs),
        "model_name": model_name,
        "resolution": resolution,
        "text_embed_shape": list(all_prompt_embeds.shape),
        "pooled_embed_shape": list(all_pooled_embeds.shape),
        "dino_embed_shape": list(all_dino_embeds.shape),
        "samples": [{k: v for k, v in p.items()} for p in pairs],
    }
    write_json(index_path, index_data)

    print(f"[Stage 3A] Encoding complete. Saved to {output_dir}")
    return EncodeResult(
        text_embeds_path=str(text_embeds_path),
        pooled_embeds_path=str(pooled_embeds_path),
        dino_embeds_path=str(dino_embeds_path),
        index_path=str(index_path),
        num_samples=len(pairs),
        text_embed_dim=all_prompt_embeds.shape[-1],
        pooled_embed_dim=all_pooled_embeds.shape[-1],
        dino_embed_dim=all_dino_embeds.shape[-1],
    )
