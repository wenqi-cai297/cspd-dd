"""Stage 3A — Encode images to VAE latents and captions to text embeddings.

Uses the same SDXL components as Stage 2 to ensure consistent latent/embedding spaces.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from cspd_stage1.io_utils import write_json


@dataclass(slots=True)
class EncodeResult:
    """Result of encoding a dataset split."""

    latents_path: str          # .pt file with stacked VAE latents
    text_embeds_path: str      # .pt file with stacked text embeddings
    pooled_embeds_path: str    # .pt file with stacked pooled text embeddings
    index_path: str            # .json mapping index → record metadata
    num_samples: int
    latent_shape: list[int]
    text_embed_dim: int
    pooled_embed_dim: int


def _load_image(path: str | Path, resolution: int) -> Image.Image:
    """Load and preprocess an image to target resolution."""
    img = Image.open(path).convert("RGB")
    img = img.resize((resolution, resolution), Image.LANCZOS)
    return img


def _image_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert PIL image to normalized tensor [-1, 1] in (C, H, W) format."""
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # HWC → CHW
    tensor = tensor * 2.0 - 1.0  # [0,1] → [-1,1]
    return tensor


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
    """Encode all paired images and captions to latent/embedding tensors.

    Args:
        dataset_root: ImageFolder dataset root (same as Stage 2).
        render_input: Path to Stage 1C records.jsonl.
        output_dir: Directory for output .pt and index files.
        model_name: SDXL model identifier.
        resolution: Image resolution for VAE encoding.
        batch_size: Encoding batch size.
        device: Torch device.
        dtype: Weight dtype (float16 or bfloat16).

    Returns:
        EncodeResult with paths to saved tensors and metadata.
    """
    from diffusers import AutoencoderKL
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16

    # Build pairs index
    print("[Stage 3A] Building pairs index...")
    pairs = _load_pairs_index(render_input, dataset_root)
    if not pairs:
        raise ValueError("No pairs found. Check dataset_root and render_input paths.")
    print(f"[Stage 3A] Found {len(pairs)} paired samples")

    # Load VAE
    print(f"[Stage 3A] Loading VAE from {model_name}...")
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae", torch_dtype=torch_dtype)
    vae = vae.to(device)
    vae.eval()
    vae_scaling_factor = vae.config.scaling_factor

    # Load text encoders + tokenizers
    print(f"[Stage 3A] Loading text encoders from {model_name}...")
    tokenizer_1 = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer_2")
    text_encoder_1 = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder", torch_dtype=torch_dtype).to(device)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(model_name, subfolder="text_encoder_2", torch_dtype=torch_dtype).to(device)
    text_encoder_1.eval()
    text_encoder_2.eval()

    # Encode images → VAE latents
    print(f"[Stage 3A] Encoding {len(pairs)} images to VAE latents...")
    all_latents = []
    for i in tqdm(range(0, len(pairs), batch_size), desc="VAE encode"):
        batch_pairs = pairs[i : i + batch_size]
        images = [_load_image(p["image_path"], resolution) for p in batch_pairs]
        pixel_values = torch.stack([_image_to_tensor(img) for img in images]).to(device, dtype=torch_dtype)
        latent_dist = vae.encode(pixel_values).latent_dist
        latents = latent_dist.sample() * vae_scaling_factor
        all_latents.append(latents.cpu().float())

    all_latents = torch.cat(all_latents, dim=0)  # (N, C, H, W)
    print(f"[Stage 3A] Latent tensor shape: {list(all_latents.shape)}")

    # Encode captions → text embeddings
    print(f"[Stage 3A] Encoding {len(pairs)} captions to text embeddings...")
    all_prompt_embeds = []
    all_pooled_embeds = []
    for i in tqdm(range(0, len(pairs), batch_size), desc="Text encode"):
        batch_pairs = pairs[i : i + batch_size]
        captions = [p["canonical_caption"] for p in batch_pairs]

        # Tokenize for both encoders
        tokens_1 = tokenizer_1(captions, padding="max_length", max_length=tokenizer_1.model_max_length, truncation=True, return_tensors="pt").input_ids.to(device)
        tokens_2 = tokenizer_2(captions, padding="max_length", max_length=tokenizer_2.model_max_length, truncation=True, return_tensors="pt").input_ids.to(device)

        # Encode with both CLIP encoders (SDXL concatenates penultimate hidden states)
        enc_out_1 = text_encoder_1(tokens_1, output_hidden_states=True, return_dict=True)
        enc_out_2 = text_encoder_2(tokens_2, output_hidden_states=True, return_dict=True)

        prompt_embeds_1 = enc_out_1.hidden_states[-2]  # penultimate layer
        prompt_embeds_2 = enc_out_2.hidden_states[-2]
        prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)  # concat along feature dim

        pooled_prompt_embeds = enc_out_2.text_embeds  # pooled from second encoder only

        all_prompt_embeds.append(prompt_embeds.cpu().float())
        all_pooled_embeds.append(pooled_prompt_embeds.cpu().float())

    all_prompt_embeds = torch.cat(all_prompt_embeds, dim=0)    # (N, seq_len, dim)
    all_pooled_embeds = torch.cat(all_pooled_embeds, dim=0)    # (N, pooled_dim)
    print(f"[Stage 3A] Text embedding shape: {list(all_prompt_embeds.shape)}")
    print(f"[Stage 3A] Pooled embedding shape: {list(all_pooled_embeds.shape)}")

    # Save tensors
    latents_path = output_dir / "latents.pt"
    text_embeds_path = output_dir / "text_embeds.pt"
    pooled_embeds_path = output_dir / "pooled_embeds.pt"
    index_path = output_dir / "encode_index.json"

    torch.save(all_latents, latents_path)
    torch.save(all_prompt_embeds, text_embeds_path)
    torch.save(all_pooled_embeds, pooled_embeds_path)

    # Save index (lightweight metadata without tensors)
    index_data = {
        "num_samples": len(pairs),
        "model_name": model_name,
        "resolution": resolution,
        "latent_shape": list(all_latents.shape),
        "text_embed_shape": list(all_prompt_embeds.shape),
        "pooled_embed_shape": list(all_pooled_embeds.shape),
        "vae_scaling_factor": vae_scaling_factor,
        "samples": [{k: v for k, v in p.items()} for p in pairs],
    }
    write_json(index_path, index_data)

    # Cleanup GPU
    del vae, text_encoder_1, text_encoder_2
    torch.cuda.empty_cache()

    print(f"[Stage 3A] Encoding complete. Saved to {output_dir}")
    return EncodeResult(
        latents_path=str(latents_path),
        text_embeds_path=str(text_embeds_path),
        pooled_embeds_path=str(pooled_embeds_path),
        index_path=str(index_path),
        num_samples=len(pairs),
        latent_shape=list(all_latents.shape),
        text_embed_dim=all_prompt_embeds.shape[-1],
        pooled_embed_dim=all_pooled_embeds.shape[-1],
    )
