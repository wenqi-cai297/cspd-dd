"""Stage 3A/3B/3C — Per-class clustering and visual/semantic mode extraction.

Clusters VAE latents per class using K-Means (K = IPC), then extracts:
- Visual modes: centroid + medoid per cluster in latent space
- Semantic modes: mean text embedding per cluster + representative caption
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import numpy as np
from sklearn.cluster import KMeans

from cspd_stage1.io_utils import write_json


@dataclass(slots=True)
class ClusterMode:
    """A single visual+semantic mode for one cluster."""

    cluster_id: int
    class_name: str
    class_name_raw: str
    archetype: str
    num_members: int

    # Visual mode
    visual_centroid_index: int       # index into latents tensor (virtual, stored separately)
    visual_medoid_index: int         # index of closest real sample to centroid
    visual_medoid_record_id: str

    # Semantic mode
    representative_caption: str      # caption of the medoid (closest to embedding mean)
    semantic_medoid_index: int       # index of sample closest to text embedding mean
    semantic_medoid_record_id: str
    semantic_medoid_caption: str

    # Member indices (into the per-class subset)
    member_indices: list[int] = field(default_factory=list)


@dataclass(slots=True)
class ClassClusterResult:
    """Clustering result for one class."""

    class_name: str
    class_name_raw: str
    archetype: str
    num_samples: int
    ipc: int
    num_clusters: int
    modes: list[ClusterMode]


@dataclass(slots=True)
class Stage3Result:
    """Full Stage 3 result across all classes."""

    output_dir: str
    ipc: int
    num_classes: int
    total_modes: int
    class_results: list[ClassClusterResult]
    visual_modes_path: str       # .pt: (total_modes, C, H, W)
    semantic_modes_path: str     # .pt: (total_modes, seq_len, dim)
    pooled_modes_path: str       # .pt: (total_modes, pooled_dim)
    modes_index_path: str        # .json: per-mode metadata


def _flatten_latent(latent: torch.Tensor) -> np.ndarray:
    """Flatten a (C, H, W) latent to a 1D vector for clustering."""
    return latent.reshape(-1).numpy()


def cluster_class(
    *,
    class_indices: list[int],
    latents: torch.Tensor,
    text_embeds: torch.Tensor,
    pooled_embeds: torch.Tensor,
    samples: list[dict[str, Any]],
    ipc: int,
    seed: int = 42,
) -> tuple[ClassClusterResult, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cluster one class and extract visual/semantic modes.

    Args:
        class_indices: Indices into the global tensors for this class.
        latents: (N, C, H, W) full latent tensor.
        text_embeds: (N, seq_len, dim) full text embedding tensor.
        pooled_embeds: (N, pooled_dim) full pooled embedding tensor.
        samples: Full sample index list.
        ipc: Images per class (number of clusters).
        seed: Random seed for K-Means.

    Returns:
        Tuple of (ClassClusterResult, visual_modes, semantic_modes, pooled_modes).
    """
    class_meta = samples[class_indices[0]]
    class_name = class_meta["class_name"]
    class_name_raw = class_meta["class_name_raw"]
    archetype = class_meta["archetype"]

    # Extract class-specific tensors
    class_latents = latents[class_indices]       # (M, C, H, W)
    class_text = text_embeds[class_indices]       # (M, seq_len, dim)
    class_pooled = pooled_embeds[class_indices]   # (M, pooled_dim)

    n_samples = len(class_indices)
    n_clusters = min(ipc, n_samples)

    # Flatten latents for K-Means
    flat_latents = np.stack([_flatten_latent(class_latents[i]) for i in range(n_samples)])

    # K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(flat_latents)

    # Extract modes per cluster
    modes = []
    visual_mode_latents = []
    semantic_mode_embeds = []
    pooled_mode_embeds = []

    for k in range(n_clusters):
        member_mask = labels == k
        member_local_indices = np.where(member_mask)[0].tolist()
        member_global_indices = [class_indices[i] for i in member_local_indices]

        if len(member_local_indices) == 0:
            continue

        # --- Visual mode ---
        # Centroid in latent space
        cluster_latents = class_latents[member_local_indices]  # (M_k, C, H, W)
        visual_centroid = cluster_latents.mean(dim=0)          # (C, H, W)

        # Medoid: real sample closest to centroid
        centroid_flat = _flatten_latent(visual_centroid)
        member_flat = flat_latents[member_local_indices]
        distances_to_centroid = np.linalg.norm(member_flat - centroid_flat, axis=1)
        visual_medoid_local = member_local_indices[int(np.argmin(distances_to_centroid))]
        visual_medoid_global = class_indices[visual_medoid_local]

        # --- Semantic mode ---
        # Mean text embedding across cluster members
        cluster_text = class_text[member_local_indices]        # (M_k, seq_len, dim)
        cluster_pooled = class_pooled[member_local_indices]    # (M_k, pooled_dim)
        semantic_centroid = cluster_text.mean(dim=0)           # (seq_len, dim)
        pooled_centroid = cluster_pooled.mean(dim=0)           # (pooled_dim)

        # Semantic medoid: caption closest to mean pooled embedding
        pooled_flat = cluster_pooled.numpy()
        pooled_centroid_np = pooled_centroid.numpy()
        distances_to_text_centroid = np.linalg.norm(pooled_flat - pooled_centroid_np, axis=1)
        semantic_medoid_local = member_local_indices[int(np.argmin(distances_to_text_centroid))]
        semantic_medoid_global = class_indices[semantic_medoid_local]

        visual_mode_latents.append(visual_centroid)
        semantic_mode_embeds.append(semantic_centroid)
        pooled_mode_embeds.append(pooled_centroid)

        modes.append(ClusterMode(
            cluster_id=k,
            class_name=class_name,
            class_name_raw=class_name_raw,
            archetype=archetype,
            num_members=len(member_local_indices),
            visual_centroid_index=-1,  # filled later when stacking across classes
            visual_medoid_index=visual_medoid_global,
            visual_medoid_record_id=samples[visual_medoid_global]["record_id"],
            representative_caption=samples[visual_medoid_global]["canonical_caption"],
            semantic_medoid_index=semantic_medoid_global,
            semantic_medoid_record_id=samples[semantic_medoid_global]["record_id"],
            semantic_medoid_caption=samples[semantic_medoid_global]["canonical_caption"],
            member_indices=member_global_indices,
        ))

    visual_modes = torch.stack(visual_mode_latents) if visual_mode_latents else torch.empty(0)
    semantic_modes = torch.stack(semantic_mode_embeds) if semantic_mode_embeds else torch.empty(0)
    pooled_modes = torch.stack(pooled_mode_embeds) if pooled_mode_embeds else torch.empty(0)

    class_result = ClassClusterResult(
        class_name=class_name,
        class_name_raw=class_name_raw,
        archetype=archetype,
        num_samples=n_samples,
        ipc=ipc,
        num_clusters=len(modes),
        modes=modes,
    )

    return class_result, visual_modes, semantic_modes, pooled_modes


def run_stage3_clustering(
    *,
    encode_dir: str | Path,
    output_dir: str | Path,
    ipc: int = 10,
    seed: int = 42,
) -> Stage3Result:
    """Run full Stage 3 pipeline: load encoded tensors, cluster per class, extract modes.

    Args:
        encode_dir: Directory containing Stage 3A encode outputs (latents.pt, text_embeds.pt, etc.).
        output_dir: Directory for Stage 3 mode outputs.
        ipc: Images per class — number of clusters per class.
        seed: Random seed.

    Returns:
        Stage3Result with paths to mode tensors and metadata.
    """
    encode_dir = Path(encode_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load encoded tensors
    print("[Stage 3] Loading encoded tensors...")
    latents = torch.load(encode_dir / "latents.pt", weights_only=True)
    text_embeds = torch.load(encode_dir / "text_embeds.pt", weights_only=True)
    pooled_embeds = torch.load(encode_dir / "pooled_embeds.pt", weights_only=True)

    with open(encode_dir / "encode_index.json", encoding="utf-8") as f:
        encode_index = json.load(f)
    samples = encode_index["samples"]

    print(f"[Stage 3] Loaded {len(samples)} samples, latents {list(latents.shape)}")

    # Group samples by class
    class_groups: dict[str, list[int]] = {}
    for i, sample in enumerate(samples):
        key = sample["class_name_raw"]
        if key not in class_groups:
            class_groups[key] = []
        class_groups[key].append(i)

    print(f"[Stage 3] Found {len(class_groups)} classes, IPC={ipc}")

    # Cluster each class
    all_class_results = []
    all_visual_modes = []
    all_semantic_modes = []
    all_pooled_modes = []
    global_mode_index = 0

    for class_raw, indices in sorted(class_groups.items()):
        class_name = samples[indices[0]]["class_name"]
        print(f"[Stage 3] Clustering class '{class_name}' ({len(indices)} samples, K={ipc})...")

        class_result, vis_modes, sem_modes, pool_modes = cluster_class(
            class_indices=indices,
            latents=latents,
            text_embeds=text_embeds,
            pooled_embeds=pooled_embeds,
            samples=samples,
            ipc=ipc,
            seed=seed,
        )

        # Assign global mode indices
        for mode in class_result.modes:
            mode.visual_centroid_index = global_mode_index
            global_mode_index += 1

        all_class_results.append(class_result)
        if vis_modes.numel() > 0:
            all_visual_modes.append(vis_modes)
            all_semantic_modes.append(sem_modes)
            all_pooled_modes.append(pool_modes)

    # Stack all modes
    total_visual_modes = torch.cat(all_visual_modes, dim=0) if all_visual_modes else torch.empty(0)
    total_semantic_modes = torch.cat(all_semantic_modes, dim=0) if all_semantic_modes else torch.empty(0)
    total_pooled_modes = torch.cat(all_pooled_modes, dim=0) if all_pooled_modes else torch.empty(0)

    total_modes = total_visual_modes.shape[0]
    print(f"[Stage 3] Total modes: {total_modes}")
    print(f"[Stage 3] Visual modes shape: {list(total_visual_modes.shape)}")
    print(f"[Stage 3] Semantic modes shape: {list(total_semantic_modes.shape)}")

    # Save mode tensors
    visual_modes_path = output_dir / "visual_modes.pt"
    semantic_modes_path = output_dir / "semantic_modes.pt"
    pooled_modes_path = output_dir / "pooled_modes.pt"
    modes_index_path = output_dir / "modes_index.json"

    torch.save(total_visual_modes, visual_modes_path)
    torch.save(total_semantic_modes, semantic_modes_path)
    torch.save(total_pooled_modes, pooled_modes_path)

    # Build modes index
    modes_index_data: list[dict[str, Any]] = []
    for class_result in all_class_results:
        for mode in class_result.modes:
            modes_index_data.append({
                "global_mode_index": mode.visual_centroid_index,
                "cluster_id": mode.cluster_id,
                "class_name": mode.class_name,
                "class_name_raw": mode.class_name_raw,
                "archetype": mode.archetype,
                "num_members": mode.num_members,
                "visual_medoid_record_id": mode.visual_medoid_record_id,
                "representative_caption": mode.representative_caption,
                "semantic_medoid_record_id": mode.semantic_medoid_record_id,
                "semantic_medoid_caption": mode.semantic_medoid_caption,
            })

    summary = {
        "ipc": ipc,
        "seed": seed,
        "num_classes": len(all_class_results),
        "total_modes": total_modes,
        "visual_modes_shape": list(total_visual_modes.shape),
        "semantic_modes_shape": list(total_semantic_modes.shape),
        "pooled_modes_shape": list(total_pooled_modes.shape),
        "encode_dir": str(encode_dir.resolve()),
        "class_summary": [
            {
                "class_name": cr.class_name,
                "class_name_raw": cr.class_name_raw,
                "archetype": cr.archetype,
                "num_samples": cr.num_samples,
                "num_clusters": cr.num_clusters,
                "cluster_sizes": [m.num_members for m in cr.modes],
            }
            for cr in all_class_results
        ],
        "modes": modes_index_data,
    }
    write_json(modes_index_path, summary)
    write_json(output_dir / "stage3_summary.json", summary)

    print(f"[Stage 3] Complete. Saved to {output_dir}")

    return Stage3Result(
        output_dir=str(output_dir),
        ipc=ipc,
        num_classes=len(all_class_results),
        total_modes=total_modes,
        class_results=all_class_results,
        visual_modes_path=str(visual_modes_path),
        semantic_modes_path=str(semantic_modes_path),
        pooled_modes_path=str(pooled_modes_path),
        modes_index_path=str(modes_index_path),
    )
