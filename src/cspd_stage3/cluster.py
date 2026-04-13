"""Stage 3A/3B/3C — Per-class clustering and visual/semantic mode extraction.

Two clustering methods are supported:
- **kmeans** (baseline): K-Means with K = IPC. Simple and fast, but may over-represent
  head modes in long-tailed distributions.
- **hdbscan**: Density-based mode discovery that first finds natural modes via HDBSCAN,
  then allocates IPC representatives proportionally. Better captures tail modes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

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


def _extract_modes_from_labels(
    *,
    labels: np.ndarray,
    n_clusters: int,
    class_indices: list[int],
    class_latents: torch.Tensor,
    class_text: torch.Tensor,
    class_pooled: torch.Tensor,
    flat_latents: np.ndarray,
    samples: list[dict[str, Any]],
    class_name: str,
    class_name_raw: str,
    archetype: str,
) -> tuple[list[ClusterMode], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Shared mode extraction logic: given cluster labels, extract visual/semantic modes."""
    modes = []
    visual_mode_latents = []
    semantic_mode_embeds = []
    pooled_mode_embeds = []

    unique_labels = sorted(set(int(l) for l in labels if l >= 0))  # skip noise label -1

    for k_idx, k in enumerate(unique_labels):
        member_mask = labels == k
        member_local_indices = np.where(member_mask)[0].tolist()
        member_global_indices = [class_indices[i] for i in member_local_indices]

        if len(member_local_indices) == 0:
            continue

        # --- Visual mode ---
        cluster_latents = class_latents[member_local_indices]
        visual_centroid = cluster_latents.mean(dim=0)

        centroid_flat = _flatten_latent(visual_centroid)
        member_flat = flat_latents[member_local_indices]
        distances_to_centroid = np.linalg.norm(member_flat - centroid_flat, axis=1)
        visual_medoid_local = member_local_indices[int(np.argmin(distances_to_centroid))]
        visual_medoid_global = class_indices[visual_medoid_local]

        # --- Semantic mode ---
        cluster_text = class_text[member_local_indices]
        cluster_pooled = class_pooled[member_local_indices]
        semantic_centroid = cluster_text.mean(dim=0)
        pooled_centroid = cluster_pooled.mean(dim=0)

        pooled_flat = cluster_pooled.numpy()
        pooled_centroid_np = pooled_centroid.numpy()
        distances_to_text_centroid = np.linalg.norm(pooled_flat - pooled_centroid_np, axis=1)
        semantic_medoid_local = member_local_indices[int(np.argmin(distances_to_text_centroid))]
        semantic_medoid_global = class_indices[semantic_medoid_local]

        visual_mode_latents.append(visual_centroid)
        semantic_mode_embeds.append(semantic_centroid)
        pooled_mode_embeds.append(pooled_centroid)

        modes.append(ClusterMode(
            cluster_id=k_idx,
            class_name=class_name,
            class_name_raw=class_name_raw,
            archetype=archetype,
            num_members=len(member_local_indices),
            visual_centroid_index=-1,
            visual_medoid_index=visual_medoid_global,
            visual_medoid_record_id=samples[visual_medoid_global]["record_id"],
            representative_caption=samples[visual_medoid_global]["canonical_caption"],
            semantic_medoid_index=semantic_medoid_global,
            semantic_medoid_record_id=samples[semantic_medoid_global]["record_id"],
            semantic_medoid_caption=samples[semantic_medoid_global]["canonical_caption"],
            member_indices=member_global_indices,
        ))

    return modes, visual_mode_latents, semantic_mode_embeds, pooled_mode_embeds


def _farthest_point_sampling(points: np.ndarray, n: int) -> list[int]:
    """Greedy farthest-point sampling. Returns indices of n most spread-out points."""
    if n >= len(points):
        return list(range(len(points)))
    selected = [0]
    min_distances = np.full(len(points), np.inf)
    for _ in range(n - 1):
        last = points[selected[-1]]
        dists = np.linalg.norm(points - last, axis=1)
        min_distances = np.minimum(min_distances, dists)
        min_distances[selected] = -1  # exclude already selected
        selected.append(int(np.argmax(min_distances)))
    return selected


def _allocate_ipc_to_modes(mode_sizes: list[int], ipc: int) -> list[int]:
    """Allocate IPC representatives across discovered modes proportionally.

    Every mode gets at least 1 representative. Remaining quota is distributed
    proportionally to mode size (with rounding).
    """
    n_modes = len(mode_sizes)
    if n_modes == 0:
        return []
    if n_modes >= ipc:
        # More modes than IPC: each selected mode gets exactly 1
        return [1] * ipc

    # Every mode gets at least 1
    allocation = [1] * n_modes
    remaining = ipc - n_modes

    if remaining > 0:
        total = sum(mode_sizes)
        # Proportional allocation of remaining quota
        raw_shares = [(s / total) * remaining for s in mode_sizes]
        int_shares = [int(s) for s in raw_shares]
        # Distribute rounding remainders to largest fractional parts
        fractions = [s - int(s) for s in raw_shares]
        leftover = remaining - sum(int_shares)
        top_indices = sorted(range(n_modes), key=lambda i: -fractions[i])
        for i in range(leftover):
            int_shares[top_indices[i]] += 1
        allocation = [a + s for a, s in zip(allocation, int_shares)]

    return allocation


def _sub_cluster_mode(
    flat_latents: np.ndarray,
    member_indices: list[int],
    n_sub: int,
    seed: int,
) -> list[np.ndarray]:
    """Sub-cluster a single mode into n_sub groups using K-Means."""
    if n_sub <= 1:
        return [np.array(member_indices)]
    member_flat = flat_latents[member_indices]
    n_sub = min(n_sub, len(member_indices))
    km = KMeans(n_clusters=n_sub, random_state=seed, n_init=5)
    sub_labels = km.fit_predict(member_flat)
    groups = []
    for k in range(n_sub):
        group = [member_indices[i] for i in range(len(member_indices)) if sub_labels[i] == k]
        if group:
            groups.append(np.array(group))
    return groups


def cluster_class_kmeans(
    *,
    class_indices: list[int],
    latents: torch.Tensor,
    text_embeds: torch.Tensor,
    pooled_embeds: torch.Tensor,
    samples: list[dict[str, Any]],
    ipc: int,
    seed: int = 42,
) -> tuple[ClassClusterResult, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cluster one class using K-Means (baseline method)."""
    class_meta = samples[class_indices[0]]
    class_name = class_meta["class_name"]
    class_name_raw = class_meta["class_name_raw"]
    archetype = class_meta["archetype"]

    class_latents = latents[class_indices]
    class_text = text_embeds[class_indices]
    class_pooled = pooled_embeds[class_indices]

    n_samples = len(class_indices)
    n_clusters = min(ipc, n_samples)

    flat_latents = np.stack([_flatten_latent(class_latents[i]) for i in range(n_samples)])

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(flat_latents)

    modes, vis_list, sem_list, pool_list = _extract_modes_from_labels(
        labels=labels, n_clusters=n_clusters,
        class_indices=class_indices, class_latents=class_latents,
        class_text=class_text, class_pooled=class_pooled,
        flat_latents=flat_latents, samples=samples,
        class_name=class_name, class_name_raw=class_name_raw, archetype=archetype,
    )

    visual_modes = torch.stack(vis_list) if vis_list else torch.empty(0)
    semantic_modes = torch.stack(sem_list) if sem_list else torch.empty(0)
    pooled_modes = torch.stack(pool_list) if pool_list else torch.empty(0)

    return ClassClusterResult(
        class_name=class_name, class_name_raw=class_name_raw, archetype=archetype,
        num_samples=n_samples, ipc=ipc, num_clusters=len(modes), modes=modes,
    ), visual_modes, semantic_modes, pooled_modes


def cluster_class_hdbscan(
    *,
    class_indices: list[int],
    latents: torch.Tensor,
    text_embeds: torch.Tensor,
    pooled_embeds: torch.Tensor,
    samples: list[dict[str, Any]],
    ipc: int,
    seed: int = 42,
    min_cluster_size: int = 15,
    pca_dim: int = 50,
) -> tuple[ClassClusterResult, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cluster one class using HDBSCAN mode discovery + proportional IPC allocation.

    Steps:
    1. PCA to reduce dimensionality (HDBSCAN struggles in very high dims)
    2. HDBSCAN to discover natural modes (no preset K)
    3. Allocate IPC quota to modes proportionally (every mode gets >= 1)
    4. If a mode gets > 1 quota, sub-cluster with K-Means for intra-mode diversity
    5. If more modes than IPC, select IPC most diverse modes via farthest-point sampling
    6. Extract visual/semantic modes from final assignment
    """
    import hdbscan

    class_meta = samples[class_indices[0]]
    class_name = class_meta["class_name"]
    class_name_raw = class_meta["class_name_raw"]
    archetype = class_meta["archetype"]

    class_latents = latents[class_indices]
    class_text = text_embeds[class_indices]
    class_pooled = pooled_embeds[class_indices]

    n_samples = len(class_indices)
    flat_latents = np.stack([_flatten_latent(class_latents[i]) for i in range(n_samples)])

    # Step 1: PCA dimensionality reduction
    pca_components = min(pca_dim, n_samples, flat_latents.shape[1])
    pca = PCA(n_components=pca_components, random_state=seed)
    flat_reduced = pca.fit_transform(flat_latents)

    # Step 2: HDBSCAN mode discovery
    min_cs = min(min_cluster_size, max(n_samples // ipc, 5))
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cs, min_samples=3)
    hdb_labels = clusterer.fit_predict(flat_reduced)

    discovered_modes = sorted(set(int(l) for l in hdb_labels if l >= 0))
    noise_indices = [i for i in range(n_samples) if hdb_labels[i] == -1]
    n_discovered = len(discovered_modes)

    # Assign noise points to nearest discovered mode
    if noise_indices and discovered_modes:
        mode_centroids_reduced = []
        for m in discovered_modes:
            mask = hdb_labels == m
            mode_centroids_reduced.append(flat_reduced[mask].mean(axis=0))
        mode_centroids_reduced = np.stack(mode_centroids_reduced)
        for ni in noise_indices:
            dists = np.linalg.norm(mode_centroids_reduced - flat_reduced[ni], axis=1)
            hdb_labels[ni] = discovered_modes[int(np.argmin(dists))]

    # Fallback: if HDBSCAN found 0 or 1 modes, fall back to K-Means
    if n_discovered <= 1:
        return cluster_class_kmeans(
            class_indices=class_indices, latents=latents, text_embeds=text_embeds,
            pooled_embeds=pooled_embeds, samples=samples, ipc=ipc, seed=seed,
        )

    # Build per-mode member lists
    mode_members: dict[int, list[int]] = {}
    for m in discovered_modes:
        mode_members[m] = [i for i in range(n_samples) if hdb_labels[i] == m]
    mode_sizes = [len(mode_members[m]) for m in discovered_modes]

    # Step 3+5: If more modes than IPC, select most diverse via farthest-point sampling
    if n_discovered > ipc:
        mode_centroids = np.stack([
            flat_latents[mode_members[m]].mean(axis=0) for m in discovered_modes
        ])
        selected_mode_indices = _farthest_point_sampling(mode_centroids, ipc)
        discovered_modes = [discovered_modes[i] for i in selected_mode_indices]
        mode_sizes = [len(mode_members[m]) for m in discovered_modes]
        # Reassign unselected modes' members to nearest selected mode
        selected_centroids = np.stack([
            flat_latents[mode_members[m]].mean(axis=0) for m in discovered_modes
        ])
        all_mode_ids = set(int(l) for l in hdb_labels if l >= 0)
        unselected = all_mode_ids - set(discovered_modes)
        for um in unselected:
            um_centroid = flat_latents[mode_members[um]].mean(axis=0)
            dists = np.linalg.norm(selected_centroids - um_centroid, axis=1)
            nearest = discovered_modes[int(np.argmin(dists))]
            for idx in mode_members[um]:
                hdb_labels[idx] = nearest
            mode_members[nearest].extend(mode_members[um])
        mode_sizes = [len(mode_members[m]) for m in discovered_modes]

    # Step 3: Allocate IPC quota proportionally
    allocation = _allocate_ipc_to_modes(mode_sizes, ipc)

    # Step 4: Sub-cluster modes that got > 1 quota, build final labels
    final_labels = np.full(n_samples, -1, dtype=np.int32)
    final_cluster_id = 0
    for mode_idx, mode_id in enumerate(discovered_modes):
        members = mode_members[mode_id]
        n_alloc = allocation[mode_idx]
        if n_alloc <= 1:
            for mi in members:
                final_labels[mi] = final_cluster_id
            final_cluster_id += 1
        else:
            sub_groups = _sub_cluster_mode(flat_latents, members, n_alloc, seed)
            for group in sub_groups:
                for mi in group:
                    final_labels[mi] = final_cluster_id
                final_cluster_id += 1

    # Handle any remaining unassigned (shouldn't happen but defensive)
    for i in range(n_samples):
        if final_labels[i] < 0:
            final_labels[i] = 0

    modes, vis_list, sem_list, pool_list = _extract_modes_from_labels(
        labels=final_labels, n_clusters=final_cluster_id,
        class_indices=class_indices, class_latents=class_latents,
        class_text=class_text, class_pooled=class_pooled,
        flat_latents=flat_latents, samples=samples,
        class_name=class_name, class_name_raw=class_name_raw, archetype=archetype,
    )

    visual_modes = torch.stack(vis_list) if vis_list else torch.empty(0)
    semantic_modes = torch.stack(sem_list) if sem_list else torch.empty(0)
    pooled_modes = torch.stack(pool_list) if pool_list else torch.empty(0)

    return ClassClusterResult(
        class_name=class_name, class_name_raw=class_name_raw, archetype=archetype,
        num_samples=n_samples, ipc=ipc, num_clusters=len(modes), modes=modes,
    ), visual_modes, semantic_modes, pooled_modes


def cluster_class(
    *,
    class_indices: list[int],
    latents: torch.Tensor,
    text_embeds: torch.Tensor,
    pooled_embeds: torch.Tensor,
    samples: list[dict[str, Any]],
    ipc: int,
    seed: int = 42,
    method: str = "kmeans",
    min_cluster_size: int = 15,
    pca_dim: int = 50,
) -> tuple[ClassClusterResult, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cluster one class and extract visual/semantic modes.

    Args:
        method: "kmeans" (baseline) or "hdbscan" (mode discovery).
    """
    if method == "hdbscan":
        return cluster_class_hdbscan(
            class_indices=class_indices, latents=latents, text_embeds=text_embeds,
            pooled_embeds=pooled_embeds, samples=samples, ipc=ipc, seed=seed,
            min_cluster_size=min_cluster_size, pca_dim=pca_dim,
        )
    else:
        return cluster_class_kmeans(
            class_indices=class_indices, latents=latents, text_embeds=text_embeds,
            pooled_embeds=pooled_embeds, samples=samples, ipc=ipc, seed=seed,
        )


def run_stage3_clustering(
    *,
    encode_dir: str | Path,
    output_dir: str | Path,
    ipc: int = 10,
    seed: int = 42,
    method: str = "kmeans",
    min_cluster_size: int = 15,
    pca_dim: int = 50,
) -> Stage3Result:
    """Run full Stage 3 pipeline: load encoded tensors, cluster per class, extract modes.

    Args:
        encode_dir: Directory containing Stage 3A encode outputs (latents.pt, text_embeds.pt, etc.).
        output_dir: Directory for Stage 3 mode outputs.
        ipc: Images per class — number of clusters per class.
        seed: Random seed.
        method: "kmeans" (baseline) or "hdbscan" (mode discovery).
        min_cluster_size: HDBSCAN min_cluster_size parameter.
        pca_dim: PCA dimensions for HDBSCAN pre-processing.

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

    print(f"[Stage 3] Found {len(class_groups)} classes, IPC={ipc}, method={method}")

    # Cluster each class
    all_class_results = []
    all_visual_modes = []
    all_semantic_modes = []
    all_pooled_modes = []
    global_mode_index = 0

    for class_raw, indices in sorted(class_groups.items()):
        class_name = samples[indices[0]]["class_name"]
        print(f"[Stage 3] Clustering class '{class_name}' ({len(indices)} samples, method={method})...")

        class_result, vis_modes, sem_modes, pool_modes = cluster_class(
            class_indices=indices,
            latents=latents,
            text_embeds=text_embeds,
            pooled_embeds=pooled_embeds,
            samples=samples,
            ipc=ipc,
            seed=seed,
            method=method,
            min_cluster_size=min_cluster_size,
            pca_dim=pca_dim,
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
        "cluster_method": method,
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
