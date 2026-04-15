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
    """A single mode for one cluster."""

    cluster_id: int
    class_name: str
    class_name_raw: str
    archetype: str
    num_members: int

    # Medoid (closest real sample to DINOv2 centroid)
    medoid_index: int
    medoid_record_id: str
    representative_caption: str      # caption of the medoid

    # Member indices (into the global samples list)
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
    modes_index_path: str        # .json: per-mode metadata


def _extract_modes_from_labels(
    *,
    labels: np.ndarray,
    n_clusters: int,
    class_indices: list[int],
    class_dino: np.ndarray,
    samples: list[dict[str, Any]],
    class_name: str,
    class_name_raw: str,
    archetype: str,
) -> list[ClusterMode]:
    """Shared mode extraction: given cluster labels, find DINOv2 medoid per cluster.

    The medoid's canonical caption becomes the representative caption for Stage 4.
    """
    modes = []

    unique_labels = sorted(set(int(l) for l in labels if l >= 0))  # skip noise label -1

    for k_idx, k in enumerate(unique_labels):
        member_mask = labels == k
        member_local_indices = np.where(member_mask)[0].tolist()
        member_global_indices = [class_indices[i] for i in member_local_indices]

        if len(member_local_indices) == 0:
            continue

        # Medoid: closest real sample to DINOv2 centroid
        member_dino = class_dino[member_local_indices]
        dino_centroid = member_dino.mean(axis=0)
        distances_to_centroid = np.linalg.norm(member_dino - dino_centroid, axis=1)
        medoid_local = member_local_indices[int(np.argmin(distances_to_centroid))]
        medoid_global = class_indices[medoid_local]

        modes.append(ClusterMode(
            cluster_id=k_idx,
            class_name=class_name,
            class_name_raw=class_name_raw,
            archetype=archetype,
            num_members=len(member_local_indices),
            medoid_index=medoid_global,
            medoid_record_id=samples[medoid_global]["record_id"],
            representative_caption=samples[medoid_global]["canonical_caption"],
            member_indices=member_global_indices,
        ))

    return modes


def _caption_token_distance(cap1: str, cap2: str) -> float:
    """Token-level Jaccard distance between two captions. 1.0 = completely different."""
    words1 = set(cap1.lower().split())
    words2 = set(cap2.lower().split())
    if not words1 or not words2:
        return 1.0
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    return 1.0 - intersection / union


def _diversify_captions(
    modes: list[ClusterMode],
    samples: list[dict[str, Any]],
) -> None:
    """Replace representative captions with most diverse alternatives from each cluster.

    Greedy selection: process modes in order, for each mode pick the member caption
    that maximizes minimum token distance from all already-selected captions.
    The medoid_index/medoid_record_id stay as the visual medoid (DINOv2 centroid);
    only the representative_caption may change to a different member's caption.
    """
    if len(modes) <= 1:
        return

    selected_captions: list[str] = []

    for mode in modes:
        member_captions = [
            (idx, samples[idx]["canonical_caption"])
            for idx in mode.member_indices
            if samples[idx].get("canonical_caption")
        ]

        if not selected_captions:
            # First mode: keep medoid caption
            selected_captions.append(mode.representative_caption)
            continue

        # Find the member caption most different from all already-selected
        best_caption = mode.representative_caption
        best_min_dist = -1.0

        for _, caption in member_captions:
            min_dist = min(_caption_token_distance(caption, sel) for sel in selected_captions)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_caption = caption

        mode.representative_caption = best_caption
        selected_captions.append(best_caption)


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
    dino_embeds: torch.Tensor,
    samples: list[dict[str, Any]],
    ipc: int,
    seed: int = 42,
) -> ClassClusterResult:
    """Cluster one class using K-Means on DINOv2 features."""
    class_meta = samples[class_indices[0]]
    class_name = class_meta["class_name"]
    class_name_raw = class_meta["class_name_raw"]
    archetype = class_meta["archetype"]

    class_dino = dino_embeds[class_indices].numpy()
    n_samples = len(class_indices)
    n_clusters = min(ipc, n_samples)

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(class_dino)

    modes = _extract_modes_from_labels(
        labels=labels, n_clusters=n_clusters,
        class_indices=class_indices, class_dino=class_dino,
        samples=samples,
        class_name=class_name, class_name_raw=class_name_raw, archetype=archetype,
    )
    _diversify_captions(modes, samples)

    return ClassClusterResult(
        class_name=class_name, class_name_raw=class_name_raw, archetype=archetype,
        num_samples=n_samples, ipc=ipc, num_clusters=len(modes), modes=modes,
    )


def cluster_class_hdbscan(
    *,
    class_indices: list[int],
    dino_embeds: torch.Tensor,
    samples: list[dict[str, Any]],
    ipc: int,
    seed: int = 42,
    min_cluster_size: int = 15,
    min_samples: int = 3,
    pca_dim: int = 50,
) -> ClassClusterResult:
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

    class_dino = dino_embeds[class_indices].numpy()

    n_samples = len(class_indices)
    cluster_features = class_dino

    # Step 1: Optional PCA dimensionality reduction
    # pca_dim=0 skips PCA; DINO features (768-dim) usually don't need PCA
    if pca_dim > 0 and cluster_features.shape[1] > pca_dim:
        pca_components = min(pca_dim, n_samples, cluster_features.shape[1])
        pca = PCA(n_components=pca_components, random_state=seed)
        flat_reduced = pca.fit_transform(cluster_features)
    else:
        flat_reduced = cluster_features

    # Step 2: HDBSCAN mode discovery
    min_cs = min(min_cluster_size, max(n_samples // ipc, 5))
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cs, min_samples=min_samples)
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
            class_indices=class_indices, dino_embeds=dino_embeds,
            samples=samples, ipc=ipc, seed=seed,
        )

    # Build per-mode member lists
    mode_members: dict[int, list[int]] = {}
    for m in discovered_modes:
        mode_members[m] = [i for i in range(n_samples) if hdb_labels[i] == m]
    mode_sizes = [len(mode_members[m]) for m in discovered_modes]

    # Step 3+5: If more modes than IPC, select most diverse via farthest-point sampling
    if n_discovered > ipc:
        mode_centroids = np.stack([
            cluster_features[mode_members[m]].mean(axis=0) for m in discovered_modes
        ])
        selected_mode_indices = _farthest_point_sampling(mode_centroids, ipc)
        discovered_modes = [discovered_modes[i] for i in selected_mode_indices]
        mode_sizes = [len(mode_members[m]) for m in discovered_modes]
        # Reassign unselected modes' members to nearest selected mode
        selected_centroids = np.stack([
            cluster_features[mode_members[m]].mean(axis=0) for m in discovered_modes
        ])
        all_mode_ids = set(int(l) for l in hdb_labels if l >= 0)
        unselected = all_mode_ids - set(discovered_modes)
        for um in unselected:
            um_centroid = cluster_features[mode_members[um]].mean(axis=0)
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
            sub_groups = _sub_cluster_mode(class_dino, members, n_alloc, seed)
            for group in sub_groups:
                for mi in group:
                    final_labels[mi] = final_cluster_id
                final_cluster_id += 1

    # Handle any remaining unassigned (shouldn't happen but defensive)
    for i in range(n_samples):
        if final_labels[i] < 0:
            final_labels[i] = 0

    modes = _extract_modes_from_labels(
        labels=final_labels, n_clusters=final_cluster_id,
        class_indices=class_indices, class_dino=class_dino,
        samples=samples,
        class_name=class_name, class_name_raw=class_name_raw, archetype=archetype,
    )
    _diversify_captions(modes, samples)

    return ClassClusterResult(
        class_name=class_name, class_name_raw=class_name_raw, archetype=archetype,
        num_samples=n_samples, ipc=ipc, num_clusters=len(modes), modes=modes,
    )


def cluster_class(
    *,
    class_indices: list[int],
    dino_embeds: torch.Tensor,
    samples: list[dict[str, Any]],
    ipc: int,
    seed: int = 42,
    method: str = "kmeans",
    min_cluster_size: int = 15,
    min_samples: int = 3,
    pca_dim: int = 50,
) -> ClassClusterResult:
    """Cluster one class using DINOv2 features and extract modes.

    Args:
        method: "kmeans" (baseline) or "hdbscan" (mode discovery).
        dino_embeds: DINOv2 features tensor (required).
    """
    if method == "hdbscan":
        return cluster_class_hdbscan(
            class_indices=class_indices, dino_embeds=dino_embeds,
            samples=samples, ipc=ipc, seed=seed,
            min_cluster_size=min_cluster_size, min_samples=min_samples, pca_dim=pca_dim,
        )
    else:
        return cluster_class_kmeans(
            class_indices=class_indices, dino_embeds=dino_embeds,
            samples=samples, ipc=ipc, seed=seed,
        )


def run_stage3_clustering(
    *,
    encode_dir: str | Path,
    output_dir: str | Path,
    ipc: int = 10,
    seed: int = 42,
    method: str = "kmeans",
    min_cluster_size: int = 15,
    min_samples: int = 3,
    pca_dim: int = 50,
) -> Stage3Result:
    """Run full Stage 3 pipeline: load encoded tensors, cluster per class, extract modes.

    Clustering uses DINOv2 features. VAE latents are not needed (Stage 4 uses text2img).

    Args:
        encode_dir: Directory containing Stage 3A encode outputs (text_embeds.pt, dino_embeds.pt, etc.).
        output_dir: Directory for Stage 3 mode outputs.
        ipc: Images per class — number of clusters per class.
        seed: Random seed.
        method: "kmeans" (baseline) or "hdbscan" (mode discovery).
        min_cluster_size: HDBSCAN min_cluster_size parameter.
        min_samples: HDBSCAN min_samples parameter (core point neighborhood density).
        pca_dim: PCA dimensions for HDBSCAN pre-processing (0 to skip PCA).

    Returns:
        Stage3Result with paths to mode tensors and metadata.
    """
    encode_dir = Path(encode_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load encoded tensors
    print("[Stage 3] Loading encoded tensors...")
    dino_embeds = torch.load(encode_dir / "dino_embeds.pt", weights_only=True)
    print(f"[Stage 3] Loaded DINOv2 features: {list(dino_embeds.shape)}")

    with open(encode_dir / "encode_index.json", encoding="utf-8") as f:
        encode_index = json.load(f)
    samples = encode_index["samples"]

    print(f"[Stage 3] Loaded {len(samples)} samples, DINOv2 {list(dino_embeds.shape)}")

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

    for class_raw, indices in sorted(class_groups.items()):
        class_name = samples[indices[0]]["class_name"]
        print(f"[Stage 3] Clustering class '{class_name}' ({len(indices)} samples, method={method})...")

        class_result = cluster_class(
            class_indices=indices,
            dino_embeds=dino_embeds,
            samples=samples,
            ipc=ipc,
            seed=seed,
            method=method,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            pca_dim=pca_dim,
        )

        all_class_results.append(class_result)

    total_modes = sum(cr.num_clusters for cr in all_class_results)
    print(f"[Stage 3] Total modes: {total_modes}")

    # Save modes index
    modes_index_path = output_dir / "modes_index.json"

    # Build modes index
    modes_index_data: list[dict[str, Any]] = []
    global_idx = 0
    for class_result in all_class_results:
        for mode in class_result.modes:
            modes_index_data.append({
                "global_mode_index": global_idx,
                "cluster_id": mode.cluster_id,
                "class_name": mode.class_name,
                "class_name_raw": mode.class_name_raw,
                "archetype": mode.archetype,
                "num_members": mode.num_members,
                "medoid_record_id": mode.medoid_record_id,
                "representative_caption": mode.representative_caption,
            })
            global_idx += 1

    summary = {
        "ipc": ipc,
        "seed": seed,
        "cluster_method": method,
        "cluster_space": "dino",
        "num_classes": len(all_class_results),
        "total_modes": total_modes,
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
        modes_index_path=str(modes_index_path),
    )
