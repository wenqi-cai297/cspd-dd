"""Stage 3B/3C — Per-class mode discovery and medoid-caption extraction.

HDBSCAN finds natural density modes in DINOv2 feature space and allocates the
IPC budget proportionally. K-Means is retained as the internal fallback for
classes where HDBSCAN collapses (<=1 mode) and as the sub-clustering strategy
when a parent mode receives more than one slot from the proportional
allocator. Each final cluster contributes its medoid's canonical caption as
the representative for Stage 4.
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

    # Distribution info (for representativeness-aware generation)
    weight: float = 0.0              # fraction of class samples in this mode
    density: float = 0.0             # compactness: 1 / mean_dist_to_centroid
    dino_centroid: np.ndarray | None = None  # DINOv2 centroid (768-dim)

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
    diagnostics: dict[str, Any] = field(default_factory=dict)


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
    archetype: str) -> list[ClusterMode]:
    """Shared mode extraction: find the DINOv2 medoid per cluster."""
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

        # Mode distribution info
        total_class_samples = len(class_dino)
        weight = len(member_local_indices) / total_class_samples if total_class_samples > 0 else 0.0
        mean_dist = float(distances_to_centroid.mean()) if len(distances_to_centroid) > 0 else 1.0
        density = 1.0 / max(mean_dist, 1e-8)

        modes.append(ClusterMode(
            cluster_id=k_idx,
            class_name=class_name,
            class_name_raw=class_name_raw,
            archetype=archetype,
            num_members=len(member_local_indices),
            medoid_index=medoid_global,
            medoid_record_id=samples[medoid_global]["record_id"],
            representative_caption=samples[medoid_global]["canonical_caption"],
            weight=weight,
            density=density,
            dino_centroid=dino_centroid,
            member_indices=member_global_indices))

    return modes


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
    seed: int) -> list[np.ndarray]:
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
    seed: int = 42) -> ClassClusterResult:
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
        class_name=class_name, class_name_raw=class_name_raw, archetype=archetype)

    return ClassClusterResult(
        class_name=class_name, class_name_raw=class_name_raw, archetype=archetype,
        num_samples=n_samples, ipc=ipc, num_clusters=len(modes), modes=modes,
        diagnostics={"branch": "kmeans", "kmeans_k": n_clusters})


def cluster_class_hdbscan(
    *,
    class_indices: list[int],
    dino_embeds: torch.Tensor,
    samples: list[dict[str, Any]],
    ipc: int,
    seed: int = 42,
    min_cluster_size: int = 15,
    min_samples: int = 3,
    pca_dim: int = 50) -> ClassClusterResult:
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
    n_noise = len(noise_indices)
    # Mode sizes as discovered by HDBSCAN (before any noise reassignment or
    # sub-clustering). Useful for attributing seed sensitivity: if
    # n_discovered >= ipc the final medoid set is seed-invariant (HDBSCAN is
    # deterministic and farthest-point sampling starts from index 0); if
    # n_discovered < ipc, K-Means sub-clustering kicks in and uses `seed`.
    hdbscan_discovered_sizes = [
        int((hdb_labels == m).sum()) for m in discovered_modes
    ]

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
        result = cluster_class_kmeans(
            class_indices=class_indices, dino_embeds=dino_embeds,
            samples=samples, ipc=ipc, seed=seed)
        result.diagnostics = {
            "branch": "hdbscan_fallback_kmeans",
            "hdbscan_n_discovered": n_discovered,
            "hdbscan_n_noise": n_noise,
            "hdbscan_discovered_sizes": hdbscan_discovered_sizes,
            "kmeans_k": ipc,
        }
        return result

    # Build per-mode member lists
    mode_members: dict[int, list[int]] = {}
    for m in discovered_modes:
        mode_members[m] = [i for i in range(n_samples) if hdb_labels[i] == m]
    mode_sizes = [len(mode_members[m]) for m in discovered_modes]

    took_farthest_point = n_discovered > ipc

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
        class_name=class_name, class_name_raw=class_name_raw, archetype=archetype)

    n_modes_subclustered = sum(1 for a in allocation if a > 1)
    branch = "hdbscan_farthest_point" if took_farthest_point else "hdbscan_proportional"
    diagnostics = {
        "branch": branch,
        "hdbscan_n_discovered": n_discovered,
        "hdbscan_n_noise": n_noise,
        "hdbscan_discovered_sizes": hdbscan_discovered_sizes,
        # After farthest-point absorption (if taken) or as-is
        "modes_after_fps_sizes": mode_sizes,
        # Slots per mode; any slot > 1 means K-Means sub-clustering was used
        # within that mode (uses `seed`)
        "allocation": allocation,
        "n_modes_subclustered": n_modes_subclustered,
    }

    return ClassClusterResult(
        class_name=class_name, class_name_raw=class_name_raw, archetype=archetype,
        num_samples=n_samples, ipc=ipc, num_clusters=len(modes), modes=modes,
        diagnostics=diagnostics)


def run_stage3_clustering(
    *,
    encode_dir: str | Path,
    output_dir: str | Path,
    ipc: int = 10,
    seed: int = 42,
    min_cluster_size: int = 15,
    min_samples: int = 3,
    pca_dim: int = 50) -> Stage3Result:
    """Run the full Stage 3 pipeline: load encoded tensors, cluster per class, extract modes.

    Uses HDBSCAN on DINOv2 features with K-Means fallback/sub-clustering.

    Args:
        encode_dir: Directory containing Stage 3A encode outputs.
        output_dir: Directory for Stage 3 mode outputs.
        ipc: Images per class — number of clusters per class.
        seed: Random seed (affects PCA + K-Means fallback / sub-clustering).
        min_cluster_size: HDBSCAN min_cluster_size.
        min_samples: HDBSCAN min_samples.
        pca_dim: PCA dimensions for HDBSCAN pre-processing (0 skips PCA).

    Returns:
        Stage3Result with the path to the modes index JSON.
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

    print(f"[Stage 3] Loaded {len(samples)} samples")

    # Group samples by class
    class_groups: dict[str, list[int]] = {}
    for i, sample in enumerate(samples):
        key = sample["class_name_raw"]
        if key not in class_groups:
            class_groups[key] = []
        class_groups[key].append(i)

    print(f"[Stage 3] Found {len(class_groups)} classes, IPC={ipc}, method=hdbscan")

    # Cluster each class on DINOv2 features. K-Means is still used internally
    # by the HDBSCAN path as the <=1-mode fallback and sub-clustering strategy.
    all_class_results = []

    for class_raw, indices in sorted(class_groups.items()):
        class_name = samples[indices[0]]["class_name"]
        print(f"[Stage 3] Clustering class '{class_name}' ({len(indices)} samples)...")

        class_result = cluster_class_hdbscan(
            class_indices=indices,
            dino_embeds=dino_embeds,
            samples=samples,
            ipc=ipc,
            seed=seed,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            pca_dim=pca_dim,
        )

        all_class_results.append(class_result)

    total_modes = sum(cr.num_clusters for cr in all_class_results)
    print(f"[Stage 3] Total modes: {total_modes}")

    # Rollup of HDBSCAN branch decisions per class
    branch_counts: dict[str, int] = {}
    for cr in all_class_results:
        b = cr.diagnostics.get("branch", "unknown")
        branch_counts[b] = branch_counts.get(b, 0) + 1
    n_sub_total = sum(
        cr.diagnostics.get("n_modes_subclustered", 0) for cr in all_class_results
    )
    print(f"[Stage 3] HDBSCAN branch rollup: {branch_counts}; "
          f"total parent modes sub-clustered via seeded K-Means: {n_sub_total}")


    # Save modes index
    modes_index_path = output_dir / "modes_index.json"

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
                "weight": round(mode.weight, 4),
                "density": round(mode.density, 4),
                "medoid_record_id": mode.medoid_record_id,
                "representative_caption": mode.representative_caption,
            })
            global_idx += 1

    summary = {
        "ipc": ipc,
        "num_classes": len(all_class_results),
        "total_modes": total_modes,
        "clustering": {
            "method": "hdbscan",
            "feature_space": "DINOv2 (dinov2_vitb14, 768-dim CLS token)",
            "caption_selection": "medoid (closest sample to DINOv2 cluster centroid)",
            "seed": seed,
            "hdbscan_min_cluster_size": min_cluster_size,
            "hdbscan_min_samples": min_samples,
            "hdbscan_pca_dim": pca_dim,
        },
        "branch_rollup": branch_counts,
        "source": {
            "encode_dir": str(encode_dir.resolve()),
        },
        "class_summary": [
            {
                "class_name": cr.class_name,
                "class_name_raw": cr.class_name_raw,
                "archetype": cr.archetype,
                "num_samples": cr.num_samples,
                "num_clusters": cr.num_clusters,
                "cluster_sizes": [m.num_members for m in cr.modes],
                "diagnostics": cr.diagnostics,
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
        modes_index_path=str(modes_index_path))
