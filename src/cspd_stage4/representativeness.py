"""Set-level representativeness scoring and refinement for distilled datasets.

Evaluates whether a generated set of IPC images adequately covers the real
data distribution of each class. Identifies coverage gaps and suggests which
modes should be regenerated.

Two key metrics:
  - Coverage: fraction of real DINOv2 clusters that have at least one
    nearby synthetic image (within a distance threshold)
  - MMD (Maximum Mean Discrepancy): distributional distance between
    the synthetic set and real data in DINOv2 feature space

Inspired by:
  - D³HR (ICML 2025): group sampling with distribution statistics matching
  - CoDA (ICLR 2026): core distribution alignment
  - DAP (ICLR 2026): representativeness guidance in feature space
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Any


class RepresentativenessScorer:
    """Evaluates and improves the representativeness of a generated set."""

    def __init__(self, device: str = "cuda"):
        self.device = device

        # DINOv2 encoder (shared with CandidateSelector if both are used)
        self.dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        self.dino_model = self.dino_model.to(device)
        self.dino_model.eval()

        from torchvision import transforms as T
        self.transform = T.Compose([
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encode a PIL image to L2-normalized DINOv2 feature."""
        pixel_values = self.transform(image).unsqueeze(0).to(self.device)
        feature = self.dino_model(pixel_values).squeeze(0)
        return F.normalize(feature, dim=0)

    @torch.no_grad()
    def compute_mmd(
        self,
        real_features: torch.Tensor,
        synthetic_features: torch.Tensor,
        kernel: str = "rbf",
        bandwidth: float = 1.0,
    ) -> float:
        """Compute Maximum Mean Discrepancy between real and synthetic feature sets.

        Lower MMD = better distributional match.

        Args:
            real_features: (N, D) real data features (L2-normalized DINOv2).
            synthetic_features: (M, D) synthetic data features.
            kernel: "rbf" (Gaussian) or "linear".
            bandwidth: RBF kernel bandwidth.

        Returns:
            MMD² estimate (unbiased).
        """
        real = real_features.to(self.device)
        synth = synthetic_features.to(self.device)

        if kernel == "linear":
            k_rr = (real @ real.T).mean()
            k_ss = (synth @ synth.T).mean()
            k_rs = (real @ synth.T).mean()
        else:  # rbf
            def rbf_kernel(x, y):
                dists = torch.cdist(x, y, p=2) ** 2
                return torch.exp(-dists / (2 * bandwidth ** 2))

            k_rr = rbf_kernel(real, real).mean()
            k_ss = rbf_kernel(synth, synth).mean()
            k_rs = rbf_kernel(real, synth).mean()

        mmd_sq = k_rr + k_ss - 2 * k_rs
        return max(float(mmd_sq), 0.0)

    @torch.no_grad()
    def compute_coverage(
        self,
        real_features: torch.Tensor,
        synthetic_features: torch.Tensor,
        threshold: float = 0.5,
    ) -> dict[str, Any]:
        """Compute coverage: what fraction of real distribution is near a synthetic sample.

        Args:
            real_features: (N, D) real data features.
            synthetic_features: (M, D) synthetic data features.
            threshold: cosine similarity threshold for "covered".

        Returns:
            Dict with coverage ratio, uncovered indices, and per-sample details.
        """
        real = F.normalize(real_features.to(self.device), dim=1)
        synth = F.normalize(synthetic_features.to(self.device), dim=1)

        # For each real sample, find max cosine similarity to any synthetic sample
        sim_matrix = real @ synth.T  # (N, M)
        max_sims, nearest_synth = sim_matrix.max(dim=1)

        covered = max_sims >= threshold
        coverage_ratio = float(covered.float().mean())

        # Find the least-covered regions
        uncovered_mask = ~covered
        uncovered_indices = torch.where(uncovered_mask)[0].tolist()

        return {
            "coverage_ratio": coverage_ratio,
            "num_covered": int(covered.sum()),
            "num_uncovered": int(uncovered_mask.sum()),
            "total_real": len(real),
            "mean_max_similarity": float(max_sims.mean()),
            "min_max_similarity": float(max_sims.min()),
            "uncovered_indices": uncovered_indices[:50],  # Top 50 for reference
        }

    @torch.no_grad()
    def find_gap_modes(
        self,
        real_features: torch.Tensor,
        synthetic_features: torch.Tensor,
        mode_centroids: list[np.ndarray],
        threshold: float = 0.5,
    ) -> list[int]:
        """Find which modes have the worst coverage and should be regenerated.

        Args:
            real_features: (N, D) real data features for one class.
            synthetic_features: (M, D) synthetic data features for one class.
            mode_centroids: List of DINOv2 centroids per mode (from ClusterMode.dino_centroid).
            threshold: cosine similarity threshold.

        Returns:
            List of mode indices sorted by coverage gap (worst first).
        """
        real = F.normalize(real_features.to(self.device), dim=1)
        synth = F.normalize(synthetic_features.to(self.device), dim=1)

        # For each real sample, find max sim to synthetic set
        sim_matrix = real @ synth.T
        max_sims = sim_matrix.max(dim=1).values
        uncovered_mask = max_sims < threshold

        if not uncovered_mask.any():
            return []  # Full coverage

        # For each uncovered real sample, find which mode centroid it's closest to
        centroids = torch.stack([
            F.normalize(torch.from_numpy(c).float(), dim=0)
            for c in mode_centroids
        ]).to(self.device)

        uncovered_features = real[uncovered_mask]
        centroid_sims = uncovered_features @ centroids.T  # (U, K)
        nearest_modes = centroid_sims.argmax(dim=1)  # (U,)

        # Count uncovered samples per mode
        mode_gap_counts: dict[int, int] = {}
        for mode_idx in nearest_modes.tolist():
            mode_gap_counts[mode_idx] = mode_gap_counts.get(mode_idx, 0) + 1

        # Sort by gap count (worst first)
        sorted_modes = sorted(mode_gap_counts.keys(), key=lambda m: -mode_gap_counts[m])
        return sorted_modes

    def score_set(
        self,
        real_features: torch.Tensor,
        synthetic_features: torch.Tensor,
        bandwidth: float = 1.0,
        coverage_threshold: float = 0.5,
    ) -> dict[str, Any]:
        """Comprehensive set-level scoring.

        Returns:
            Dict with MMD, coverage, and overall representativeness score.
        """
        mmd = self.compute_mmd(real_features, synthetic_features, bandwidth=bandwidth)
        coverage = self.compute_coverage(real_features, synthetic_features, threshold=coverage_threshold)

        # Composite representativeness score (higher = better)
        # Normalize MMD to [0, 1] range approximately (empirical)
        mmd_score = max(0.0, 1.0 - mmd)
        repr_score = 0.5 * mmd_score + 0.5 * coverage["coverage_ratio"]

        return {
            "mmd": mmd,
            "mmd_score": mmd_score,
            "coverage": coverage,
            "representativeness_score": repr_score,
        }
