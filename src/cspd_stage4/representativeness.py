"""Set-level representativeness scoring and refinement for distilled datasets.

Evaluates whether a generated set of IPC images adequately covers the real
data distribution of each class. Identifies coverage gaps and suggests which
modes should be regenerated.

Three scoring methods (all computed in DINOv2 feature space):
  - MMD (linear kernel, preferred per DAP Table 8): distributional distance
  - Moment matching (D³HR-style): mean + std + 0.1*skewness alignment
  - Coverage: fraction of real samples near a synthetic sample (diagnostic)

Paper references:
  - D³HR (ICML 2025): score = mean_diff + std_diff + 0.1*skew_diff in latent space
    (we adapt to DINOv2 space; their implementation uses DiT latent space)
  - DAP (ICLR 2026): kernel distance D_K(x,y) with linear kernel K(x,y)=x^Ty
    (linear > RBF per their Table 8; we use their linear kernel as default)
  - CoDA (ICLR 2026): core distribution alignment concept
  - DDOQ (ICLR 2026): dataset distillation as measure quantization
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
        kernel: str = "linear",
        bandwidth: float = 1.0,
    ) -> float:
        """Compute Maximum Mean Discrepancy between real and synthetic feature sets.

        Lower MMD = better distributional match.
        Linear kernel is preferred per DAP (ICLR 2026) Table 8:
        linear 66.4% > RBF 65.7-66.0% on ImageNette.

        Args:
            real_features: (N, D) real data features (L2-normalized DINOv2).
            synthetic_features: (M, D) synthetic data features.
            kernel: "linear" (preferred, per DAP) or "rbf".
            bandwidth: RBF kernel bandwidth (only used if kernel="rbf").

        Returns:
            MMD² estimate.
        """
        real = real_features.to(self.device)
        synth = synthetic_features.to(self.device)

        if kernel == "linear":
            # DAP-style linear kernel: K(x,y) = x^T y
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

    @torch.no_grad()
    def compute_moment_distance(
        self,
        real_features: torch.Tensor,
        synthetic_features: torch.Tensor,
        skewness_weight: float = 0.1,
    ) -> dict[str, float]:
        """Compute D³HR-style moment matching distance (mean + std + skewness).

        D³HR (ICML 2025) matches these statistics between real and synthetic
        samples in feature space. Skewness target is 0 (Gaussian assumption).

        The scoring formula from D³HR:
            score = mean_diff + std_diff + 0.1 * skew_diff

        Args:
            real_features: (N, D) real data features.
            synthetic_features: (M, D) synthetic data features.
            skewness_weight: Weight for skewness term (D³HR uses 0.1).

        Returns:
            Dict with mean_diff, std_diff, skew_diff, and total score.
        """
        real = real_features.to(self.device).float()
        synth = synthetic_features.to(self.device).float()

        # Mean difference
        real_mean = real.mean(dim=0)
        synth_mean = synth.mean(dim=0)
        mean_diff = float(torch.norm(synth_mean - real_mean))

        # Std difference
        real_std = real.std(dim=0)
        synth_std = synth.std(dim=0) if synth.shape[0] > 1 else torch.zeros_like(real_std)
        std_diff = float(torch.norm(synth_std - real_std))

        # Skewness (target = 0 per D³HR's Gaussian assumption)
        if synth.shape[0] > 2:
            synth_centered = synth - synth_mean
            synth_skew = (synth_centered ** 3).mean(dim=0) / (synth_std ** 3 + 1e-8)
            skew_diff = float(torch.norm(synth_skew))
        else:
            skew_diff = 0.0

        total = mean_diff + std_diff + skewness_weight * skew_diff

        return {
            "mean_diff": round(mean_diff, 6),
            "std_diff": round(std_diff, 6),
            "skew_diff": round(skew_diff, 6),
            "moment_score": round(total, 6),
        }

    @torch.no_grad()
    def select_set_greedy(
        self,
        real_features: torch.Tensor,
        candidates_per_mode: list[torch.Tensor],
        objective: str = "moments",
        skewness_weight: float = 0.1,
    ) -> tuple[list[int], list[float]]:
        """Greedy set-level selection: pick one candidate per mode to minimize
        set-level distance to the real class distribution.

        D³HR-style greedy matching adapted with a 1-per-mode constraint so
        Stage 3 mode structure is preserved (each mode still contributes
        exactly one image).

        Args:
            real_features: (N_real, D) real data features for one class.
            candidates_per_mode: list of length K (num modes), each entry is
                (N_cand, D) candidate features for that mode.
            objective: "moments" (D³HR mean+std+skew) or "mmd" (DAP linear kernel).
            skewness_weight: skew term weight for moments objective (D³HR uses 0.1).

        Returns:
            (selected_indices, scores) where selected_indices[i] is the chosen
            candidate index for mode i, and scores[i] is the running set score
            after adding that mode's pick.
        """
        real = F.normalize(real_features.to(self.device).float(), dim=1)
        real_mean = real.mean(dim=0)
        real_std = real.std(dim=0)

        selected_features: list[torch.Tensor] = []
        selected_indices: list[int] = []
        running_scores: list[float] = []

        for mode_idx, cand_feats in enumerate(candidates_per_mode):
            cand = F.normalize(cand_feats.to(self.device).float(), dim=1)
            if cand.shape[0] == 0:
                raise ValueError(f"Mode {mode_idx} has no candidates")

            best_score = float("inf")
            best_idx = 0

            for cand_idx in range(cand.shape[0]):
                tentative = torch.stack(selected_features + [cand[cand_idx]])
                if objective == "mmd":
                    k_rr = (real @ real.T).mean()
                    k_ss = (tentative @ tentative.T).mean()
                    k_rs = (real @ tentative.T).mean()
                    score = max(float(k_rr + k_ss - 2 * k_rs), 0.0)
                else:  # moments
                    synth_mean = tentative.mean(dim=0)
                    mean_diff = float(torch.norm(synth_mean - real_mean))
                    if tentative.shape[0] > 1:
                        synth_std = tentative.std(dim=0)
                        std_diff = float(torch.norm(synth_std - real_std))
                    else:
                        std_diff = float(torch.norm(real_std))
                    if tentative.shape[0] > 2:
                        synth_centered = tentative - synth_mean
                        synth_skew = (synth_centered ** 3).mean(dim=0) / (synth_std ** 3 + 1e-8)
                        skew_diff = float(torch.norm(synth_skew))
                    else:
                        skew_diff = 0.0
                    score = mean_diff + std_diff + skewness_weight * skew_diff

                if score < best_score:
                    best_score = score
                    best_idx = cand_idx

            selected_features.append(cand[best_idx])
            selected_indices.append(best_idx)
            running_scores.append(best_score)

        return selected_indices, running_scores

    def score_set(
        self,
        real_features: torch.Tensor,
        synthetic_features: torch.Tensor,
        coverage_threshold: float = 0.5,
    ) -> dict[str, Any]:
        """Comprehensive set-level scoring.

        Combines:
        - MMD (DAP-style, linear kernel): distributional distance
        - Moment matching (D³HR-style): mean + std + skewness alignment
        - Coverage: diagnostic metric for gap detection

        Returns:
            Dict with all metrics and composite representativeness score.
        """
        mmd = self.compute_mmd(real_features, synthetic_features, kernel="linear")
        moments = self.compute_moment_distance(real_features, synthetic_features)
        coverage = self.compute_coverage(real_features, synthetic_features, threshold=coverage_threshold)

        # Composite representativeness score (higher = better)
        mmd_score = max(0.0, 1.0 - mmd)
        # Normalize moment_score: lower is better, cap at reasonable range
        moment_score_norm = max(0.0, 1.0 - moments["moment_score"] / 10.0)
        repr_score = 0.4 * mmd_score + 0.4 * moment_score_norm + 0.2 * coverage["coverage_ratio"]

        return {
            "mmd": mmd,
            "moments": moments,
            "coverage": coverage,
            "representativeness_score": repr_score,
        }
