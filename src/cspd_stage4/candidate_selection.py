"""Post-generation candidate selection for distilled dataset quality.

Generates N candidates per mode using different seeds, then selects the
best candidate based on a combined score of:
  - Discriminative score: DINOv2 linear probe confidence for the correct class
  - Diversity score: minimum cosine distance to already-selected images

Inspired by IGDS (arXiv 2507.04619) and Label-Consistent DGR (arXiv 2507.13074),
but implemented as post-generation filtering (no gradient injection into denoising).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image


class CandidateSelector:
    """Scores and selects the best candidate from a pool per mode.

    Uses a DINOv2 encoder + linear probe for discriminative scoring,
    and pairwise cosine distance for diversity scoring.
    """

    def __init__(
        self,
        class_names_raw: list[str],
        device: str = "cuda",
        beta: float = 0.5,
    ):
        """Initialize the selector.

        Args:
            class_names_raw: List of class folder names (used to map class_name_raw → class_id).
            device: Torch device.
            beta: Weight for diversity score. 0 = pure discriminative, 1 = pure diversity.
        """
        self.device = device
        self.beta = beta
        self.class_names_raw = sorted(class_names_raw)
        self.class_to_id = {name: i for i, name in enumerate(self.class_names_raw)}
        self.num_classes = len(self.class_names_raw)

        # DINOv2 encoder
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

        # Linear probe (trained on the fly from real data features)
        self.probe: torch.nn.Linear | None = None

        # Already-selected embeddings per class (for diversity scoring)
        self.selected_embeddings: dict[str, list[torch.Tensor]] = {}

    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encode a PIL image to DINOv2 CLS feature (768-dim)."""
        pixel_values = self.transform(image).unsqueeze(0).to(self.device)
        feature = self.dino_model(pixel_values)  # (1, 768)
        return feature.squeeze(0)  # (768,)

    def train_probe(self, features: torch.Tensor, labels: torch.Tensor, epochs: int = 100, lr: float = 0.01):
        """Train a linear probe on DINOv2 features from real data.

        Args:
            features: (N, 768) DINOv2 features of real training images.
            labels: (N,) integer class labels.
            epochs: Training epochs.
            lr: Learning rate.
        """
        dim = features.shape[1]
        self.probe = torch.nn.Linear(dim, self.num_classes).to(self.device)
        optimizer = torch.optim.SGD(self.probe.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

        features = features.to(self.device)
        labels = labels.to(self.device)

        self.probe.train()
        for _ in range(epochs):
            logits = self.probe(features)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.probe.eval()
        with torch.no_grad():
            logits = self.probe(features)
            acc = (logits.argmax(dim=1) == labels).float().mean().item()
        print(f"[CandidateSelector] Linear probe accuracy on real data: {acc:.1%}")

    @torch.no_grad()
    def score_candidate(
        self,
        image: Image.Image,
        class_name_raw: str,
    ) -> tuple[float, float, float, torch.Tensor]:
        """Score a candidate image.

        Returns:
            (total_score, disc_score, div_score, embedding)
        """
        embedding = self.encode_image(image)

        # Discriminative score: log P(correct class)
        disc_score = 0.0
        if self.probe is not None:
            logits = self.probe(embedding.unsqueeze(0))
            log_probs = F.log_softmax(logits, dim=1)
            class_id = self.class_to_id.get(class_name_raw, 0)
            disc_score = log_probs[0, class_id].item()

        # Diversity score: minimum cosine distance to already-selected same-class images
        div_score = 1.0  # Default: maximally diverse if no prior selections
        selected = self.selected_embeddings.get(class_name_raw, [])
        if selected:
            selected_stack = torch.stack(selected)  # (M, 768)
            cos_sims = F.cosine_similarity(embedding.unsqueeze(0), selected_stack, dim=1)
            max_sim = cos_sims.max().item()
            div_score = 1.0 - max_sim  # Higher = more diverse

        total_score = disc_score + self.beta * div_score
        return total_score, disc_score, div_score, embedding

    def accept_candidate(self, class_name_raw: str, embedding: torch.Tensor):
        """Record an accepted candidate's embedding for future diversity scoring."""
        if class_name_raw not in self.selected_embeddings:
            self.selected_embeddings[class_name_raw] = []
        self.selected_embeddings[class_name_raw].append(embedding)

    def reset_class(self, class_name_raw: str):
        """Reset selected embeddings for a class."""
        self.selected_embeddings.pop(class_name_raw, None)
