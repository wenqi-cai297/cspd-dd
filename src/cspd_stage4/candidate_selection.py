"""Post-generation candidate selection for distilled dataset quality.

Generates N candidates per mode, scores each by:
  - S_proto: cosine similarity to the class prototype (mean of real DINOv2 features)
  - S_div: minimum cosine distance to already-selected images in the same class

No proxy classifier is used — scoring is architecture-agnostic, ensuring the
selected set generalizes across all eval architectures (ConvNet-6, ResNet-18,
ResNetAP-10) without biasing toward any one.

The balance between prototype faithfulness and diversity should be IPC-dependent:
  - Low IPC (10): favor S_proto (stay close to real distribution core)
  - High IPC (50+): favor S_div (spread out to cover more of the distribution)

Inspired by:
  - D³HR (ICML 2025): representativeness as explicit objective
  - IGDS (arXiv 2507.04619): IPC-dependent prototype/context balance
  - DAP (ICLR 2026): feature-space representativeness guidance
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from PIL import Image


class CandidateSelector:
    """Scores and selects the best candidate from a pool per mode.

    Architecture-agnostic: uses DINOv2 features for both prototype similarity
    and diversity scoring. No proxy classifier needed.
    """

    def __init__(
        self,
        device: str = "cuda",
        beta: float = 0.5,
    ):
        """Initialize the selector.

        Args:
            device: Torch device.
            beta: Weight for diversity score relative to prototype score.
                  0 = pure prototype faithfulness, higher = more diversity.
                  Recommended: 0.3 for IPC=10, 0.5 for IPC=20, 0.7 for IPC=50.
        """
        self.device = device
        self.beta = beta

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

        # Class prototypes: mean DINOv2 feature per class from real data
        self.class_prototypes: dict[str, torch.Tensor] = {}

        # Already-selected embeddings per class (for diversity scoring)
        self.selected_embeddings: dict[str, list[torch.Tensor]] = {}

    def build_prototypes(self, features: torch.Tensor, samples: list[dict]) -> None:
        """Build class prototypes (mean DINOv2 feature) from real data.

        Args:
            features: (N, 768) DINOv2 features of real training images.
            samples: encode_index samples with 'class_name_raw' field.
        """
        features = features.to(self.device)

        # Group by class
        class_features: dict[str, list[torch.Tensor]] = {}
        for i, sample in enumerate(samples):
            cls = sample.get("class_name_raw", "")
            if cls not in class_features:
                class_features[cls] = []
            class_features[cls].append(features[i])

        # Compute mean prototype per class
        for cls, feats in class_features.items():
            stacked = torch.stack(feats)
            prototype = stacked.mean(dim=0)
            prototype = F.normalize(prototype, dim=0)  # L2 normalize
            self.class_prototypes[cls] = prototype

        print(f"[CandidateSelector] Built prototypes for {len(self.class_prototypes)} classes")

    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encode a PIL image to DINOv2 CLS feature (768-dim), L2-normalized."""
        pixel_values = self.transform(image).unsqueeze(0).to(self.device)
        feature = self.dino_model(pixel_values).squeeze(0)  # (768,)
        return F.normalize(feature, dim=0)

    @torch.no_grad()
    def score_candidate(
        self,
        image: Image.Image,
        class_name_raw: str,
    ) -> tuple[float, float, float, torch.Tensor]:
        """Score a candidate image.

        Returns:
            (total_score, proto_score, div_score, embedding)
        """
        embedding = self.encode_image(image)

        # S_proto: cosine similarity to class prototype
        proto_score = 0.0
        prototype = self.class_prototypes.get(class_name_raw)
        if prototype is not None:
            proto_score = F.cosine_similarity(
                embedding.unsqueeze(0), prototype.unsqueeze(0), dim=1
            ).item()

        # S_div: minimum cosine distance to already-selected same-class images
        div_score = 1.0  # Default: maximally diverse if no prior selections
        selected = self.selected_embeddings.get(class_name_raw, [])
        if selected:
            selected_stack = torch.stack(selected)  # (M, 768)
            cos_sims = F.cosine_similarity(embedding.unsqueeze(0), selected_stack, dim=1)
            max_sim = cos_sims.max().item()
            div_score = 1.0 - max_sim  # Higher = more diverse

        total_score = proto_score + self.beta * div_score
        return total_score, proto_score, div_score, embedding

    def accept_candidate(self, class_name_raw: str, embedding: torch.Tensor):
        """Record an accepted candidate's embedding for future diversity scoring."""
        if class_name_raw not in self.selected_embeddings:
            self.selected_embeddings[class_name_raw] = []
        self.selected_embeddings[class_name_raw].append(embedding)
