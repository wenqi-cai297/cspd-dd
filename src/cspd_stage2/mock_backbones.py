from __future__ import annotations

"""Small local mock backbones for Stage 2 inspection/injection smoke tests.

These are not intended as research models. They only provide a stable torch module
shape so Stage 2 module inspection and adapter injection can be validated without
claiming a real FLUX Kontext loader exists.
"""

from typing import Any


def build_mock_transformer_backbone() -> Any:
    import torch

    class MockAttentionBlock(torch.nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()
            self.attn = torch.nn.Linear(width, width)
            self.cross_attn = torch.nn.Linear(width, width)
            self.context = torch.nn.Linear(width, width)
            self.txt_proj = torch.nn.Linear(width, width)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.txt_proj(self.context(self.cross_attn(self.attn(x))))

    class MockBackbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer = torch.nn.ModuleDict(
                {
                    "block0": MockAttentionBlock(16),
                    "block1": MockAttentionBlock(16),
                }
            )
            self.context_embedder = torch.nn.Linear(16, 16)
            self.decoder = torch.nn.Linear(16, 16)
            self.vae = torch.nn.Linear(16, 16)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.transformer["block0"](x)
            x = self.transformer["block1"](x)
            x = self.context_embedder(x)
            x = self.decoder(x)
            return self.vae(x)

    return MockBackbone()
