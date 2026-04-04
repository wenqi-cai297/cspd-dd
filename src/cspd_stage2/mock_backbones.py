from __future__ import annotations

"""Small local mock backbones for Stage 2 inspection/injection smoke tests.

These are not intended as research models. They only provide a stable torch module
shape so Stage 2 module inspection and adapter injection can be validated without
claiming a real FLUX Kontext loader exists.
"""

from typing import Any


def build_mock_transformer_backbone() -> Any:
    import torch

    class MockNorm1Context(torch.nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(width, width)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x)

    class MockAddedAttention(torch.nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()
            self.add_q_proj = torch.nn.Linear(width, width)
            self.add_k_proj = torch.nn.Linear(width, width)
            self.add_v_proj = torch.nn.Linear(width, width)
            self.to_add_out = torch.nn.Sequential(torch.nn.Linear(width, width))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.to_add_out(self.add_q_proj(x) + self.add_k_proj(x) + self.add_v_proj(x))

    class MockTransformerBlock(torch.nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()
            self.norm1_context = MockNorm1Context(width)
            self.attn = MockAddedAttention(width)
            self.ff_context = torch.nn.Sequential(torch.nn.Linear(width, width), torch.nn.GELU(), torch.nn.Linear(width, width))
            self.ff = torch.nn.Sequential(torch.nn.Linear(width, width), torch.nn.GELU(), torch.nn.Linear(width, width))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.norm1_context(x)
            x = self.attn(x)
            x = self.ff_context(x)
            return self.ff(x)

    class MockFluxTransformer(torch.nn.Module):
        def __init__(self, width: int = 16, depth: int = 2) -> None:
            super().__init__()
            self.context_embedder = torch.nn.Linear(width, width)
            self.time_text_embed = torch.nn.Linear(width, width)
            self.time_text_embed_timestep = torch.nn.Linear(width, width)
            self.transformer_blocks = torch.nn.ModuleList([MockTransformerBlock(width) for _ in range(depth)])
            self.output_proj = torch.nn.Linear(width, width)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.context_embedder(x) + self.time_text_embed(x) + self.time_text_embed_timestep(x)
            for block in self.transformer_blocks:
                x = block(x)
            return self.output_proj(x)

    class MockPipeline(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer = MockFluxTransformer()
            self.text_encoder = torch.nn.Linear(16, 16)
            self.vae = torch.nn.Linear(16, 16)
            self.image_encoder = torch.nn.Linear(16, 16)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.vae(self.transformer(x))

    return MockPipeline()
