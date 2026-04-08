from __future__ import annotations

import torch

from cspd_stage2.training import _prepare_pixart_forward_inputs


class _BiasModule(torch.nn.Module):
    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(1, dtype=dtype))


class _DummyPixArtTransformer(torch.nn.Module):
    def __init__(self, *, pos_dtype: torch.dtype, caption_dtype: torch.dtype, adaln_dtype: torch.dtype) -> None:
        super().__init__()
        self.pos_embed = _BiasModule(pos_dtype)
        self.caption_projection = _BiasModule(caption_dtype)
        self.adaln_single = _BiasModule(adaln_dtype)


def test_prepare_pixart_forward_inputs_keeps_partial_path_boundary_dtypes() -> None:
    transformer = _DummyPixArtTransformer(
        pos_dtype=torch.float16,
        caption_dtype=torch.float32,
        adaln_dtype=torch.float32,
    )
    noisy_latents = torch.randn(2, 4, 64, 64, dtype=torch.float16)
    prompt_embeds = torch.randn(2, 300, 4096, dtype=torch.float16)

    prepared = _prepare_pixart_forward_inputs(
        transformer=transformer,
        noisy_latents=noisy_latents,
        prompt_embeds=prompt_embeds,
        device=torch.device("cpu"),
        train_dtype=torch.float16,
    )

    assert prepared["hidden_states"].dtype == torch.float16
    assert prepared["encoder_hidden_states"].dtype == torch.float32
    assert prepared["added_cond_kwargs"]["resolution"].dtype == torch.float32
    assert prepared["added_cond_kwargs"]["aspect_ratio"].dtype == torch.float32
    assert prepared["dtype_plan"] == {
        "hidden_states": "float16",
        "encoder_hidden_states": "float32",
        "added_cond_kwargs": "float32",
    }


def test_prepare_pixart_forward_inputs_preserves_full_transformer_fp32_path() -> None:
    transformer = _DummyPixArtTransformer(
        pos_dtype=torch.float32,
        caption_dtype=torch.float32,
        adaln_dtype=torch.float32,
    )
    noisy_latents = torch.randn(1, 4, 64, 64, dtype=torch.float16)
    prompt_embeds = torch.randn(1, 300, 4096, dtype=torch.float16)

    prepared = _prepare_pixart_forward_inputs(
        transformer=transformer,
        noisy_latents=noisy_latents,
        prompt_embeds=prompt_embeds,
        device=torch.device("cpu"),
        train_dtype=torch.float16,
    )

    assert prepared["hidden_states"].dtype == torch.float32
    assert prepared["encoder_hidden_states"].dtype == torch.float32
    assert prepared["added_cond_kwargs"]["resolution"].dtype == torch.float32
    assert prepared["added_cond_kwargs"]["aspect_ratio"].dtype == torch.float32
