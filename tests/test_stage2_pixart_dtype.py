from __future__ import annotations

import torch

from cspd_stage2.backbone import LoRALinearAdapter
from cspd_stage2.training import (
    Stage2TrainConfig,
    _prepare_pixart_forward_inputs,
    _resolve_lora_master_weight_dtype,
    _resolve_pixart_partial_full_update_fp32_exclude_patterns,
    _upcast_trainable_parameters_,
)


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


def test_pixart_partial_full_update_keeps_adaln_single_at_boundary_dtype() -> None:
    transformer = _DummyPixArtTransformer(
        pos_dtype=torch.float16,
        caption_dtype=torch.float16,
        adaln_dtype=torch.float16,
    )
    for parameter in transformer.parameters():
        parameter.requires_grad = True

    summary = _upcast_trainable_parameters_(transformer, dtype=torch.float32, exclude_patterns=["adaln_single.*"])

    assert transformer.pos_embed.bias.dtype == torch.float32
    assert transformer.caption_projection.bias.dtype == torch.float32
    assert transformer.adaln_single.bias.dtype == torch.float16
    assert summary["excluded_parameter_patterns"] == ["adaln_single.*"]
    assert summary["skipped_parameter_count"] == 1
    assert summary["skipped_parameter_names_sample"] == ["adaln_single.bias"]


def test_resolve_pixart_partial_full_update_fp32_exclude_patterns() -> None:
    config = Stage2TrainConfig(
        dataset_root="/tmp/dataset",
        render_input="/tmp/render.jsonl",
        output_dir="/tmp/output",
        backbone_name="PixArt-alpha/PixArt-Sigma-XL-2-512-MS",
        training_parameterization="full",
        trainable_component_groups=["conditioning_transformer"],
        full_update_fp32_for_pixart=True,
    )

    assert _resolve_pixart_partial_full_update_fp32_exclude_patterns(config=config) == ["adaln_single.*"]


def test_resolve_pixart_lora_master_weight_dtype_defaults_to_fp32() -> None:
    config = Stage2TrainConfig(
        dataset_root="/tmp/dataset",
        render_input="/tmp/render.jsonl",
        output_dir="/tmp/output",
        backbone_name="PixArt-alpha/PixArt-Sigma-XL-2-512-MS",
        training_parameterization="lora",
        trainable_component_groups=["full_transformer"],
        lora_fp32_for_pixart=True,
    )

    assert _resolve_lora_master_weight_dtype(config=config) == "float32"


def test_lora_linear_adapter_supports_fp32_master_weights_over_fp16_base() -> None:
    base = torch.nn.Linear(8, 4, bias=False, dtype=torch.float16)
    adapter = LoRALinearAdapter(base, rank=2, alpha=2.0, dropout=0.0, adapter_dtype="float32")
    inputs = torch.randn(3, 8, dtype=torch.float16)

    outputs = adapter(inputs)

    assert adapter.base_layer.weight.dtype == torch.float16
    assert adapter.lora_A.weight.dtype == torch.float32
    assert adapter.lora_B.weight.dtype == torch.float32
    assert outputs.dtype == torch.float16
    loss = outputs.float().sum()
    loss.backward()
    assert adapter.lora_A.weight.grad is not None
    assert adapter.lora_B.weight.grad is not None
    assert adapter.lora_A.weight.grad.dtype == torch.float32
    assert adapter.lora_B.weight.grad.dtype == torch.float32
