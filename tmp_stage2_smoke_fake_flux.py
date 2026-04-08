from types import SimpleNamespace

import torch

from cspd_stage2.training import _freeze_stage2_modules, _run_real_flux_train_step, Stage2TrainConfig


class FakeLatentDist:
    def __init__(self, tensor):
        self._tensor = tensor

    def sample(self):
        return self._tensor


class FakeEncodeResult:
    def __init__(self, tensor):
        self.latent_dist = FakeLatentDist(tensor)


class FakeVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(shift_factor=0.0, scaling_factor=1.0)

    def encode(self, pixel_values):
        pooled = torch.nn.functional.avg_pool2d(pixel_values, kernel_size=8, stride=8)
        latent = pooled.mean(dim=1, keepdim=True).repeat(1, 16, 1, 1)
        return FakeEncodeResult(latent)


class FakeAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.add_q_proj = torch.nn.Linear(64, 64)
        self.add_k_proj = torch.nn.Linear(64, 64)
        self.add_v_proj = torch.nn.Linear(64, 64)
        self.to_add_out = torch.nn.Sequential(torch.nn.Linear(64, 64))


class FakeBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = FakeAttention()
        self.norm1_context = torch.nn.Linear(64, 64)
        self.ff_context = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.GELU(), torch.nn.Linear(64, 64))


class FakeTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(guidance_embeds=True)
        self.context_embedder = torch.nn.Linear(64, 64)
        self.time_text_embed = torch.nn.Linear(64, 64)
        self.time_text_embed_timestep = torch.nn.Linear(64, 64)
        self.transformer_blocks = torch.nn.ModuleList([FakeBlock() for _ in range(2)])
        self.proj = torch.nn.Linear(64, 64)

    def forward(self, hidden_states, encoder_hidden_states=None, pooled_projections=None, timestep=None, img_ids=None, txt_ids=None, guidance=None, return_dict=True, **kwargs):
        del encoder_hidden_states, pooled_projections, timestep, img_ids, txt_ids, guidance, kwargs
        hidden_states = self.context_embedder(hidden_states) + self.time_text_embed(hidden_states) + self.time_text_embed_timestep(hidden_states)
        for block in self.transformer_blocks:
            hidden_states = hidden_states + block.attn.add_q_proj(hidden_states)
            hidden_states = hidden_states + block.attn.add_k_proj(hidden_states)
            hidden_states = hidden_states + block.attn.add_v_proj(hidden_states)
            hidden_states = block.attn.to_add_out(hidden_states)
            hidden_states = block.norm1_context(hidden_states)
            hidden_states = block.ff_context(hidden_states)
        out = self.proj(hidden_states)
        if return_dict:
            return SimpleNamespace(sample=out)
        return (out,)


class FakeTextEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))


class FakePipeline(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = FakeTransformer()
        self.text_encoder = FakeTextEncoder()
        self.text_encoder_2 = FakeTextEncoder()
        self.vae = FakeVAE()
        self.image_encoder = FakeTextEncoder()

    def encode_prompt(self, prompt, prompt_2, device, num_images_per_prompt, max_sequence_length):
        del prompt_2, num_images_per_prompt, max_sequence_length
        batch = len(prompt)
        prompt_embeds = torch.zeros((batch, 4, 64), device=device, dtype=torch.float32)
        pooled = torch.zeros((batch, 64), device=device, dtype=torch.float32)
        text_ids = torch.zeros((4, 3), device=device, dtype=torch.float32)
        return prompt_embeds, pooled, text_ids

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        return latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2).permute(0, 2, 4, 1, 3, 5).reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

    @staticmethod
    def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
        del batch_size
        latent_image_ids = torch.zeros(height, width, 3, device=device, dtype=dtype)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height, device=device, dtype=dtype)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width, device=device, dtype=dtype)[None, :]
        return latent_image_ids.reshape(height * width, 3)


def run_case(parameterization: str):
    pipeline = FakePipeline()
    config = Stage2TrainConfig(
        dataset_root='x',
        render_input='y',
        output_dir='z',
        training_parameterization=parameterization,
        trainable_component_groups=['conditioning_transformer'] if parameterization == 'lora' else ['full_transformer'],
    )
    selection = _freeze_stage2_modules(pipeline, config)
    summary = selection['trainable_parameter_summary']
    optimizer = torch.optim.AdamW((p for p in pipeline.transformer.parameters() if p.requires_grad), lr=1e-3)
    batch = {
        'pixel_values': torch.randn(2, 3, 64, 64),
        'conditioning_text': ['caption a', 'caption b'],
    }
    loss = _run_real_flux_train_step(
        pipeline=pipeline,
        transformer=pipeline.transformer,
        batch=batch,
        optimizer=optimizer,
        accelerator=None,
        device=torch.device('cpu'),
        train_dtype=torch.float32,
        resolution=64,
        memory_log_path=Path('tmp_stage2_smoke_memory.jsonl'),
        component_move_log_path=None,
        epoch=1,
        global_step=1,
        optimizer_step=1,
        keep_frozen_modules_on_cpu_until_needed=False,
        offload_frozen_modules_after_step=False,
        move_state={'component_move_events': [], 'component_move_failures': [], 'last_component_move_attempt': None},
    )
    return {
        'parameterization': parameterization,
        'loss': float(loss.detach().cpu().item()),
        'trainable_parameter_count': summary['trainable_parameter_count'],
        'lora_parameter_count': summary['lora_parameter_count'],
        'only_lora_parameters_trainable': summary['only_lora_parameters_trainable'],
        'adapter_injection_count': 0 if selection['adapter_injection'] is None else len(selection['adapter_injection'].injected_modules),
    }


from pathlib import Path

print(run_case('full'))
print(run_case('lora'))
