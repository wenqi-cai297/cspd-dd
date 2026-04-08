from pathlib import Path
from types import SimpleNamespace

import torch

from cspd_stage2.training import Stage2TrainConfig, _freeze_stage2_modules, _run_real_pixart_train_step


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
        self.config = SimpleNamespace(scaling_factor=0.13025, shift_factor=0.0)

    def encode(self, pixel_values):
        pooled = torch.nn.functional.avg_pool2d(pixel_values, kernel_size=8, stride=8)
        latent = pooled.mean(dim=1, keepdim=True).repeat(1, 4, 1, 1)
        return FakeEncodeResult(latent)


class FakePixArtBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn1 = torch.nn.Linear(16, 16)
        self.attn2 = torch.nn.Linear(16, 16)
        self.ff = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.GELU(), torch.nn.Linear(16, 16))

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.attn1(hidden_states)
        hidden_states = hidden_states + self.attn2(hidden_states)
        hidden_states = self.ff(hidden_states)
        return hidden_states


class FakePixArtTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(out_channels=8, in_channels=4)
        self.input_projection = torch.nn.Linear(4, 16)
        self.caption_projection = torch.nn.Linear(16, 16)
        self.adaln_single = torch.nn.Linear(16, 16)
        self.transformer_blocks = torch.nn.ModuleList([FakePixArtBlock() for _ in range(2)])
        self.output_proj = torch.nn.Linear(16, 8)

    def forward(self, hidden_states, encoder_hidden_states=None, encoder_attention_mask=None, timestep=None, added_cond_kwargs=None, return_dict=True):
        del encoder_attention_mask, timestep, added_cond_kwargs
        batch, channels, height, width = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        hidden_states = self.input_projection(hidden_states)
        if encoder_hidden_states is not None:
            conditioning = encoder_hidden_states.mean(dim=1, keepdim=True)
            hidden_states = hidden_states + conditioning
        hidden_states = self.caption_projection(hidden_states)
        hidden_states = self.adaln_single(hidden_states)
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states)
        hidden_states = self.output_proj(hidden_states)
        hidden_states = hidden_states.reshape(batch, height, width, 8).permute(0, 3, 1, 2)
        if return_dict:
            return SimpleNamespace(sample=hidden_states)
        return (hidden_states,)


class FakeScheduler:
    def __init__(self):
        self.config = SimpleNamespace(num_train_timesteps=1000)

    def add_noise(self, latents, noise, timesteps):
        scale = timesteps.float().view(-1, 1, 1, 1) / float(self.config.num_train_timesteps)
        return latents + scale * noise


class FakePixArtPipeline(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = FakePixArtTransformer()
        self.vae = FakeVAE()
        self.text_encoder = torch.nn.Linear(1, 1)
        self.scheduler = FakeScheduler()
        self.last_encode_prompt_max_sequence_length = None

    def encode_prompt(self, prompt, do_classifier_free_guidance, device, num_images_per_prompt, max_sequence_length):
        del prompt, do_classifier_free_guidance, num_images_per_prompt
        self.last_encode_prompt_max_sequence_length = max_sequence_length
        batch = 2
        prompt_embeds = torch.zeros((batch, 5, 16), device=device, dtype=torch.float32)
        prompt_attention_mask = torch.ones((batch, 5), device=device, dtype=torch.long)
        negative_prompt_embeds = torch.zeros((batch, 5, 16), device=device, dtype=torch.float32)
        negative_prompt_attention_mask = torch.ones((batch, 5), device=device, dtype=torch.long)
        return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask


pipeline = FakePixArtPipeline()
config = Stage2TrainConfig(
    dataset_root='x',
    render_input='y',
    output_dir='z',
    backbone_name='PixArt-alpha/PixArt-Sigma-XL-2-512-MS',
    training_parameterization='lora',
    trainable_component_groups=['conditioning_transformer'],
)
selection = _freeze_stage2_modules(pipeline, config)
summary = selection['trainable_parameter_summary']
optimizer = torch.optim.AdamW((p for p in pipeline.transformer.parameters() if p.requires_grad), lr=1e-3)
batch = {
    'pixel_values': torch.randn(2, 3, 64, 64),
    'conditioning_text': ['caption a', 'caption b'],
}
loss = _run_real_pixart_train_step(
    pipeline=pipeline,
    transformer=pipeline.transformer,
    batch=batch,
    optimizer=optimizer,
    accelerator=None,
    device=torch.device('cpu'),
    train_dtype=torch.float32,
    memory_log_path=Path('tmp_stage2_pixart_smoke_memory.jsonl'),
    epoch=1,
    global_step=1,
    optimizer_step=1,
    config=config,
)
print({
    'loss': float(loss.detach().cpu().item()),
    'trainable_parameter_count': summary['trainable_parameter_count'],
    'lora_parameter_count': summary['lora_parameter_count'],
    'only_lora_parameters_trainable': summary['only_lora_parameters_trainable'],
    'adapter_injection_count': 0 if selection['adapter_injection'] is None else len(selection['adapter_injection'].injected_modules),
    'encode_prompt_max_sequence_length': pipeline.last_encode_prompt_max_sequence_length,
})
