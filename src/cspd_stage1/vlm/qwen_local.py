from __future__ import annotations

"""Local Qwen2.5-VL backend for Stage 1 attribute extraction.

This backend is meant for server-side execution where the model is loaded from
Hugging Face and run on local GPUs. The model/processor are loaded lazily in the
constructor so the rest of the pipeline can stay backend-agnostic.
"""

from pathlib import Path

from PIL import Image

from cspd_stage1.schema import SampleRecord
from cspd_stage1.vlm.base import BaseVLMClient, VLMResponse
from cspd_stage1.vlm.json_utils import parse_json_object


class QwenLocalVLMClient(BaseVLMClient):
    """Run Stage 1 extraction with a locally hosted Qwen2.5-VL checkpoint."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype: str = "float16",
        device_map: str = "auto",
        use_fast_processor: bool = True,
        max_new_tokens: int = 256,
    ) -> None:
        self.model_name = model_name
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.use_fast_processor = use_fast_processor
        self.max_new_tokens = max_new_tokens

        # Import transformers/torch lazily so the mock backend and lightweight
        # local checks do not require heavyweight GPU dependencies.
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self._torch = torch
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=self._resolve_torch_dtype(torch_dtype),
            device_map=self.device_map,
        )
        self._processor = AutoProcessor.from_pretrained(
            self.model_name,
            use_fast=self.use_fast_processor,
        )

    def extract_attributes(self, sample: SampleRecord, user_prompt: str, system_prompt: str) -> VLMResponse:
        """Run the local Qwen-VL model on one image and parse the JSON result."""
        image_path = Path(sample.image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self._processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self._model.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

        generated_ids = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        payload = parse_json_object(output_text)
        return VLMResponse(payload=payload, raw_text=output_text)

    def _resolve_torch_dtype(self, torch_dtype: str):
        """Map a user-friendly dtype string into a torch dtype object."""
        normalized = torch_dtype.strip().lower()
        mapping = {
            "float16": self._torch.float16,
            "fp16": self._torch.float16,
            "bfloat16": self._torch.bfloat16,
            "bf16": self._torch.bfloat16,
            "float32": self._torch.float32,
            "fp32": self._torch.float32,
        }
        if normalized not in mapping:
            raise ValueError(f"Unsupported torch dtype: {torch_dtype}")
        return mapping[normalized]
