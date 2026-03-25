from __future__ import annotations

"""Deterministic mock backend for local Stage 1 testing.

This backend exists only to validate pipeline plumbing:
- prompt wiring,
- schema validation,
- artifact generation,
- CLI behavior.

It does *not* inspect image content, so its outputs are useless for actual
research. That is deliberate.
"""

from cspd_stage1.schema import ATTRIBUTE_FIELDS, SampleRecord
from cspd_stage1.vlm.base import BaseVLMClient, VLMResponse


class MockVLMClient(BaseVLMClient):
    """Return a predictable attribute payload derived from sample metadata."""

    def extract_attributes(self, sample: SampleRecord, user_prompt: str, system_prompt: str) -> VLMResponse:
        # Fill every slot so the pipeline can be tested end-to-end without
        # depending on a real multimodal model.
        payload = {field: "unknown" for field in ATTRIBUTE_FIELDS}

        # The only meaningful signal we inject is the class name into `subject`.
        # This makes smoke-test outputs easy to eyeball.
        payload["subject"] = sample.class_name.lower().strip() or "unknown"
        return VLMResponse(payload=payload, raw_text=str(payload))
