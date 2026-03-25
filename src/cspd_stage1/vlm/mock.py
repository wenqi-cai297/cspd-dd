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

from cspd_stage1.schema import SampleRecord
from cspd_stage1.vlm.base import BaseVLMClient, VLMResponse


class MockVLMClient(BaseVLMClient):
    """Return a predictable attribute payload derived from sample metadata."""

    def extract_attributes(self, sample: SampleRecord, user_prompt: str, system_prompt: str) -> VLMResponse:
        # Fill every requested slot so the pipeline can be tested end-to-end
        # without depending on a real multimodal model.
        attributes = {field: "unknown" for field in sample.slot_schema}

        # Inject the class name into the first slot to make smoke-test outputs
        # easy to inspect. This is just a deterministic placeholder.
        if sample.slot_schema:
            attributes[sample.slot_schema[0]] = sample.class_name.lower().strip() or "unknown"

        payload = {
            "archetype": sample.archetype,
            "attributes": attributes,
        }
        return VLMResponse(payload=payload, raw_text=str(payload))
