from __future__ import annotations

"""Prompt construction for Stage 1 attribute extraction.

The prompt strategy is intentionally simple:
- a fixed system prompt that enforces JSON-only behavior,
- a per-sample user prompt that injects class metadata and the required schema.

This keeps the output format stable enough for automatic parsing.
"""

from cspd_stage1.schema import ATTRIBUTE_FIELDS, SampleRecord


# Global instruction shared across all samples.
# We explicitly ban reasoning text and hallucinated attributes because both are
# common failure modes for multimodal models in structured extraction tasks.
SYSTEM_PROMPT = (
    "You are a vision-language attribute extractor for dataset distillation. "
    "Inspect the given image and output JSON only. "
    "Never include reasoning. Never hallucinate invisible attributes. "
    "If a field is unclear, use 'unknown'. If not applicable, use 'not_applicable'."
)


def build_user_prompt(sample: SampleRecord) -> str:
    """Build the sample-specific prompt shown to the VLM.

    We include class metadata because it can help the model choose more sensible
    attribute phrases, but we still instruct it to stay image-grounded.
    """
    schema_lines = "\n".join([f'- {field}: short phrase' for field in ATTRIBUTE_FIELDS])
    return (
        f"Class name: {sample.class_name}\n"
        f"Class id: {sample.class_id}\n"
        "Return a JSON object with exactly these fields:\n"
        f"{schema_lines}\n"
        "Rules:\n"
        "- Output JSON only\n"
        "- Use short phrases, not sentences\n"
        "- Keep semantics image-grounded\n"
        "- Use unknown / not_applicable when necessary\n"
    )
