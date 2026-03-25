from __future__ import annotations

"""Prompt construction for Stage 1 attribute extraction.

The prompt is class-adaptive:
- a fixed system prompt enforces JSON-only behavior,
- a per-sample user prompt includes the readable class name,
- the requested slot schema depends on the class archetype.

Crucially, the schema is shown as an actual JSON template instead of a bullet
list. This reduces the chance that the VLM copies a pseudo-list format such as
`- key: value` instead of emitting valid JSON.
"""

import json

from cspd_stage1.schema import SampleRecord


SYSTEM_PROMPT = (
    "You are a vision-language attribute extractor for dataset distillation. "
    "Inspect the given image and output JSON only. "
    "Never include reasoning. Never hallucinate invisible attributes. "
    "If a field is unclear, use 'unknown'. If not applicable, use 'not_applicable'."
)


def build_user_prompt(sample: SampleRecord) -> str:
    """Build the sample-specific prompt shown to the VLM."""
    template_payload = {
        "archetype": sample.archetype,
        "attributes": {field: "short phrase" for field in sample.slot_schema},
    }
    template_json = json.dumps(template_payload, ensure_ascii=False, indent=2)
    return (
        f"Class name: {sample.class_name}\n"
        f"Original class label: {sample.class_name_raw}\n"
        f"Class id: {sample.class_id}\n"
        f"Semantic archetype: {sample.archetype}\n"
        "Return JSON only in exactly this structure:\n"
        f"{template_json}\n"
        "Rules:\n"
        "- Output JSON only\n"
        "- Do not use markdown or code fences\n"
        "- Do not use bullet points like '- key: value'\n"
        "- Keep all JSON keys double-quoted\n"
        "- Keep the archetype unchanged\n"
        "- Fill only the requested attribute slots\n"
        "- Use short phrases, not sentences\n"
        "- Keep semantics image-grounded\n"
        "- Use unknown / not_applicable when necessary\n"
    )
