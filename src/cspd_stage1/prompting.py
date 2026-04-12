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

# Per-slot guidance: tells the VLM what kind of value each slot expects.
# This prevents common mistakes like writing colors for backgrounds or bare on/off for states.
SLOT_GUIDANCE: dict[str, str] = {
    # --- Identity / type slots ---
    "species_or_category": "specific species or subcategory name",
    "plant_or_fungus_type": "specific plant or fungus name",
    "food_or_drink_type": "specific food or drink name",
    "vehicle_type": "specific vehicle type",
    "wearable_type": "specific garment or accessory name",
    "furniture_type": "specific furniture name",
    "container_type": "specific container name",
    "tool_type": "specific tool name",
    "device_or_appliance_type": "specific device or appliance name",
    "instrument_type": "specific instrument name",
    "weapon_type": "specific weapon name",
    "sports_or_toy_type": "specific sports equipment or toy name",
    "household_object_type": "specific household item name",
    "structure_or_building_type": "specific building or structure type",
    "scene_or_landform_type": "specific scene or landform type",
    "person_type_or_role": "role or occupation, e.g. athlete, diver, bride",
    "text_or_media_object_type": "specific media object type",
    "decorative_or_symbolic_object_type": "specific decorative or symbolic object name",
    # --- Color ---
    "color": "dominant color(s), e.g. red, dark blue, silver and black",
    "color_or_pattern": "color and/or pattern, e.g. brown spotted, black and white striped",
    "dominant_color_or_tone": "overall color tone of the scene",
    # --- Material ---
    "material": "primary material, e.g. wood, metal, plastic, fabric",
    "material_or_finish": "material or surface finish, e.g. brushed steel, matte plastic",
    "material_or_texture": "material or texture, e.g. leather, knitted wool, denim",
    "material_or_surface": "exterior material, e.g. brick, stone, concrete, glass",
    # --- Shape / structure ---
    "shape_or_structure": "distinctive shape features visible in THIS image, e.g. long handle with blade, rectangular with rounded edges",
    "shape_or_growth_form": "growth form, e.g. bushy, climbing vine, mushroom cap",
    "shape_or_style": "garment style or cut, e.g. A-line, fitted, oversized",
    "architectural_style_or_form": "architectural style, e.g. gothic, baroque, modern, colonial",
    "scale_or_extent": "apparent size, e.g. small, medium, large, towering",
    # --- State / pose / action ---
    "pose_or_state": "what the animal is doing, e.g. swimming, being held by person, curled up sleeping, running through grass",
    "state_or_action": "vehicle state, e.g. driving on highway, parked in driveway, being loaded",
    "operating_state_or_display_state": "device state with detail, e.g. playing music with display lit, idle with closed lid, recording. Do NOT write just 'on' or 'off'",
    "playing_state_or_pose": "instrument state, e.g. being played by musician, resting in case, hanging on wall",
    "activity_or_usage_state": "usage state, e.g. in play on field, stored in bag, being inflated",
    "usage_state": "current usage, e.g. occupied, empty and open, being cleaned",
    "usage_or_display_state": "usage or display state, e.g. sheathed, mounted on wall, being wielded",
    "wearing_state_or_pose": "how it is worn/displayed, e.g. worn by person, folded on shelf, hanging on rack",
    "fill_state_or_contents_visibility": "container fullness, e.g. empty, half-full with liquid, overflowing with items",
    "growth_state": "growth phase, e.g. flowering, mature, seedling, dried",
    "body_pose_or_action": "person's pose, e.g. standing, diving underwater, running",
    "physical_or_display_state": "physical condition, e.g. opened, sealed, weathered, torn",
    # --- Background / environment / context ---
    "background_or_habitat": "scene or place WHERE the subject is, e.g. grassy field, lake shore, wooden table, veterinary clinic. Do NOT write just a color",
    "background_or_context": "scene or setting WHERE the object is, e.g. kitchen counter, store shelf, workshop bench, car dashboard. Do NOT write just a color",
    "background_or_room_context": "room or setting, e.g. living room, office desk, outdoor patio",
    "background_or_activity_context": "activity setting, e.g. concert stage, underwater reef, sports field",
    "environment": "surrounding environment, e.g. urban street, rural farmland, parking lot, forest road",
    "surrounding_environment": "surroundings, e.g. city skyline, countryside, courtyard with trees",
    "container_or_context": "container or setting, e.g. ceramic plate, wooden cutting board, glass display case",
    "display_or_usage_context": "display setting, e.g. museum shelf, living room mantle, outdoor garden",
    "vegetation_or_natural_context": "vegetation, e.g. dense forest, sparse grassland, tropical plants",
    # --- Viewpoint ---
    "viewpoint": "camera angle, e.g. front view, side view, top-down view, close-up view, low angle view",
    # --- Salient parts / details ---
    "body_trait": "distinctive body feature visible in image, e.g. long floppy ears, bushy tail, sharp claws",
    "visible_part": "visible plant part, e.g. leaves, petals, stem, cap",
    "salient_part_or_focus": "most prominent detail, e.g. fins and scales, paws, antlers, beak",
    "salient_part_or_accessory": "notable part or accessory, e.g. side mirror, exhaust pipe, logo on door",
    "salient_structural_part": "notable architectural detail, e.g. bell tower, stained glass window, dome",
    "salient_topping_or_ingredient": "visible topping or ingredient, e.g. melted cheese, fresh basil, whipped cream",
    "salient_geographic_feature": "notable geographic feature, e.g. waterfall, rock formation, sandy shore",
    "ornamentation_or_symbolic_trait": "decorative detail, e.g. carved pattern, painted motif, religious symbol",
    "held_object_or_equipment": "object held or worn, e.g. guitar, surfboard, camera",
    "clothing_or_gear": "clothing description, e.g. wetsuit, formal suit, hiking gear",
    "preparation_or_serving_style": "how food is prepared/served, e.g. grilled, sliced, served on banana leaf",
    "content_or_symbol_type": "content shown, e.g. text headline, crossword grid, street name",
    "layout_or_format": "layout, e.g. single page, grid layout, scrolling list",
    "terrain_or_surface_trait": "terrain, e.g. sandy, rocky, muddy, snow-covered",
    "weather_or_water_state": "weather or water, e.g. sunny, overcast, calm water, rough waves",
}

# Fallback for slots not in SLOT_GUIDANCE
_DEFAULT_GUIDANCE = "short descriptive phrase based on what you see in the image"


def build_user_prompt(sample: SampleRecord) -> str:
    """Build the sample-specific prompt shown to the VLM."""
    template_payload = {
        "archetype": sample.archetype,
        "attributes": {
            field: SLOT_GUIDANCE.get(field, _DEFAULT_GUIDANCE)
            for field in sample.slot_schema
        },
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
        "- Output JSON only, no markdown or code fences\n"
        "- Keep all JSON keys double-quoted\n"
        "- Keep the archetype unchanged\n"
        "- Fill only the requested attribute slots\n"
        "- Use short phrases (2-5 words), not full sentences\n"
        "- Describe ONLY what you can see in this specific image\n"
        "- For background/environment slots: describe the PLACE or SCENE, not just a color\n"
        "- For state/pose slots: describe the specific ACTION or CONDITION, not just 'on'/'off'\n"
        "- Each slot should have a SINGLE value, not a comma-separated list\n"
        "- Use 'unknown' only when truly not visible; prefer a coarse description over 'unknown'\n"
    )
