from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TemplateSpec:
    archetype: str
    template_id: str
    anchor_slot: str
    pre_anchor_slots: tuple[str, ...]
    post_anchor_slots: tuple[str, ...]
    slot_prefixes: dict[str, str]
    drop_if_unknown: tuple[str, ...]
    optional_slots: tuple[str, ...]
    fallback_anchor: str | None = None


TEMPLATE_SPECS: dict[str, TemplateSpec] = {
    "animal": TemplateSpec(
        archetype="animal",
        template_id="animal_basic_v1",
        anchor_slot="species_or_category",
        pre_anchor_slots=("color_or_pattern", "body_trait"),
        post_anchor_slots=("pose_or_state", "background_or_habitat", "viewpoint", "salient_part_or_focus"),
        slot_prefixes={"pose_or_state": "", "background_or_habitat": "in", "viewpoint": "from", "salient_part_or_focus": "with"},
        drop_if_unknown=("color_or_pattern", "body_trait", "pose_or_state", "background_or_habitat", "viewpoint", "salient_part_or_focus"),
        optional_slots=("color_or_pattern", "body_trait", "pose_or_state", "background_or_habitat", "viewpoint", "salient_part_or_focus"),
    ),
    "plant_or_fungus": TemplateSpec(
        archetype="plant_or_fungus",
        template_id="plant_or_fungus_basic_v1",
        anchor_slot="plant_or_fungus_type",
        pre_anchor_slots=("color", "shape_or_growth_form"),
        post_anchor_slots=("visible_part", "growth_state", "background_or_habitat", "viewpoint"),
        slot_prefixes={"visible_part": "showing", "growth_state": "in", "background_or_habitat": "in", "viewpoint": "from"},
        drop_if_unknown=("color", "shape_or_growth_form", "visible_part", "growth_state", "background_or_habitat", "viewpoint"),
        optional_slots=("color", "shape_or_growth_form", "visible_part", "growth_state", "background_or_habitat", "viewpoint"),
    ),
    "food_and_drink": TemplateSpec(
        archetype="food_and_drink",
        template_id="food_and_drink_basic_v1",
        anchor_slot="food_or_drink_type",
        pre_anchor_slots=("color", "shape_or_structure"),
        post_anchor_slots=("preparation_or_serving_style", "container_or_context", "viewpoint", "salient_topping_or_ingredient"),
        slot_prefixes={"preparation_or_serving_style": "served", "container_or_context": "in", "viewpoint": "from", "salient_topping_or_ingredient": "with"},
        drop_if_unknown=("color", "shape_or_structure", "preparation_or_serving_style", "container_or_context", "viewpoint", "salient_topping_or_ingredient"),
        optional_slots=("color", "shape_or_structure", "preparation_or_serving_style", "container_or_context", "viewpoint", "salient_topping_or_ingredient"),
    ),
    "vehicle": TemplateSpec(
        archetype="vehicle",
        template_id="vehicle_basic_v1",
        anchor_slot="vehicle_type",
        pre_anchor_slots=("color", "shape_or_structure"),
        post_anchor_slots=("state_or_action", "environment", "viewpoint", "salient_part_or_accessory"),
        slot_prefixes={"state_or_action": "", "environment": "in", "viewpoint": "from", "salient_part_or_accessory": "with"},
        drop_if_unknown=("color", "shape_or_structure", "state_or_action", "environment", "viewpoint", "salient_part_or_accessory"),
        optional_slots=("color", "shape_or_structure", "state_or_action", "environment", "viewpoint", "salient_part_or_accessory"),
    ),
    "clothing_and_wearable": TemplateSpec(
        archetype="clothing_and_wearable",
        template_id="clothing_and_wearable_basic_v1",
        anchor_slot="wearable_type",
        pre_anchor_slots=("color_or_pattern", "material_or_texture", "shape_or_style"),
        post_anchor_slots=("wearing_state_or_pose", "background_or_context", "viewpoint"),
        slot_prefixes={"wearing_state_or_pose": "", "background_or_context": "in", "viewpoint": "from"},
        drop_if_unknown=("color_or_pattern", "material_or_texture", "shape_or_style", "wearing_state_or_pose", "background_or_context", "viewpoint"),
        optional_slots=("color_or_pattern", "material_or_texture", "shape_or_style", "wearing_state_or_pose", "background_or_context", "viewpoint"),
    ),
    "furniture": TemplateSpec(
        archetype="furniture",
        template_id="furniture_basic_v1",
        anchor_slot="furniture_type",
        pre_anchor_slots=("color", "material", "shape_or_structure"),
        post_anchor_slots=("usage_state", "background_or_room_context", "viewpoint"),
        slot_prefixes={"usage_state": "", "background_or_room_context": "in", "viewpoint": "from"},
        drop_if_unknown=("color", "material", "shape_or_structure", "usage_state", "background_or_room_context", "viewpoint"),
        optional_slots=("color", "material", "shape_or_structure", "usage_state", "background_or_room_context", "viewpoint"),
    ),
    "container": TemplateSpec(
        archetype="container",
        template_id="container_basic_v1",
        anchor_slot="container_type",
        pre_anchor_slots=("color", "material", "shape_or_structure"),
        post_anchor_slots=("fill_state_or_contents_visibility", "background_or_context", "viewpoint"),
        slot_prefixes={"fill_state_or_contents_visibility": "", "background_or_context": "in", "viewpoint": "from"},
        drop_if_unknown=("color", "material", "shape_or_structure", "fill_state_or_contents_visibility", "background_or_context", "viewpoint"),
        optional_slots=("color", "material", "shape_or_structure", "fill_state_or_contents_visibility", "background_or_context", "viewpoint"),
    ),
    "tool": TemplateSpec(
        archetype="tool",
        template_id="tool_basic_v1",
        anchor_slot="tool_type",
        pre_anchor_slots=("color", "material", "shape_or_structure"),
        post_anchor_slots=("usage_state", "background_or_context", "viewpoint"),
        slot_prefixes={"usage_state": "", "background_or_context": "in", "viewpoint": "from"},
        drop_if_unknown=("color", "material", "shape_or_structure", "usage_state", "background_or_context", "viewpoint"),
        optional_slots=("color", "material", "shape_or_structure", "usage_state", "background_or_context", "viewpoint"),
    ),
    "device_or_appliance": TemplateSpec(
        archetype="device_or_appliance",
        template_id="device_or_appliance_basic_v1",
        anchor_slot="device_or_appliance_type",
        pre_anchor_slots=("color", "material_or_finish", "shape_or_structure"),
        post_anchor_slots=("operating_state_or_display_state", "background_or_context", "viewpoint"),
        slot_prefixes={"operating_state_or_display_state": "", "background_or_context": "in", "viewpoint": "from"},
        drop_if_unknown=("color", "material_or_finish", "shape_or_structure", "operating_state_or_display_state", "background_or_context", "viewpoint"),
        optional_slots=("color", "material_or_finish", "shape_or_structure", "operating_state_or_display_state", "background_or_context", "viewpoint"),
    ),
    "instrument": TemplateSpec(
        archetype="instrument",
        template_id="instrument_basic_v1",
        anchor_slot="instrument_type",
        pre_anchor_slots=("color", "material", "shape_or_structure"),
        post_anchor_slots=("playing_state_or_pose", "background_or_context", "viewpoint"),
        slot_prefixes={"playing_state_or_pose": "", "background_or_context": "in", "viewpoint": "from"},
        drop_if_unknown=("color", "material", "shape_or_structure", "playing_state_or_pose", "background_or_context", "viewpoint"),
        optional_slots=("color", "material", "shape_or_structure", "playing_state_or_pose", "background_or_context", "viewpoint"),
    ),
    "weapon": TemplateSpec(
        archetype="weapon",
        template_id="weapon_basic_v1",
        anchor_slot="weapon_type",
        pre_anchor_slots=("color", "material", "shape_or_structure"),
        post_anchor_slots=("usage_or_display_state", "background_or_context", "viewpoint"),
        slot_prefixes={"usage_or_display_state": "", "background_or_context": "in", "viewpoint": "from"},
        drop_if_unknown=("color", "material", "shape_or_structure", "usage_or_display_state", "background_or_context", "viewpoint"),
        optional_slots=("color", "material", "shape_or_structure", "usage_or_display_state", "background_or_context", "viewpoint"),
    ),
    "sports_or_toy": TemplateSpec(
        archetype="sports_or_toy",
        template_id="sports_or_toy_basic_v1",
        anchor_slot="sports_or_toy_type",
        pre_anchor_slots=("color_or_pattern", "material", "shape_or_structure"),
        post_anchor_slots=("activity_or_usage_state", "background_or_context", "viewpoint"),
        slot_prefixes={"activity_or_usage_state": "", "background_or_context": "in", "viewpoint": "from"},
        drop_if_unknown=("color_or_pattern", "material", "shape_or_structure", "activity_or_usage_state", "background_or_context", "viewpoint"),
        optional_slots=("color_or_pattern", "material", "shape_or_structure", "activity_or_usage_state", "background_or_context", "viewpoint"),
    ),
    "household_object": TemplateSpec(
        archetype="household_object",
        template_id="household_object_basic_v1",
        anchor_slot="household_object_type",
        pre_anchor_slots=("color", "material", "shape_or_structure"),
        post_anchor_slots=("usage_state", "background_or_room_context", "viewpoint"),
        slot_prefixes={"usage_state": "", "background_or_room_context": "in", "viewpoint": "from"},
        drop_if_unknown=("color", "material", "shape_or_structure", "usage_state", "background_or_room_context", "viewpoint"),
        optional_slots=("color", "material", "shape_or_structure", "usage_state", "background_or_room_context", "viewpoint"),
    ),
    "structure_or_building": TemplateSpec(
        archetype="structure_or_building",
        template_id="structure_or_building_basic_v1",
        anchor_slot="structure_or_building_type",
        pre_anchor_slots=("material_or_surface", "architectural_style_or_form", "scale_or_extent"),
        post_anchor_slots=("surrounding_environment", "viewpoint", "salient_structural_part"),
        slot_prefixes={"surrounding_environment": "in", "viewpoint": "from", "salient_structural_part": "with"},
        drop_if_unknown=("material_or_surface", "architectural_style_or_form", "scale_or_extent", "surrounding_environment", "viewpoint", "salient_structural_part"),
        optional_slots=("material_or_surface", "architectural_style_or_form", "scale_or_extent", "surrounding_environment", "viewpoint", "salient_structural_part"),
    ),
    "natural_scene_or_landform": TemplateSpec(
        archetype="natural_scene_or_landform",
        template_id="natural_scene_or_landform_basic_v1",
        anchor_slot="scene_or_landform_type",
        pre_anchor_slots=("dominant_color_or_tone", "terrain_or_surface_trait"),
        post_anchor_slots=("weather_or_water_state", "vegetation_or_natural_context", "viewpoint", "salient_geographic_feature"),
        slot_prefixes={"weather_or_water_state": "with", "vegetation_or_natural_context": "in", "viewpoint": "from", "salient_geographic_feature": "with"},
        drop_if_unknown=("dominant_color_or_tone", "terrain_or_surface_trait", "weather_or_water_state", "vegetation_or_natural_context", "viewpoint", "salient_geographic_feature"),
        optional_slots=("dominant_color_or_tone", "terrain_or_surface_trait", "weather_or_water_state", "vegetation_or_natural_context", "viewpoint", "salient_geographic_feature"),
    ),
    "human_or_person": TemplateSpec(
        archetype="human_or_person",
        template_id="human_or_person_basic_v1",
        anchor_slot="person_type_or_role",
        pre_anchor_slots=("visible_body_trait", "clothing_or_gear"),
        post_anchor_slots=("body_pose_or_action", "background_or_activity_context", "viewpoint", "held_object_or_equipment"),
        slot_prefixes={"body_pose_or_action": "", "background_or_activity_context": "in", "viewpoint": "from", "held_object_or_equipment": "with"},
        drop_if_unknown=("visible_body_trait", "clothing_or_gear", "body_pose_or_action", "background_or_activity_context", "viewpoint", "held_object_or_equipment"),
        optional_slots=("visible_body_trait", "clothing_or_gear", "body_pose_or_action", "background_or_activity_context", "viewpoint", "held_object_or_equipment"),
    ),
    "text_or_media_object": TemplateSpec(
        archetype="text_or_media_object",
        template_id="text_or_media_object_basic_v1",
        anchor_slot="text_or_media_object_type",
        pre_anchor_slots=("dominant_color", "layout_or_format"),
        post_anchor_slots=("content_or_symbol_type", "physical_or_display_state", "background_or_context", "viewpoint"),
        slot_prefixes={"content_or_symbol_type": "showing", "physical_or_display_state": "", "background_or_context": "in", "viewpoint": "from"},
        drop_if_unknown=("dominant_color", "layout_or_format", "content_or_symbol_type", "physical_or_display_state", "background_or_context", "viewpoint"),
        optional_slots=("dominant_color", "layout_or_format", "content_or_symbol_type", "physical_or_display_state", "background_or_context", "viewpoint"),
    ),
    "decorative_or_symbolic_object": TemplateSpec(
        archetype="decorative_or_symbolic_object",
        template_id="decorative_or_symbolic_object_basic_v1",
        anchor_slot="decorative_or_symbolic_object_type",
        pre_anchor_slots=("color_or_pattern", "material", "ornamentation_or_symbolic_trait"),
        post_anchor_slots=("display_or_usage_context", "background_or_context", "viewpoint"),
        slot_prefixes={"display_or_usage_context": "in", "background_or_context": "in", "viewpoint": "from"},
        drop_if_unknown=("color_or_pattern", "material", "ornamentation_or_symbolic_trait", "display_or_usage_context", "background_or_context", "viewpoint"),
        optional_slots=("color_or_pattern", "material", "ornamentation_or_symbolic_trait", "display_or_usage_context", "background_or_context", "viewpoint"),
    ),
}


def get_template_spec(archetype: str) -> TemplateSpec:
    try:
        return TEMPLATE_SPECS[archetype]
    except KeyError as exc:
        raise KeyError(f"Unsupported Stage 2 archetype: {archetype}") from exc
