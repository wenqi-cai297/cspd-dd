from __future__ import annotations

"""Backward-compatible aliases for the Stage 1 render pipeline.

Canonical Stage 1 render code now lives under ``cspd_stage1``.
Keep these exports so older imports continue to work.
"""

from cspd_stage1.render_pipeline import (
    Stage1RenderConfig,
    build_summary,
    config_from_args,
    render_row,
    run_stage1_render,
)

Stage2Config = Stage1RenderConfig
run_stage2 = run_stage1_render

__all__ = [
    "Stage1RenderConfig",
    "Stage2Config",
    "build_summary",
    "config_from_args",
    "render_row",
    "run_stage1_render",
    "run_stage2",
]
