from __future__ import annotations

"""Backend factory for Stage 1 VLM clients.

The immediate goal is to keep the rest of the pipeline backend-agnostic.
Real model integrations can be added later without rewriting pipeline logic.
"""

from cspd_stage1.vlm.base import BaseVLMClient
from cspd_stage1.vlm.mock import MockVLMClient


class UnsupportedBackendError(ValueError):
    """Raised when the requested backend name is unknown."""

    pass


class PlaceholderVLMClient(BaseVLMClient):
    """Temporary stand-in for real backends that are not implemented yet."""

    def __init__(self, backend_name: str):
        self.backend_name = backend_name

    def extract_attributes(self, sample, user_prompt: str, system_prompt: str):  # type: ignore[override]
        raise NotImplementedError(
            f"Backend '{self.backend_name}' is not implemented yet. "
            "Add a concrete client under src/cspd_stage1/vlm/."
        )


def create_vlm_client(backend: str) -> BaseVLMClient:
    """Instantiate the requested backend implementation.

    Today only `mock` is real. The named placeholders are here so the config/API
    shape is already frozen before we wire an actual provider.
    """
    backend = backend.lower().strip()
    if backend == "mock":
        return MockVLMClient()
    if backend in {"openai", "qwen-vl", "internvl", "llava", "claude-vision"}:
        return PlaceholderVLMClient(backend)
    raise UnsupportedBackendError(f"Unsupported backend: {backend}")
