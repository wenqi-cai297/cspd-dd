from __future__ import annotations

"""Backend factory for Stage 1 VLM clients.

The immediate goal is to keep the rest of the pipeline backend-agnostic.
Real model integrations can be added later without rewriting pipeline logic.
"""

from cspd_stage1.vlm.base import BaseVLMClient
from cspd_stage1.vlm.mock import MockVLMClient
from cspd_stage1.vlm.qwen_local import QwenLocalVLMClient


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


def create_vlm_client(
    backend: str,
    *,
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype: str = "float16",
    device_map: str = "auto",
    use_fast_processor: bool = True,
    max_new_tokens: int = 256,
) -> BaseVLMClient:
    """Instantiate the requested backend implementation.

    The extra keyword arguments are ignored by lightweight backends like `mock`
    but consumed by real local model backends such as `qwen_local`.
    """
    backend = backend.lower().strip()
    if backend == "mock":
        return MockVLMClient()
    if backend == "qwen_local":
        return QwenLocalVLMClient(
            model_name=model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            use_fast_processor=use_fast_processor,
            max_new_tokens=max_new_tokens,
        )
    if backend in {"openai", "qwen-vl", "internvl", "llava", "claude-vision"}:
        return PlaceholderVLMClient(backend)
    raise UnsupportedBackendError(f"Unsupported backend: {backend}")
