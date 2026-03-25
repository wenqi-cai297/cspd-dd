from __future__ import annotations

"""Abstract interface for VLM backends used by Stage 1."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from cspd_stage1.schema import SampleRecord


@dataclass(slots=True)
class VLMResponse:
    """Normalized response container returned by a VLM backend.

    Attributes:
        payload: Parsed JSON-like dictionary containing attribute fields.
        raw_text: Optional raw backend output kept for debugging and audits.
    """

    payload: dict[str, Any]
    raw_text: str | None = None


class BaseVLMClient(ABC):
    """Common interface all concrete VLM clients must implement."""

    @abstractmethod
    def extract_attributes(self, sample: SampleRecord, user_prompt: str, system_prompt: str) -> VLMResponse:
        """Run attribute extraction for one sample and return a normalized response."""
        raise NotImplementedError
