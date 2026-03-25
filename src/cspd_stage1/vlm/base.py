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


class VLMOutputParseError(ValueError):
    """Raised when a backend produced text but the text could not be parsed.

    We carry `raw_text` with the exception so Stage 1 can preserve failed model
    output inside `failed_samples.jsonl` instead of dropping the most useful
    debugging signal.
    """

    def __init__(self, message: str, raw_text: str | None = None):
        super().__init__(message)
        self.raw_text = raw_text


class BaseVLMClient(ABC):
    """Common interface all concrete VLM clients must implement."""

    @abstractmethod
    def extract_attributes(self, sample: SampleRecord, user_prompt: str, system_prompt: str) -> VLMResponse:
        """Run attribute extraction for one sample and return a normalized response."""
        raise NotImplementedError
