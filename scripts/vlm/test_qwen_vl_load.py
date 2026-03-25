"""Minimal smoke test for loading a local Qwen2.5-VL model.

Purpose:
- verify that transformers can download the checkpoint,
- verify that CUDA is visible to PyTorch,
- verify that the processor and model load without crashing.

This script does NOT run real image inference yet.
"""

from __future__ import annotations

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"


def main() -> None:
    """Load the model and processor, then print basic environment info."""
    print("Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    # Keep references alive so the load is not optimized away by refactors.
    _ = (model, processor)

    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    print("Model loaded successfully.")


if __name__ == "__main__":
    main()
