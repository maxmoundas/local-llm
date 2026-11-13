"""
Returns the right local-model class for the current platform.

  macOS  → LocalMLXLLM  (mlx-lm, Apple Silicon Metal)
  other  → LocalLLM     (HuggingFace Transformers + bitsandbytes, CUDA)

Imports are deferred so the unused backend's heavy dependencies
(torch/bitsandbytes on Mac, mlx on Windows) are never loaded.
"""
from __future__ import annotations

import platform
from typing import Any


def create_local_llm(cfg: dict[str, Any]):
    if platform.system() == "Darwin":
        from src.model_mlx import LocalMLXLLM
        return LocalMLXLLM(cfg)
    from src.model import LocalLLM
    return LocalLLM(cfg)
