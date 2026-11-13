"""
Model loading and inference wrapper using MLX-LM (Mac / Apple Silicon).

MLX is Apple's ML framework optimised for Apple Silicon. This class
mirrors the interface of LocalLLM so the rest of the app is unaware of
which backend is active.

Vision/image input is not supported on this backend — image requests
raise a RuntimeError that surfaces as an error message in the UI.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from typing import Any

from src.logger import get_logger

logger = get_logger(__name__)


class LocalMLXLLM:
    supports_vision = False

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg["model"]
        self.model: Any = None
        self.tokenizer: Any = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        # Deferred so non-Mac environments never try to import mlx_lm.
        from mlx_lm import load  # type: ignore[import]

        repo_id = self.cfg["repo_id"]
        cache_dir = self.cfg.get("cache_dir", "./models")

        logger.info("Loading MLX model from %s (cache_dir=%s)", repo_id, cache_dir)
        t0 = time.perf_counter()
        self.model, self.tokenizer = load(
            repo_id,
            tokenizer_config={"trust_remote_code": False},
        )
        logger.info(
            "MLX model loaded in %.1fs | class=%s",
            time.perf_counter() - t0,
            type(self.model).__name__,
        )

    # ------------------------------------------------------------------
    # Text inference
    # ------------------------------------------------------------------

    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
    ) -> str:
        from mlx_lm import generate as mlx_generate  # type: ignore[import]

        assert self.model is not None and self.tokenizer is not None

        max_new_tokens = max_new_tokens or self.cfg.get("max_new_tokens", 512)
        temperature = temperature if temperature is not None else self.cfg.get("temperature", 0.7)
        top_p = top_p if top_p is not None else self.cfg.get("top_p", 0.9)
        repetition_penalty = repetition_penalty or self.cfg.get("repetition_penalty", 1.1)

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        t0 = time.perf_counter()
        with self._lock:
            result = mlx_generate(
                self.model,
                self.tokenizer,
                prompt,
                max_tokens=max_new_tokens,
                temp=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                verbose=False,
            )
        # mlx_lm.generate returns a string in all versions.
        text = result.strip()
        n_tokens = self.count_tokens(text)
        elapsed = time.perf_counter() - t0
        logger.info(
            "MLX generation done | ~tokens=%d | %.1fs | %.1f tok/s",
            n_tokens, elapsed, n_tokens / max(elapsed, 0.001),
        )
        return text

    def stream(
        self,
        messages: list[dict],
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
    ) -> Iterator[str]:
        from mlx_lm import stream_generate  # type: ignore[import]

        assert self.model is not None and self.tokenizer is not None

        max_new_tokens = max_new_tokens or self.cfg.get("max_new_tokens", 512)
        temperature = temperature if temperature is not None else self.cfg.get("temperature", 0.7)
        top_p = top_p if top_p is not None else self.cfg.get("top_p", 0.9)
        repetition_penalty = repetition_penalty or self.cfg.get("repetition_penalty", 1.1)

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # stream_generate yields GenerationResponse objects (.text = new token text).
        # Older builds (<0.14) yielded plain strings — the getattr handles both.
        with self._lock:
            for chunk in stream_generate(
                self.model,
                self.tokenizer,
                prompt,
                max_tokens=max_new_tokens,
                temp=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            ):
                token_text = getattr(chunk, "text", chunk)
                if token_text:
                    yield token_text

    # ------------------------------------------------------------------
    # Vision inference (not supported on MLX backend)
    # ------------------------------------------------------------------

    def stream_with_image(
        self,
        image: Any,
        messages: list[dict],
        max_new_tokens: int | None = None,
    ) -> Iterator[str]:
        raise RuntimeError(
            "Image input is not supported on the MLX (Mac) backend. "
            "Use an Anthropic or OpenAI cloud model for vision requests."
        )

    def generate_with_image(
        self,
        image: Any,
        prompt: str,
        max_new_tokens: int = 512,
    ) -> str:
        raise RuntimeError(
            "Image input is not supported on the MLX (Mac) backend. "
            "Use an Anthropic or OpenAI cloud model for vision requests."
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        if self.tokenizer is None:
            return max(1, len(text) // 4)
        tokens = self.tokenizer.encode(text)
        return len(tokens)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    @property
    def repo_id(self) -> str:
        return self.cfg["repo_id"]
