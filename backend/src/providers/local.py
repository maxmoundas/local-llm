from __future__ import annotations

import base64
from collections.abc import Iterator
from io import BytesIO
from typing import TYPE_CHECKING

from PIL import Image as PILImage

from src import usage
from src.providers.base import BaseProvider, ModelOption, with_image

if TYPE_CHECKING:  # avoid importing torch on the MLX backend
    from src.model import LocalLLM


class LocalProvider(BaseProvider):
    def __init__(self, llm: LocalLLM) -> None:
        self._llm = llm

    @property
    def provider_name(self) -> str:
        return "local"

    @property
    def models(self) -> list[ModelOption]:
        short_name = self._llm.repo_id.split("/")[-1]
        if getattr(self, "_model_option", None) is None or self._model_option.id != self._llm.repo_id:
            # Prompted tool calls: in-process models have no tool-calling API.
            self._model_option = ModelOption(
                id=self._llm.repo_id, name=short_name, provider="local", is_cheapest=False, tools="prompted",
                vision=getattr(self._llm, "supports_vision", True),
                max_output_tokens=int(getattr(self._llm, "cfg", {}).get("max_new_tokens", 2048) or 2048),
            )
        return [self._model_option]

    @property
    def ready(self) -> bool:
        return self._llm.is_loaded

    def stream(self, messages: list[dict], model_id: str, image_base64: str | None = None,
               options: dict | None = None) -> Iterator[str]:
        if not self._llm.is_loaded:
            raise RuntimeError("Local model is still loading — please wait a moment and try again")
        messages = with_image(messages, image_base64)
        # The in-process model takes one image: the most recent one in the conversation.
        latest = next((m["images"][-1] for m in reversed(messages) if m.get("images")), None)
        plain = [{"role": m["role"], "content": m["content"]} for m in messages]
        if latest:
            image = PILImage.open(BytesIO(base64.b64decode(latest))).convert("RGB")
            return _metered(self._llm.stream_with_image(image, plain), plain, model_id)
        temperature = (options or {}).get("temperature")
        if temperature is not None:
            return _metered(self._llm.stream(plain, temperature=float(temperature)), plain, model_id)
        return _metered(self._llm.stream(plain), plain, model_id)

    def complete(self, messages: list[dict], model_id: str, max_tokens: int | None = None) -> str:
        if not self._llm.is_loaded:
            raise RuntimeError("Local model is still loading — please wait a moment and try again")
        messages = [{"role": m["role"], "content": m["content"]} for m in messages]
        if max_tokens is None:
            out = self._llm.generate(messages)
        else:
            out = self._llm.generate(messages, max_new_tokens=max_tokens)
        usage.record(model_id, _chars(messages) // 4, len(out or "") // 4)
        return out

    def count_tokens(self, text: str) -> int:
        if not self._llm.is_loaded:
            return len(text) // 4  # rough estimate while model loads
        return self._llm.count_tokens(text)


def _chars(messages: list[dict]) -> int:
    return sum(len(m["content"]) if isinstance(m.get("content"), str) else 0 for m in messages)


def _metered(tokens: Iterator[str], messages: list[dict], model_id: str) -> Iterator[str]:
    """Record approximate usage (free, but it keeps token counts comparable across models)."""
    out = 0
    try:
        for t in tokens:
            out += len(t)
            yield t
    finally:
        usage.record(model_id, _chars(messages) // 4, out // 4)
