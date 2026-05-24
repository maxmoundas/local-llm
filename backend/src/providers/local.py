from __future__ import annotations

import base64
from collections.abc import Iterator
from io import BytesIO

from PIL import Image as PILImage

from src.model import LocalLLM
from src.providers.base import BaseProvider, ModelOption


class LocalProvider(BaseProvider):
    def __init__(self, llm: LocalLLM) -> None:
        self._llm = llm

    @property
    def provider_name(self) -> str:
        return "local"

    @property
    def models(self) -> list[ModelOption]:
        short_name = self._llm.repo_id.split("/")[-1]
        return [ModelOption(id=self._llm.repo_id, name=short_name, provider="local")]

    def stream(self, messages: list[dict], model_id: str, image_base64: str | None = None) -> Iterator[str]:
        if not self._llm.is_loaded:
            raise RuntimeError("Local model is still loading — please wait a moment and try again")
        if image_base64:
            img_bytes = base64.b64decode(image_base64)
            image = PILImage.open(BytesIO(img_bytes)).convert("RGB")
            return self._llm.stream_with_image(image, messages)
        return self._llm.stream(messages)

    def count_tokens(self, text: str) -> int:
        if not self._llm.is_loaded:
            return len(text) // 4  # rough estimate while model loads
        return self._llm.count_tokens(text)
