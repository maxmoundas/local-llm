from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass
class ModelOption:
    id: str
    name: str
    provider: str


class BaseProvider(ABC):
    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @property
    @abstractmethod
    def models(self) -> list[ModelOption]: ...

    @abstractmethod
    def stream(self, messages: list[dict], model_id: str, image_base64: str | None = None) -> Iterator[str]: ...

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)
