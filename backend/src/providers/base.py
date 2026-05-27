from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field


@dataclass
class ModelOption:
    id: str
    name: str
    provider: str
    # When True, LLMService uses this model for utility calls (e.g. title generation).
    # Mark the cheapest model per provider; cloud cheap models are preferred over local.
    is_cheapest: bool = field(default=False)


class BaseProvider(ABC):
    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @property
    @abstractmethod
    def models(self) -> list[ModelOption]: ...

    @abstractmethod
    def stream(self, messages: list[dict], model_id: str, image_base64: str | None = None) -> Iterator[str]: ...

    @abstractmethod
    def complete(self, messages: list[dict], model_id: str) -> str:
        """Non-streaming completion — returns the full response as a string."""
        ...

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)
