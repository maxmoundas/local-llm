from __future__ import annotations

import os

from src.providers.base import BaseProvider, ModelOption


class ProviderRegistry:
    def __init__(self) -> None:
        self._providers: list[BaseProvider] = []
        self._model_map: dict[str, BaseProvider] = {}

    def register(self, provider: BaseProvider) -> None:
        self._providers.append(provider)
        for model in provider.models:
            self._model_map[model.id] = provider

    def list_models(self) -> list[ModelOption]:
        return [m for p in self._providers for m in p.models]

    def get_provider(self, model_id: str) -> BaseProvider:
        if model_id not in self._model_map:
            raise KeyError(f"Unknown model: {model_id}")
        return self._model_map[model_id]

    @property
    def default_model_id(self) -> str:
        return self._providers[0].models[0].id

    @property
    def cheapest_model(self) -> ModelOption:
        """Return the first is_cheapest model whose provider is ready, in registration order.

        Providers register local → anthropic → openai, so the free local model is used once
        its weights are loaded, and a cloud cheap model (haiku, gpt-mini) covers until then.
        Falls back to the first flagged model, then the default model.
        """
        flagged = [(p, m) for p in self._providers for m in p.models if m.is_cheapest]
        for provider, model in flagged:
            if provider.ready:
                return model
        if flagged:
            return flagged[0][1]
        return self._providers[0].models[0]


def build_registry(llm) -> ProviderRegistry:
    """Auto-discover providers based on available API keys and the local model."""
    from src.providers.local import LocalProvider

    registry = ProviderRegistry()
    registry.register(LocalProvider(llm))

    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        from src.providers.anthropic_provider import AnthropicProvider
        registry.register(AnthropicProvider(anthropic_key))

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        from src.providers.openai_provider import OpenAIProvider
        registry.register(OpenAIProvider(openai_key))

    return registry
