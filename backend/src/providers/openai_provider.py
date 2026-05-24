from __future__ import annotations

from collections.abc import Iterator

from openai import OpenAI

from src.providers.base import BaseProvider, ModelOption

_MODELS = [
    # ModelOption(id="gpt-5.5", name="GPT-5.5", provider="openai"),      # expensive — uncomment to re-enable
    # ModelOption(id="gpt-5.5-pro", name="GPT-5.5 Pro", provider="openai"),  # expensive — uncomment to re-enable
    ModelOption(id="gpt-5.4-mini", name="GPT-5.4 Mini", provider="openai"),
]


class OpenAIProvider(BaseProvider):
    def __init__(self, api_key: str) -> None:
        self._client = OpenAI(api_key=api_key)

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def models(self) -> list[ModelOption]:
        return _MODELS

    def stream(self, messages: list[dict], model_id: str, image_base64: str | None = None) -> Iterator[str]:
        oai_messages: list[dict] = []
        for i, m in enumerate(messages):
            if image_base64 and m["role"] == "user" and i == len(messages) - 1:
                content = [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    {"type": "text", "text": m["content"]},
                ]
            else:
                content = m["content"]
            oai_messages.append({"role": m["role"], "content": content})

        stream = self._client.chat.completions.create(
            model=model_id,
            messages=oai_messages,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content
