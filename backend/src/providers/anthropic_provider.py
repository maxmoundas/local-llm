from __future__ import annotations

import base64
from collections.abc import Iterator

import anthropic

from src.providers.base import BaseProvider, ModelOption, ToolCall, ToolTurn

_MODELS = [
    ModelOption(id="claude-haiku-4-5-20251001", name="Claude Haiku 4.5", provider="anthropic", is_cheapest=True),
    ModelOption(id="claude-sonnet-4-6", name="Claude Sonnet 4.6", provider="anthropic"),
    # ModelOption(id="claude-opus-4-7", name="Claude Opus 4.7", provider="anthropic"),  # expensive — uncomment to re-enable
]


def _detect_media_type(data: bytes) -> str:
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


class AnthropicProvider(BaseProvider):
    def __init__(self, api_key: str) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def models(self) -> list[ModelOption]:
        return _MODELS

    def stream(self, messages: list[dict], model_id: str, image_base64: str | None = None) -> Iterator[str]:
        system: str | None = None
        filtered: list[dict] = []

        for i, m in enumerate(messages):
            if m["role"] == "system":
                system = m["content"]
                continue
            if image_base64 and m["role"] == "user" and i == len(messages) - 1:
                img_bytes = base64.b64decode(image_base64)
                media_type = _detect_media_type(img_bytes)
                content = [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": image_base64},
                    },
                    {"type": "text", "text": m["content"]},
                ]
            else:
                content = m["content"]
            filtered.append({"role": m["role"], "content": content})

        kwargs: dict = dict(model=model_id, max_tokens=4096, messages=filtered)
        if system:
            kwargs["system"] = system

        with self._client.messages.stream(**kwargs) as s:
            for text in s.text_stream:
                yield text

    def complete(self, messages: list[dict], model_id: str, max_tokens: int = 1024) -> str:
        kwargs = _split_system(messages, model_id, max_tokens)
        response = self._client.messages.create(**kwargs)
        return "".join(b.text for b in response.content if b.type == "text")

    def complete_with_tools(
        self, messages: list[dict], model_id: str, tools: list[dict], max_tokens: int = 1024,
        require: bool = False,
    ) -> ToolTurn:
        kwargs = _split_system(messages, model_id, max_tokens)
        kwargs["tools"] = [
            {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
            for t in tools
        ]
        kwargs["tool_choice"] = {"type": "any" if require else "auto"}
        response = self._client.messages.create(**kwargs)
        turn = ToolTurn()
        for block in response.content:
            if block.type == "text":
                turn.text += block.text
            elif block.type == "tool_use":
                turn.tool_calls.append(ToolCall(name=block.name, arguments=dict(block.input), id=block.id))
        return turn


def _split_system(messages: list[dict], model_id: str, max_tokens: int) -> dict:
    """Build messages.create kwargs, moving the system message to the top-level param."""
    system: str | None = None
    filtered: list[dict] = []
    for m in messages:
        if m["role"] == "system":
            system = m["content"]
        else:
            filtered.append({"role": m["role"], "content": m["content"]})
    kwargs: dict = dict(model=model_id, max_tokens=max_tokens, messages=filtered)
    if system:
        kwargs["system"] = system
    return kwargs
