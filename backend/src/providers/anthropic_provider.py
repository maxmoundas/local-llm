from __future__ import annotations

import base64
from collections.abc import Iterator

import anthropic

from src import usage
from src.logger import get_logger
from src.providers.base import (
    REQUIRE_TOOL_NOTE, BaseProvider, ModelOption, RefusalError, ToolCall, ToolTurn, with_system_suffix,
)

_log = get_logger(__name__)

# Used when AnthropicProvider is built without a catalog (tests, old call sites).
# The real list lives in models.yaml.
_MODELS = [
    ModelOption(id="claude-haiku-5-5", name="Claude Haiku 5.5", provider="anthropic", is_cheapest=True,
                tools="native", vision=True, context_window=1_000_000, max_output_tokens=16_000,
                input_price=0.10, output_price=0.50),
    ModelOption(id="claude-sonnet-5-5", name="Claude Sonnet 5.5", provider="anthropic", tools="native",
                vision=True, context_window=1_000_000, max_output_tokens=32_000,
                input_price=2.0, output_price=10.0),
]


def _detect_media_type(data: bytes) -> str:
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[8:12] == b"WEBP":
        return "image/webp"
    if data[:4] == b"GIF8":
        return "image/gif"
    return "image/jpeg"


class AnthropicProvider(BaseProvider):
    default_tool_mode = "native"

    def __init__(self, api_key: str, models: list[ModelOption] | None = None, discover: bool = False,
                 discovered_defaults: dict | None = None) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)
        self._models = list(models) if models is not None else [ModelOption(**m.__dict__) for m in _MODELS]
        if discover:
            self._discover(discovered_defaults or {})

    def _discover(self, defaults: dict) -> None:
        """Add models from the Models API that the catalog doesn't list yet (no pricing known)."""
        known = {m.id for m in self._models}
        try:
            listed = list(self._client.with_options(timeout=5.0, max_retries=0).models.list())
        except Exception as exc:
            _log.info("Anthropic model discovery skipped: %s", exc)
            return
        for item in listed:
            if item.id not in known:
                name = getattr(item, "display_name", None) or item.id
                ctx = getattr(item, "max_input_tokens", None)
                opts = {"tools": "native", "vision": True, **defaults}
                if ctx:
                    opts["context_window"] = ctx
                self._models.append(ModelOption(id=item.id, name=name, provider="anthropic", **opts))

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def models(self) -> list[ModelOption]:
        return self._models

    def _request(self, messages: list[dict], model_id: str, max_tokens: int | None) -> dict:
        """Build messages.create kwargs: system to the top-level param (cached), model extras applied."""
        system: str | None = None
        filtered: list[dict] = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"] if system is None else f"{system}\n\n{m['content']}"
            else:
                filtered.append({"role": m["role"], "content": m["content"]})
        m = self.model(model_id)
        cap = m.max_output_tokens if m else 8192
        kwargs: dict = dict(model=model_id, max_tokens=min(max_tokens, cap) if max_tokens else cap, messages=filtered)
        if system:
            # Cache breakpoint on the system prompt: agent turns resend the same system + tools
            # many times, and cached reads cost ~10% of fresh input. Short prompts just don't cache.
            kwargs["system"] = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        if m and m.extra_body:
            kwargs["extra_body"] = m.extra_body
        if m and m.extra_headers:
            kwargs["extra_headers"] = m.extra_headers
        return kwargs

    def stream(self, messages: list[dict], model_id: str, image_base64: str | None = None) -> Iterator[str]:
        kwargs = self._request(messages, model_id, None)
        if image_base64:
            last = kwargs["messages"][-1]
            if last["role"] == "user":
                media_type = _detect_media_type(base64.b64decode(image_base64))
                last["content"] = [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_base64}},
                    {"type": "text", "text": last["content"]},
                ]
        with self._client.messages.stream(**kwargs) as s:
            for text in s.text_stream:
                yield text
            final = s.get_final_message()
        _record(model_id, final)
        _check_refusal(final)

    def complete(self, messages: list[dict], model_id: str, max_tokens: int = 1024) -> str:
        response = self._client.messages.create(**self._request(messages, model_id, max_tokens))
        _record(model_id, response)
        _check_refusal(response)
        return "".join(b.text for b in response.content if b.type == "text")

    def _native_tool_turn(self, messages: list[dict], model_id: str, tools: list[dict], max_tokens: int,
                          require: bool) -> ToolTurn:
        msgs = with_system_suffix(messages, REQUIRE_TOOL_NOTE) if require else messages
        kwargs = self._request(msgs, model_id, max(max_tokens, 4096))
        kwargs["tools"] = [
            {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
            for t in tools
        ]
        # Always "auto": forced choices ("any"/"tool") are rejected by current models.
        kwargs["tool_choice"] = {"type": "auto"}
        response = self._client.messages.create(**kwargs)
        _record(model_id, response)
        _check_refusal(response)
        turn = ToolTurn()
        for block in response.content:
            if block.type == "text":
                turn.text += block.text
            elif block.type == "tool_use":
                turn.tool_calls.append(ToolCall(name=block.name, arguments=dict(block.input), id=block.id))
        return turn


def _record(model_id: str, response) -> None:
    u = getattr(response, "usage", None)
    if not u:
        return
    def n(name: str) -> int:
        v = getattr(u, name, 0)
        return v if isinstance(v, int) else 0
    cached = n("cache_read_input_tokens")
    usage.record(model_id, n("input_tokens") + cached + n("cache_creation_input_tokens"), n("output_tokens"), cached)


def _check_refusal(response) -> None:
    if getattr(response, "stop_reason", None) == "refusal":  # noqa: SIM102
        details = getattr(response, "stop_details", None)
        raise RefusalError(getattr(details, "explanation", None) or "The model declined this request.")
