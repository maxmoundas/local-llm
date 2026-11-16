"""OpenAI Chat Completions provider — also the generic provider for any OpenAI-compatible server.

The same class serves OpenAI itself, Ollama (http://localhost:11434/v1), LM Studio
(http://localhost:1234/v1), llama.cpp, vLLM, OpenRouter, Groq, Gemini's compatibility
endpoint, etc. Configure instances in models.yaml; `discover: true` lists the server's
models at startup so newly pulled local models appear without code changes.
"""
from __future__ import annotations

import json
from collections.abc import Iterator

from openai import OpenAI

from src import usage
from src.logger import get_logger
from src.providers.base import (
    EFFORT_LEVELS, REQUIRE_TOOL_NOTE, BaseProvider, ModelOption, RefusalError, ToolCall, ToolTurn, with_image,
    with_system_suffix,
)

_log = get_logger(__name__)

# Used when OpenAIProvider is built without a catalog (tests, old call sites).
_MODELS = [
    ModelOption(id="gpt-5.4-mini", name="GPT-5.4 Mini", provider="openai", is_cheapest=True, tools="native",
                vision=True, context_window=400_000, max_output_tokens=16_000, temperature=False, effort=True),
]


def _media_type(b64: str) -> str:
    head = b64[:16]
    if head.startswith("iVBOR"):
        return "image/png"
    if head.startswith("UklGR"):
        return "image/webp"
    if head.startswith("R0lGOD"):
        return "image/gif"
    return "image/jpeg"


def _oai_messages(messages: list[dict]) -> list[dict]:
    """Neutral messages → Chat Completions messages (images become image_url parts)."""
    out = []
    for m in messages:
        if m.get("images"):
            parts: list[dict] = [{"type": "image_url", "image_url": {"url": f"data:{_media_type(b)};base64,{b}"}}
                                 for b in m["images"]]
            out.append({"role": m["role"], "content": [*parts, {"type": "text", "text": m["content"] or "(image)"}]})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


class OpenAICompatibleProvider(BaseProvider):
    default_tool_mode = "auto"

    def __init__(self, name: str, api_key: str | None = None, base_url: str | None = None,
                 models: list[ModelOption] | None = None, discover: bool = False,
                 discovered_defaults: dict | None = None, timeout: float = 600.0) -> None:
        self._name = name
        self._has_key = bool(api_key)
        self._client = OpenAI(api_key=api_key or "not-needed", base_url=base_url, timeout=timeout)
        self._models = list(models or [])
        self._base_url = base_url
        # Streams that reject stream_options (older local servers) are retried without it.
        self._stream_usage = True
        if discover:
            self._discover(discovered_defaults or {})

    # ── catalog ───────────────────────────────────────────────────────────

    def _discover(self, defaults: dict) -> None:
        known = {m.id for m in self._models}
        try:
            listed = list(self._client.with_options(timeout=3.0, max_retries=0).models.list())
        except Exception as exc:
            _log.info("Model discovery skipped for %s (%s): %s", self._name, self._base_url, exc)
            return
        for item in listed:
            if item.id in known or any(s in item.id for s in ("embed", "whisper", "tts", "dall-e")):
                continue
            self._models.append(ModelOption(id=item.id, name=item.id, provider=self._name, **defaults))
        _log.info("Discovered %d model(s) on %s", len(listed), self._name)

    def check_auth(self) -> None:
        if self._has_key:  # keyless local servers (Ollama, LM Studio) have nothing to check
            self._probe_auth(lambda: self._client.with_options(timeout=5.0, max_retries=0).models.list())

    @property
    def provider_name(self) -> str:
        return self._name

    @property
    def models(self) -> list[ModelOption]:
        return self._models

    def _extra(self, model_id: str) -> dict:
        m = self.model(model_id)
        if not m:
            return {}
        out: dict = {}
        if m.extra_body:
            out["extra_body"] = m.extra_body
        if m.extra_headers:
            out["extra_headers"] = m.extra_headers
        return out

    def _limit(self, model_id: str, requested: int | None) -> dict:
        """OpenAI wants `max_completion_tokens`; most compatible servers (Ollama, vLLM, llama.cpp)
        only honour the older `max_tokens`."""
        key = "max_completion_tokens" if self._name == "openai" else "max_tokens"
        return {key: self._max_tokens(model_id, requested)}

    def _max_tokens(self, model_id: str, requested: int | None) -> int:
        m = self.model(model_id)
        cap = m.max_output_tokens if m else 8192
        return min(requested, cap) if requested else cap

    # ── calls ─────────────────────────────────────────────────────────────

    def _options(self, model_id: str, options: dict | None) -> dict:
        m, opts = self.model(model_id), options or {}
        out: dict = {}
        if opts.get("temperature") is not None and (m is None or m.temperature):
            out["temperature"] = float(opts["temperature"])
        if m and m.effort and opts.get("effort") in EFFORT_LEVELS:
            out["reasoning_effort"] = opts["effort"]
        return out

    def stream(self, messages: list[dict], model_id: str, image_base64: str | None = None,
               options: dict | None = None) -> Iterator[str]:
        messages = with_image(messages, image_base64)
        kwargs = dict(model=model_id, messages=_oai_messages(messages), stream=True,
                      **self._limit(model_id, None), **self._extra(model_id), **self._options(model_id, options))
        if self._stream_usage:
            try:
                stream = self._client.chat.completions.create(stream_options={"include_usage": True}, **kwargs)
            except Exception:
                self._stream_usage = False
                stream = self._create_compat(kwargs)
        else:
            stream = self._create_compat(kwargs)

        got_usage = False
        out_chars = 0
        for chunk in stream:
            if getattr(chunk, "usage", None) is not None and _record(model_id, chunk.usage):
                got_usage = True
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if _text(getattr(delta, "refusal", None)):
                raise RefusalError(delta.refusal)
            if delta.content:
                out_chars += len(delta.content)
                yield delta.content
        if not got_usage:
            usage.record(model_id, sum(len(str(m["content"])) for m in messages) // 4, out_chars // 4)

    def complete(self, messages: list[dict], model_id: str, max_tokens: int = 1024) -> str:
        response = self._create_compat(dict(
            model=model_id,
            messages=_oai_messages(messages),
            **self._limit(model_id, max_tokens),
            stream=False,
            **self._extra(model_id),
        ))
        _record(model_id, response.usage)
        message = response.choices[0].message
        if _text(getattr(message, "refusal", None)):
            raise RefusalError(message.refusal)
        return message.content or ""

    def _native_tool_turn(self, messages: list[dict], model_id: str, tools: list[dict], max_tokens: int,
                          require: bool) -> ToolTurn:
        msgs = with_system_suffix(messages, REQUIRE_TOOL_NOTE) if require else messages
        response = self._create_compat(dict(
            model=model_id,
            messages=[{"role": m["role"], "content": m["content"]} for m in msgs],
            tools=[{"type": "function", "function": t} for t in tools],
            tool_choice="auto",
            **self._limit(model_id, max_tokens),
            stream=False,
            **self._extra(model_id),
        ))
        _record(model_id, response.usage)
        message = response.choices[0].message
        if _text(getattr(message, "refusal", None)):
            raise RefusalError(message.refusal)
        turn = ToolTurn(text=message.content or "")
        for call in message.tool_calls or []:
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            turn.tool_calls.append(ToolCall(name=call.function.name, arguments=args if isinstance(args, dict) else {},
                                            id=call.id))
        return turn

    def _create_compat(self, kwargs: dict):
        """Some OpenAI-compatible servers only know the older `max_tokens` name."""
        try:
            return self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            if "max_completion_tokens" in kwargs and "max_completion_tokens" in str(exc):
                kwargs = dict(kwargs)
                kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
                return self._client.chat.completions.create(**kwargs)
            raise


def _text(value) -> bool:
    return isinstance(value, str) and bool(value)


def _int(value) -> int:
    return value if isinstance(value, int) else 0


def _record(model_id: str, u) -> bool:
    """Record a usage object; False when it carries no token counts."""
    if u is None:
        return False
    prompt, completion = _int(getattr(u, "prompt_tokens", 0)), _int(getattr(u, "completion_tokens", 0))
    if not (prompt or completion):
        return False
    details = getattr(u, "prompt_tokens_details", None)
    usage.record(model_id, prompt, completion, _int(getattr(details, "cached_tokens", 0)) if details else 0)
    return True


class OpenAIProvider(OpenAICompatibleProvider):
    """OpenAI's own API."""

    def __init__(self, api_key: str, models: list[ModelOption] | None = None) -> None:
        super().__init__("openai", api_key=api_key, models=models if models is not None else [
            ModelOption(**{**m.__dict__}) for m in _MODELS
        ])
