"""Deterministic test doubles that plug into the real provider/registry/LLMService stack.

ScriptedProvider replays a script of replies, so agent and task tests exercise the real
plumbing (LLMService logging, usage metering, prompted-JSON parsing, the registry) without
network or weights.
"""
from __future__ import annotations

import json
from collections.abc import Iterator

from src import usage
from src.llm_service import LLMService
from src.providers.base import BaseProvider, ModelOption, ToolCall, ToolTurn
from src.providers.registry import ProviderRegistry


def calls(*pairs: tuple[str, dict]) -> ToolTurn:
    """ToolTurn(tool_calls=[...]) from (name, args) pairs."""
    return ToolTurn(tool_calls=[ToolCall(n, a, id=f"c{i}") for i, (n, a) in enumerate(pairs)])


class ScriptedProvider(BaseProvider):
    """`turns` feed complete_with_tools (ToolTurns, or raw JSON strings in prompted mode);
    `texts` feed complete(); `stream_text` is what stream() yields, word by word."""

    def __init__(self, turns=(), texts=(), stream_text: str = "hello world", tools_mode: str = "native",
                 model_id: str = "fake", tokens_per_call: tuple[int, int] = (100, 20), price=(1.0, 2.0)) -> None:
        self.turns = list(turns)
        self.texts = list(texts)
        self.stream_text = stream_text
        self.tokens = tokens_per_call
        self.seen: list[list[dict]] = []  # messages of every call, in order
        self.seen_tools: list[list[dict]] = []
        self._models = [ModelOption(id=model_id, name=model_id, provider="fake", is_cheapest=True,
                                    tools=tools_mode, input_price=price[0], output_price=price[1])]

    provider_name = "fake"

    @property
    def models(self) -> list[ModelOption]:
        return self._models

    def _bill(self) -> None:
        usage.record(self._models[0].id, *self.tokens)

    def stream(self, messages, model_id, image_base64=None) -> Iterator[str]:
        self.seen.append(messages)
        self._bill()
        for i, word in enumerate(self.stream_text.split(" ")):
            yield word if i == 0 else " " + word

    def complete(self, messages, model_id, max_tokens=None) -> str:
        self.seen.append(messages)
        self._bill()
        if self.texts:
            return self.texts.pop(0)
        prompted_call = "You can call these tools" in str(messages[0].get("content", ""))
        if prompted_call and self.turns:  # prompted tool turn
            t = self.turns.pop(0)
            if isinstance(t, str):
                return t
            return json.dumps({"tool_calls": [{"name": c.name, "arguments": c.arguments} for c in t.tool_calls]})
        return "ok"

    def _native_tool_turn(self, messages, model_id, tools, max_tokens, require) -> ToolTurn:
        self.seen.append(messages)
        self.seen_tools.append(tools)
        self._bill()
        if not self.turns:
            return ToolTurn()
        t = self.turns.pop(0)
        if isinstance(t, Exception):
            raise t
        return t


def service(provider: BaseProvider) -> LLMService:
    reg = ProviderRegistry()
    reg.register(provider)
    return LLMService(reg)
