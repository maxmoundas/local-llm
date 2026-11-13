from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field

from src.logger import get_logger

_log = get_logger(__name__)

# How a model selects tools:
#   native   — the provider's tool-calling API
#   prompted — a JSON reply described in the system prompt (works with any text model)
#   auto     — try native; on the first error fall back to prompted and remember that
TOOL_MODES = ("native", "prompted", "auto")


@dataclass
class ModelOption:
    id: str
    name: str
    provider: str
    # When True, LLMService uses this model for utility calls (e.g. title generation).
    # Mark the cheapest model per provider; cloud cheap models are preferred over local.
    is_cheapest: bool = field(default=False)
    tools: str = "auto"
    vision: bool = False
    context_window: int = 32_000
    max_output_tokens: int = 8_192
    input_price: float = 0.0   # $ per 1M input tokens
    output_price: float = 0.0  # $ per 1M output tokens
    # Provider-specific request fields (e.g. {"output_config": {"effort": "low"}}) and headers.
    extra_body: dict = field(default_factory=dict)
    extra_headers: dict = field(default_factory=dict)

    def public(self) -> dict:
        return {
            "id": self.id, "name": self.name, "provider": self.provider, "is_cheapest": self.is_cheapest,
            "tools": self.tools, "vision": self.vision, "context_window": self.context_window,
            "input_price": self.input_price, "output_price": self.output_price,
        }


@dataclass
class ToolCall:
    name: str
    arguments: dict
    id: str = ""


@dataclass
class ToolTurn:
    """One model turn in a tool loop: optional text plus zero or more tool calls."""
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class RefusalError(RuntimeError):
    """The model declined the request (e.g. a safety classifier)."""


# Appended to the system prompt by the default (prompted) tool-calling implementation.
_PROMPTED_TOOLS_TEMPLATE = """

You can call these tools:
{tools}

Reply with ONLY a JSON object, no other text:
{{"tool_calls": [{{"name": "<tool name>", "arguments": {{...}}}}]}}
Call several tools at once when they are independent. When you already have enough
information, reply with {{"tool_calls": []}}."""

# Forced tool_choice ("any"/"required") is rejected by several current models and many local
# servers, so "must call a tool" is always an instruction — it works everywhere.
REQUIRE_TOOL_NOTE = "\n\nYou must call at least one tool now."


def with_system_suffix(messages: list[dict], suffix: str) -> list[dict]:
    msgs = [dict(m) for m in messages]
    if msgs and msgs[0]["role"] == "system":
        msgs[0]["content"] = f"{msgs[0]['content']}{suffix}"
    else:
        msgs.insert(0, {"role": "system", "content": suffix.strip()})
    return msgs


class BaseProvider(ABC):
    # Tool mode for model ids the catalog doesn't describe.
    default_tool_mode = "prompted"

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

    def model(self, model_id: str) -> ModelOption | None:
        return next((m for m in self.models if m.id == model_id), None)

    def complete_with_tools(
        self, messages: list[dict], model_id: str, tools: list[dict], max_tokens: int = 1024,
        require: bool = False,
    ) -> ToolTurn:
        """Ask the model which tools to call next.

        `tools` are neutral specs: {"name", "description", "parameters": <JSON schema>}.
        `require` asks for at least one call. Native providers override `_native_tool_turn`;
        models flagged "prompted" (or "auto" models whose native call failed) use JSON replies.
        """
        m = self.model(model_id)
        mode = m.tools if m else self.default_tool_mode
        if mode != "prompted" and type(self)._native_tool_turn is not BaseProvider._native_tool_turn:
            try:
                turn = self._native_tool_turn(messages, model_id, tools, max_tokens, require)
                if turn.tool_calls or not require or mode == "native":
                    return turn
                # Required a call but got none: small models often manage with the JSON format instead.
                _log.info("No native tool call from %s when one was required; retrying with prompted JSON", model_id)
            except RefusalError:
                raise
            except Exception as exc:
                if mode == "native":
                    raise
                # "auto": this model/server can't do native tools — remember and fall back.
                _log.warning("Native tool calling failed for %s, switching to prompted JSON: %s", model_id, exc)
                if m:
                    m.tools = "prompted"
        return self._prompted_tool_turn(messages, model_id, tools, require)

    def _native_tool_turn(self, messages: list[dict], model_id: str, tools: list[dict], max_tokens: int,
                          require: bool) -> ToolTurn:
        raise NotImplementedError

    def _prompted_tool_turn(self, messages: list[dict], model_id: str, tools: list[dict], require: bool) -> ToolTurn:
        listing = "\n".join(
            f"- {t['name']}: {t['description']} Arguments (JSON schema): {json.dumps(t['parameters'])}"
            for t in tools
        )
        suffix = _PROMPTED_TOOLS_TEMPLATE.format(tools=listing) + (REQUIRE_TOOL_NOTE if require else "")
        msgs = with_system_suffix(messages, suffix)
        names = {t["name"] for t in tools}
        raw = self.complete(msgs, model_id)
        turn = parse_prompted_tool_turn(raw, names)
        if require and not turn.tool_calls:
            # Small models often wrap or mangle the JSON once; one repair attempt fixes most of it.
            repair = msgs + [
                {"role": "assistant", "content": raw[:2000]},
                {"role": "user", "content": 'That was not a valid tool call. Reply with ONLY the JSON object '
                                            '{"tool_calls": [...]} calling at least one of the listed tools.'},
            ]
            turn = parse_prompted_tool_turn(self.complete(repair, model_id), names)
        return turn

    @property
    def ready(self) -> bool:
        """False while the provider can't serve requests (e.g. local weights still loading)."""
        return True

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


def _balance_json(text: str) -> str:
    """Close brackets/strings a model left open (truncated or sloppy JSON). Text must start at '{'."""
    stack: list[str] = []
    in_str = esc = False
    for ch in text:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()
    return text + ('"' if in_str else "") + "".join(reversed(stack))


def _loads_lenient(text: str):
    """json.loads that tolerates raw newlines in strings, trailing prose, and missing closers."""
    decoder = json.JSONDecoder(strict=False)
    try:
        return decoder.raw_decode(text)[0]
    except json.JSONDecodeError:
        pass
    # Mismatched closers (e.g. `1000})` inside code) confuse a plain parse; try closing what's open.
    for candidate in (_balance_json(text), _balance_json(text.rstrip().rstrip("}]").rstrip())):
        try:
            return decoder.raw_decode(candidate)[0]
        except json.JSONDecodeError:
            continue
    return None


def parse_prompted_tool_turn(raw: str, tool_names: set[str]) -> ToolTurn:
    """Extract {"tool_calls": [...]} from a model reply, tolerating fences, stray prose, raw newlines
    in strings, and unclosed brackets."""
    start = raw.find("{")
    if start == -1:
        return ToolTurn(text=raw.strip())
    data = _loads_lenient(raw[start:])
    if data is None:
        return ToolTurn(text=raw.strip())
    if not isinstance(data, dict):
        return ToolTurn(text=raw.strip())
    calls = []
    raw_calls = data.get("tool_calls")
    if raw_calls is None and "name" in data:  # a single bare call — common from small models
        raw_calls = [data]
    for i, c in enumerate(raw_calls or []):
        if isinstance(c, dict) and c.get("name") in tool_names:
            args = c.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            calls.append(ToolCall(name=c["name"], arguments=args if isinstance(args, dict) else {}, id=f"call_{i}"))
    return ToolTurn(tool_calls=calls)
