from __future__ import annotations

import json
import re
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


# Appended to the system prompt by the default (prompted) tool-calling implementation.
_PROMPTED_TOOLS_TEMPLATE = """

You can call these tools:
{tools}

Reply with ONLY a JSON object, no other text:
{{"tool_calls": [{{"name": "<tool name>", "arguments": {{...}}}}]}}
Call several tools at once when they are independent. When you already have enough
information, reply with {{"tool_calls": []}}."""


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

    def complete_with_tools(
        self, messages: list[dict], model_id: str, tools: list[dict], max_tokens: int = 1024,
        require: bool = False,
    ) -> ToolTurn:
        """Ask the model which tools to call next.

        `tools` are neutral specs: {"name", "description", "parameters": <JSON schema>}.
        `require` forces at least one tool call. This default works with any text model
        by asking for a JSON reply; providers with native tool calling override it.
        """
        listing = "\n".join(
            f"- {t['name']}: {t['description']} Arguments (JSON schema): {json.dumps(t['parameters'])}"
            for t in tools
        )
        suffix = _PROMPTED_TOOLS_TEMPLATE.format(tools=listing)
        if require:
            suffix += "\nYou must call at least one tool now."
        msgs = [dict(m) for m in messages]
        if msgs and msgs[0]["role"] == "system":
            msgs[0]["content"] += suffix
        else:
            msgs.insert(0, {"role": "system", "content": suffix.strip()})
        return parse_prompted_tool_turn(self.complete(msgs, model_id), {t["name"] for t in tools})

    @property
    def ready(self) -> bool:
        """False while the provider can't serve requests (e.g. local weights still loading)."""
        return True

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


def parse_prompted_tool_turn(raw: str, tool_names: set[str]) -> ToolTurn:
    """Extract {"tool_calls": [...]} from a model reply, tolerating fences and stray prose."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return ToolTurn(text=raw.strip())
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return ToolTurn(text=raw.strip())
    calls = []
    for i, c in enumerate(data.get("tool_calls") or []):
        if isinstance(c, dict) and c.get("name") in tool_names:
            args = c.get("arguments") if isinstance(c.get("arguments"), dict) else {}
            calls.append(ToolCall(name=c["name"], arguments=args, id=f"call_{i}"))
    return ToolTurn(tool_calls=calls)
