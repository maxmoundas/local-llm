"""Central service for all LLM calls in the application.

Every inference request — streaming chat, non-streaming utility calls, title
generation — goes through LLMService so routing, logging, and fallback logic
live in one place. Nothing in server.py should call a provider directly.

Adding a new call type (tool execution, guardrail check, web-search summary…):
  1. Add a constant below (e.g. CALL_TYPE_TOOL = "tool_call")
  2. Call self.complete(..., call_type=CALL_TYPE_TOOL) or self.stream(...)
  All logging, timing, and error handling is inherited automatically.
"""
from __future__ import annotations

import re
import time
from collections.abc import Iterator

from src.logger import get_logger
from src.providers.base import ToolTurn
from src.providers.registry import ProviderRegistry

_log = get_logger(__name__)

# ── Call-type constants ───────────────────────────────────────────────────
# These strings appear verbatim in every log line — keep them snake_case.
# Add new entries here as the application grows; no other changes needed.
CALL_TYPE_CHAT   = "chat_response"    # main user↔assistant exchange
CALL_TYPE_TITLE  = "title_generation" # naming a new conversation
CALL_TYPE_MEMORY = "memory_update"    # extracting durable user facts
CALL_TYPE_GUARD  = "guardrail_check"  # content safety classification
CALL_TYPE_AGENT  = "agent_step"       # one tool-selection turn in the research loop
CALL_TYPE_ROUTE  = "research_routing" # LLM fallback when Jev is unavailable
CALL_TYPE_COMPACT = "agent_compaction" # summarizing old work-log entries in long tasks
CALL_TYPE_HISTORY = "history_summary"  # summarizing old chat turns that no longer fit the context window


_MEMORY_MSG_CHARS = 2000
_MEMORY_MAX_FACTS = 20
_NO_CHANGE = "NO_CHANGE"
_FORGET_RE = re.compile(r"\b(forget|remove|delete|erase|clear|wipe|don'?t remember|stop remembering)\b", re.I)

_MEMORY_SYSTEM = f"""You maintain a short profile of durable facts about the user, used to personalize future chats.

Capture: name, profession, location, languages/tools/stack, ongoing projects, stated preferences \
(e.g. answer style, units), recurring constraints.
Skip: questions, task instructions, one-off or temporary details, anything about third parties, \
secrets (passwords, keys, account numbers), and sensitive details (health, finances, etc.) unless \
the user explicitly asks you to remember them.

Rules:
- If the user explicitly asks to remember something, add it. If they ask to forget something or \
correct a fact, remove or fix it.
- If nothing should change, reply with exactly: {_NO_CHANGE}
- Otherwise reply with the COMPLETE updated profile: one "- " bullet per fact, under 15 words each, \
at most {_MEMORY_MAX_FACTS} bullets. Keep every existing fact that is still true; merge duplicates; \
replace facts the new message makes outdated.
- No headers, preamble, or commentary."""


def _parse_memory_reply(raw: str, current: str, user_message: str) -> str | None:
    """Validate the memory model's reply; return new memory text or None to keep the current one."""
    text = (raw or "").strip().strip("`").strip()
    if not text or _NO_CHANGE in text.upper():
        return None
    facts = []
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r"^(?:[-*•]|\d+[.)])\s+(.*)$", line)
        if m and m.group(1).strip() and m.group(1).strip().lower() not in ("(empty)", "(none)", "none"):
            facts.append(f"- {m.group(1).strip()}")
    if not facts:
        # No bullets at all: either an empty profile after a "forget everything" request,
        # or the model ignored the format. Only accept the former.
        return "" if current.strip() and _FORGET_RE.search(user_message) else None
    facts = list(dict.fromkeys(facts))[:_MEMORY_MAX_FACTS]
    old = [ln for ln in current.splitlines() if ln.strip()]
    # Small utility models sometimes return just the new fact. Refuse a big drop unless asked to forget.
    if len(old) >= 3 and len(facts) < len(old) / 2 and not _FORGET_RE.search(user_message):
        _log.warning("[%s] rejected update that dropped %d of %d facts", CALL_TYPE_MEMORY,
                     len(old) - len(facts), len(old))
        return None
    result = "\n".join(facts)
    return None if result == current.strip() else result


def _short(model_id: str) -> str:
    """Return the leaf segment of a model ID for concise log lines."""
    return model_id.split("/")[-1]


def _title_fallback(text: str) -> str:
    """Truncate at word boundary — used when LLM title generation fails."""
    text = text.strip()
    if len(text) > 45:
        return text[:46].rsplit(" ", 1)[0].rstrip(",.;:") + "…"
    return text


class LLMService:
    def __init__(self, registry: ProviderRegistry, cheapest_model_id: str | None = None) -> None:
        self._registry = registry
        # Overrides the is_cheapest flag when set (written by the settings page).
        self._cheapest_override: str | None = cheapest_model_id

    def set_cheapest_model(self, model_id: str | None) -> None:
        """Update the cheapest-model override at runtime (called when settings change)."""
        self._cheapest_override = model_id

    def _resolve_cheapest_id(self) -> str:
        """Return the effective cheapest model ID, honouring any saved override."""
        if self._cheapest_override:
            # Use the override only if it still exists and can serve requests right now
            # (a local model that is still loading would fail every utility call).
            try:
                if self._registry.get_provider(self._cheapest_override).ready:
                    return self._cheapest_override
            except KeyError:
                pass
        return self._registry.cheapest_model.id

    # ── Public interface ──────────────────────────────────────────────────

    def stream(
        self,
        messages: list[dict],
        model_id: str,
        image_base64: str | None = None,
        call_type: str = CALL_TYPE_CHAT,
        context_id: str = "",
        options: dict | None = None,
    ) -> Iterator[str]:
        """Stream a response token-by-token, logging start / done / error.

        `options` holds per-chat settings ({"temperature", "effort"}); providers apply what the model supports.

        context_id is an optional short label (e.g. chat UUID) included in
        log lines so individual conversations can be traced in the log file.
        """
        provider = self._registry.get_provider(model_id)
        ctx = f"id={context_id[:8]} | " if context_id else ""
        all_text = " ".join(
            m["content"] if isinstance(m.get("content"), str) else ""
            for m in messages
        )
        prompt_tokens = provider.count_tokens(all_text)

        _log.info("[%s] start | %smodel=%s | prompt_tokens≈%d",
                  call_type, ctx, _short(model_id), prompt_tokens)
        t0 = time.monotonic()
        chunks: list[str] = []

        try:
            kwargs = {"options": options} if options else {}
            for chunk in provider.stream(messages, model_id, image_base64=image_base64, **kwargs):
                chunks.append(chunk)
                yield chunk
        except Exception as exc:
            _log.error("[%s] error | %smodel=%s | %s",
                       call_type, ctx, _short(model_id), exc)
            raise
        else:
            elapsed = time.monotonic() - t0
            tokens = provider.count_tokens("".join(chunks))
            _log.info("[%s] done  | %smodel=%s | tokens≈%d | %.1fs",
                      call_type, ctx, _short(model_id), tokens, elapsed)

    def complete(
        self,
        messages: list[dict],
        model_id: str | None = None,
        call_type: str = CALL_TYPE_TITLE,
        max_tokens: int | None = None,
    ) -> str:
        """Non-streaming completion with full lifecycle logging.

        Uses the cheapest registered model when model_id is None, making it
        the right call for any internal utility request.
        """
        if model_id is None:
            model_id = self._resolve_cheapest_id()
        provider = self._registry.get_provider(model_id)

        _log.info("[%s] start | model=%s", call_type, _short(model_id))
        t0 = time.monotonic()
        try:
            if max_tokens is None:
                result = provider.complete(messages, model_id)
            else:
                result = provider.complete(messages, model_id, max_tokens=max_tokens)
            elapsed = time.monotonic() - t0
            tokens = provider.count_tokens(result)
            _log.info("[%s] done  | model=%s | tokens≈%d | %.1fs",
                      call_type, _short(model_id), tokens, elapsed)
            return result
        except Exception as exc:
            _log.error("[%s] error | model=%s | %s", call_type, _short(model_id), exc)
            raise

    def complete_with_tools(
        self,
        messages: list[dict],
        model_id: str,
        tools: list[dict],
        call_type: str = CALL_TYPE_AGENT,
        context_id: str = "",
        require: bool = False,
    ) -> ToolTurn:
        """One tool-selection turn: returns the tool calls the model wants to make next.

        `require` forces at least one call (used on the first step, once the router has
        already decided the message needs research).
        """
        provider = self._registry.get_provider(model_id)
        ctx = f"id={context_id[:8]} | " if context_id else ""
        _log.info("[%s] start | %smodel=%s | tools=%d", call_type, ctx, _short(model_id), len(tools))
        t0 = time.monotonic()
        try:
            turn = provider.complete_with_tools(messages, model_id, tools, require=require)
        except Exception as exc:
            _log.error("[%s] error | %smodel=%s | %s", call_type, ctx, _short(model_id), exc)
            raise
        _log.info("[%s] done  | %smodel=%s | calls=%s | %.1fs", call_type, ctx, _short(model_id),
                  [c.name for c in turn.tool_calls], time.monotonic() - t0)
        return turn

    def model_info(self, model_id: str):
        """The catalog entry (capabilities, context window, prices) for model_id."""
        return self._registry.get_model(model_id)

    def count_tokens(self, model_id: str, text: str) -> int:
        """Estimate token count via the provider for model_id."""
        provider = self._registry.get_provider(model_id)
        return provider.count_tokens(text)

    def extract_memory(self, user_message: str, current_memory: str, prev_reply: str = "") -> str | None:
        """Check whether user_message contains new durable facts about the user.

        prev_reply is the assistant message the user was answering, so short replies
        ("Boston", "yes, Rust") can be understood. Returns the updated memory string if
        something changed, or None if nothing new / the output looked wrong / on any error
        (always safe to ignore).
        """
        context = f"Assistant's previous message (context only):\n{prev_reply[-500:]}\n\n" if prev_reply else ""
        messages = [
            {"role": "system", "content": _MEMORY_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Existing memory:\n{current_memory.strip() or '(empty)'}\n\n"
                    f"{context}"
                    f"User's latest message:\n{user_message[:_MEMORY_MSG_CHARS]}"
                ),
            },
        ]
        try:
            raw = self.complete(messages, call_type=CALL_TYPE_MEMORY)
        except Exception:
            return None
        return _parse_memory_reply(raw, current_memory, user_message)

    def summarize_history(self, messages: list[dict], previous_summary: str = "") -> str:
        """Condense older chat turns into a short brief, so a long chat fits the model's context window."""
        transcript = "\n\n".join(f"{m['role'].upper()}: {m['content'][:4000]}" for m in messages if m.get("content"))
        prompt = (
            (f"Summary of the conversation so far:\n{previous_summary}\n\n" if previous_summary else "")
            + f"Further conversation:\n{transcript[-60000:]}"
        )
        result = self.complete([
            {"role": "system", "content": (
                "Summarize this conversation so it can continue without the original messages. Keep: the user's "
                "goals and constraints, decisions made, facts, names, numbers, code or file names that later "
                "messages may refer to, and open questions. Bullet points, under 400 words. No preamble.")},
            {"role": "user", "content": prompt},
        ], call_type=CALL_TYPE_HISTORY, max_tokens=1200)
        return result.strip()

    def generate_title_from_history(self, messages: list[dict]) -> str:
        """Title a conversation from its messages (used by the "Generate title" action)."""
        turns = [m for m in messages if m.get("role") in ("user", "assistant") and m.get("content")]
        if not turns:
            return "New Chat"
        transcript = "\n".join(f"{m['role']}: {m['content'][:400]}" for m in turns[:12])
        try:
            raw = self.complete([
                {"role": "system", "content": (
                    "Generate an ultra-short title for this conversation, covering its main topic. "
                    "Reply with ONLY the title — no quotes, no punctuation, 6 words max.")},
                {"role": "user", "content": transcript[:6000]},
            ], call_type=CALL_TYPE_TITLE)
            title = raw.strip().strip("\"'").strip()[:60]
        except Exception:
            title = ""
        return title or _title_fallback(turns[0]["content"])

    def generate_title(self, user_message: str) -> str:
        """Generate a short chat title using the cheapest available model.

        Falls back to word-boundary truncation if the LLM call fails
        (e.g. model still loading, API error, empty response).
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "Generate ultra-short chat titles. "
                    "Reply with ONLY the title — no quotes, no punctuation, 5 words max."
                ),
            },
            {
                "role": "user",
                "content": f"Title for: {user_message[:300]}",
            },
        ]
        try:
            raw = self.complete(messages, call_type=CALL_TYPE_TITLE)
            title = raw.strip().strip("\"'").strip()[:60] or _title_fallback(user_message)
        except Exception:
            title = _title_fallback(user_message)

        _log.info("[%s] result=%r", CALL_TYPE_TITLE, title)
        return title
