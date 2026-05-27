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

import time
from collections.abc import Iterator

from src.logger import get_logger
from src.providers.registry import ProviderRegistry

_log = get_logger(__name__)

# ── Call-type constants ───────────────────────────────────────────────────
# These strings appear verbatim in every log line — keep them snake_case.
# Add new entries here as the application grows; no other changes needed.
CALL_TYPE_CHAT   = "chat_response"    # main user↔assistant exchange
CALL_TYPE_TITLE  = "title_generation" # naming a new conversation
CALL_TYPE_MEMORY = "memory_update"    # extracting durable user facts
CALL_TYPE_GUARD  = "guardrail_check"  # content safety classification


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
            # Verify the override still exists; fall back silently if it doesn't.
            try:
                self._registry.get_provider(self._cheapest_override)
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
    ) -> Iterator[str]:
        """Stream a response token-by-token, logging start / done / error.

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
            for chunk in provider.stream(messages, model_id, image_base64=image_base64):
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
            result = provider.complete(messages, model_id)
            elapsed = time.monotonic() - t0
            tokens = provider.count_tokens(result)
            _log.info("[%s] done  | model=%s | tokens≈%d | %.1fs",
                      call_type, _short(model_id), tokens, elapsed)
            return result
        except Exception as exc:
            _log.error("[%s] error | model=%s | %s", call_type, _short(model_id), exc)
            raise

    def count_tokens(self, model_id: str, text: str) -> int:
        """Estimate token count via the provider for model_id."""
        provider = self._registry.get_provider(model_id)
        return provider.count_tokens(text)

    def extract_memory(self, user_message: str, current_memory: str) -> str | None:
        """Check whether user_message contains new durable facts about the user.

        Returns the updated memory string if the model decides something changed,
        or None if nothing new was found or on any error (always safe to ignore).
        """
        memory_block = current_memory.strip() or "(none)"
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a memory assistant. Maintain a lean user profile from conversation messages.\n\n"
                    "Capture: name, profession, location, tools/languages used, ongoing projects, "
                    "stated preferences, recurring constraints.\n"
                    "Skip: task instructions, questions, one-off or transient details.\n\n"
                    "Rules:\n"
                    "- Reply with ONLY the updated memory — plain text, one bullet per fact, no headers\n"
                    "- Keep it lean: ≤15 bullets, each under 15 words\n"
                    "- If nothing new: return the existing memory exactly unchanged\n"
                    "- If new info: add bullets; update or remove stale entries if relevant"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Existing memory:\n{memory_block}\n\n"
                    f"Latest message:\n{user_message[:600]}"
                ),
            },
        ]
        try:
            result = self.complete(messages, call_type=CALL_TYPE_MEMORY).strip()
            if result and result != current_memory.strip():
                return result
            return None
        except Exception:
            return None

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
