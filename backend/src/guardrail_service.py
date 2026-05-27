"""Content safety guardrail — classifies input and output messages via the cheapest LLM."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from src.llm_service import CALL_TYPE_GUARD, LLMService


@dataclass
class GuardrailResult:
    blocked: bool
    reason: str | None = None


def _build_prompt(base_prompt: str, denied_topics: list[str]) -> str:
    """Replace the {denied_section} placeholder with a bullet list of denied topics, or ''."""
    if denied_topics:
        section = "\nAdditionally block content that discusses any of these topics:\n" + \
                  "\n".join(f"- {t}" for t in denied_topics)
    else:
        section = ""
    return base_prompt.replace("{denied_section}", section)


class GuardrailService:
    def __init__(self, svc: LLMService, prompt: str, denied_topics: list[str]) -> None:
        self._svc = svc
        self._effective_prompt = _build_prompt(prompt, denied_topics)

    def check(self, text: str) -> GuardrailResult:
        """Evaluate text against safety rules.

        Returns GuardrailResult. Always fails open (allows content) on any error so
        a broken guardrail model never silently locks out the user.
        """
        messages = [
            {"role": "system", "content": self._effective_prompt},
            {"role": "user", "content": f"Text to evaluate:\n{text[:2000]}"},
        ]
        try:
            raw = self._svc.complete(messages, call_type=CALL_TYPE_GUARD).strip()
            # Strip optional markdown fences the model may wrap the JSON in
            if "```" in raw:
                raw = raw.split("```")[1].lstrip("json").strip()
            parsed = json.loads(raw)
            blocked = bool(parsed.get("blocked", False))
            reason = str(parsed["reason"]) if blocked and "reason" in parsed else None
            return GuardrailResult(blocked=blocked, reason=reason)
        except Exception:
            # Fail open: if the check itself errors, allow the content through
            return GuardrailResult(blocked=False)
