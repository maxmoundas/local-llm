"""Decisions for the research loop: whether to research at all, and when to stop.

Both use Jev when it is available and enabled for routing, and fall back to the
cheapest LLM (route) or to the planner model's own judgement (sufficiency).
Thresholds were tuned against a 24-prompt labeled set: with the state below,
research prompts scored 0.82–0.98 and non-research prompts 0.02–0.07.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from src.decision_service import (
    CALL_TYPE_ROUTE, CALL_TYPE_SUFFICIENT, DecisionError, DecisionService, noul,
)
from src.llm_service import CALL_TYPE_ROUTE as LLM_CALL_TYPE_ROUTE, LLMService
from src.logger import get_logger

_log = get_logger(__name__)

ROUTE_THRESHOLD = 0.5
ENOUGH_THRESHOLD = 0.75

_ASSISTANT_DESCRIPTION = (
    "An AI assistant with a fixed training cutoff. It has no live data unless it uses "
    "web search, news, weather, and page-fetch tools."
)
_CODE_CAPABILITY = (" It can also run code in a sandbox to calculate exactly, analyze data, test programs, "
                    "and create files for the user.")

_NEEDS_RESEARCH = noul(
    "Handling the latest message well requires the assistant to use its tools: looking up current, recent, "
    "local, or specific external information (news, weather, prices, scores, recent releases, live status, or "
    "the contents of a URL), running code for exact calculations, data work, or files, or acting on one of its "
    "connected services — rather than answering from its training data alone.",
    true="Needs tools: fresh or external information, running code or producing files, or a connected service.",
    false="Can be fully answered from general knowledge, reasoning, writing, or coding skill alone.",
)

_HAS_ENOUGH = noul(
    "The research notes contain enough current, specific, relevant information to give a complete, "
    "well-supported answer to every part of the question.",
    true="Every part of the question is covered by specific facts in the notes.",
    false="Some part of the question is missing, vague, or unsupported by the notes.",
)

_LLM_ROUTE_PROMPT = (
    "Decide if an AI assistant must use tools to handle the user's latest message well: search the web or "
    "use live data (news, weather, prices, recent events, a specific URL){extra}. Today is {today}. "
    "Reply with ONLY 'yes' or 'no'."
)


@dataclass
class RouteDecision:
    research: bool
    probability: float | None = None
    source: str = "jev"  # "jev" | "llm"


def _today() -> str:
    return datetime.now().strftime("%A, %B %d, %Y")


def _recent(history: list[dict], limit: int = 4, chars: int = 500) -> list[dict]:
    turns = [m for m in history if m.get("role") in ("user", "assistant")][-limit:]
    return [{"role": m["role"], "content": str(m.get("content", ""))[:chars]} for m in turns]


def _first_yes_no(raw: str) -> bool:
    """True when the first yes/no word in the reply is 'yes' (tolerates 'Yes.', '**Yes**', 'Answer: yes')."""
    m = re.search(r"\b(yes|no)\b", raw.lower())
    return bool(m and m.group(1) == "yes")


class ResearchRouter:
    def __init__(self, llm: LLMService, decisions: DecisionService | None, use_jev: bool = True,
                 can_run_code: bool = False, connectors: list[str] | None = None) -> None:
        self._llm = llm
        self._decisions = decisions
        self._use_jev = use_jev
        self._code = can_run_code
        self._connectors = list(connectors or [])

    def _description(self) -> str:
        text = _ASSISTANT_DESCRIPTION + (_CODE_CAPABILITY if self._code else "")
        if self._connectors:
            text += f" Connected services it can use: {', '.join(self._connectors)}."
        return text

    def _llm_extra(self) -> str:
        extra = ""
        if self._code:
            extra += ", run code for exact calculations, data analysis, or to create files"
        if self._connectors:
            extra += f", or use a connected service ({', '.join(self._connectors)})"
        return extra

    @property
    def _jev(self) -> DecisionService | None:
        if self._use_jev and self._decisions is not None and self._decisions.available:
            return self._decisions
        return None

    def decide(self, latest: str, history: list[dict]) -> RouteDecision:
        """Should this message go through the research loop? Fails closed (no research) on errors."""
        if jev := self._jev:
            state = {
                "today": _today(),
                "assistant": self._description(),
                "recent_conversation": _recent(history),
                "latest_message": latest[:2000],
            }
            try:
                p = jev.ask(state, {"needs_research": _NEEDS_RESEARCH}, call_type=CALL_TYPE_ROUTE)["needs_research"].noul
                return RouteDecision(research=p >= ROUTE_THRESHOLD, probability=p, source="jev")
            except DecisionError:
                pass  # fall through to the LLM

        # The message is wrapped as data with the question after it: small models otherwise
        # just answer the message instead of classifying it.
        messages = [
            {"role": "system", "content": _LLM_ROUTE_PROMPT.format(today=_today(), extra=self._llm_extra())},
            {"role": "user", "content": f"<message>\n{latest[:2000]}\n</message>\n\n"
                                        "Does handling this message need tools? Answer only yes or no."},
        ]
        try:
            raw = self._llm.complete(messages, call_type=LLM_CALL_TYPE_ROUTE, max_tokens=5)
            return RouteDecision(research=_first_yes_no(raw), source="llm")
        except Exception as exc:
            _log.warning("Research routing failed, answering directly: %s", exc)
            return RouteDecision(research=False, source="llm")

    def enough(self, question: str, notes: str) -> float | None:
        """Probability the notes already answer the question, or None when Jev can't judge."""
        jev = self._jev
        if jev is None:
            return None
        state = {"today": _today(), "question": question[:2000], "research_notes": notes[-20000:]}
        try:
            return jev.ask(state, {"has_enough": _HAS_ENOUGH}, call_type=CALL_TYPE_SUFFICIENT)["has_enough"].noul
        except DecisionError:
            return None
