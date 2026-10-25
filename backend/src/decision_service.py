"""Central service for structured decisions via Jev (TypeSafe's System One model).

Jev does not generate text. It takes a *state* (a string or JSON value) and a
set of typed questions, and returns one typed answer per question:

  noul(...)    → probability (0–1) that a statement is true
  choice(...)  → one option from a fixed set, with per-option probabilities
  score(...)   → a position on an ordered rubric (0 … n-1), with probabilities

Use DecisionService for any classify / route / gate step (guardrails, research
routing, "do we have enough info?", memory gating). Use LLMService for anything
that has to write text. Every caller must handle DecisionError and fall back,
because Jev can be disabled in settings or unreachable.

Calling from anywhere:
    from src import decision_service
    svc = decision_service.get()          # None when no API key is configured
    if svc and svc.available:
        answers = svc.ask(state, {"urgent": noul("The message is urgent")})
        answers["urgent"].noul            # → 0.93

Docs: https://docs.typesafe.ai/introduction
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from src.logger import get_logger

_log = get_logger(__name__)

JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"

# ── Call-type constants (mirrors llm_service — these appear in log lines) ──
CALL_TYPE_ROUTE      = "research_routing"
CALL_TYPE_SUFFICIENT = "research_sufficiency"
CALL_TYPE_GUARD      = "guardrail_check"
CALL_TYPE_MEMORY     = "memory_gate"
CALL_TYPE_API        = "api_decide"

_RETRY_STATUS = {429, 500, 502, 503, 504}


# ── Question builders ─────────────────────────────────────────────────────

def noul(instructions: str, true: str | None = None, false: str | None = None) -> dict:
    """A yes/no question. Optional `true`/`false` describe what each answer looks like."""
    q: dict[str, Any] = {"type": "noul", "instructions": instructions}
    if true is not None and false is not None:
        q["criteria"] = {"true": true, "false": false}
    return q


def choice(instructions: str, criteria: dict[str, str]) -> dict:
    """Pick one option. `criteria` maps option id → description."""
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def score(instructions: str, levels: list[str]) -> dict:
    """Rate on an ordered rubric. `levels` run from lowest (0) to highest (n-1)."""
    return {"type": "score", "instructions": instructions, "criteria": levels}


# ── Answers ───────────────────────────────────────────────────────────────

@dataclass
class Answer:
    type: str
    value: Any                      # float for noul/score, str for choice
    confidence: float | None = None  # None for noul (use |2p-1| if needed)
    probabilities: dict[str, float] = field(default_factory=dict)

    @property
    def noul(self) -> float:
        return float(self.value)

    @property
    def choice(self) -> str:
        return str(self.value)

    @property
    def score(self) -> float:
        return float(self.value)

    @classmethod
    def from_api(cls, raw: dict) -> "Answer":
        kind = raw.get("type", "")
        return cls(
            type=kind,
            value=raw.get(kind),
            confidence=raw.get("confidence"),
            probabilities=raw.get("probabilities") or {},
        )


class DecisionError(Exception):
    """Jev is disabled, unconfigured, or the call failed. Callers should fall back."""


# ── Service ───────────────────────────────────────────────────────────────

class DecisionService:
    def __init__(
        self,
        api_key: str | None,
        model: str = DEFAULT_MODEL,
        enabled: bool = True,
        timeout: float = 15.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._api_key = api_key
        self.model = model
        self.enabled = enabled
        self._client = client or httpx.Client(timeout=timeout)

    @property
    def configured(self) -> bool:
        return bool(self._api_key)

    @property
    def available(self) -> bool:
        return self.configured and self.enabled

    def ask(
        self,
        state: Any,
        questions: dict[str, dict],
        call_type: str = CALL_TYPE_API,
    ) -> dict[str, Answer]:
        """Evaluate every question against `state` in one Jev call.

        Raises DecisionError if Jev is unavailable or the request fails after one retry.
        """
        if not self.available:
            raise DecisionError("Jev is not enabled")

        payload = {"model": self.model, "state": state, "questions": questions}
        headers = {"Authorization": f"Bearer {self._api_key}"}
        _log.info("[%s] start | model=%s | questions=%d", call_type, self.model, len(questions))
        t0 = time.monotonic()

        last_error: Exception | None = None
        for attempt in range(2):
            try:
                resp = self._client.post(JEV_ENDPOINT, json=payload, headers=headers)
                if resp.status_code in _RETRY_STATUS and attempt == 0:
                    last_error = DecisionError(f"HTTP {resp.status_code}")
                    time.sleep(0.5)
                    continue
                if resp.status_code != 200:
                    raise DecisionError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                data = resp.json()
                answers = {qid: Answer.from_api(a) for qid, a in data.get("answers", {}).items()}
                missing = set(questions) - set(answers)
                if missing:
                    raise DecisionError(f"Missing answers: {sorted(missing)}")
                _log.info("[%s] done  | model=%s | %s | %.2fs", call_type, data.get("model", self.model),
                          _summarise(answers), time.monotonic() - t0)
                return answers
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt == 0:
                    continue
            except DecisionError as exc:
                last_error = exc
                break

        _log.error("[%s] error | model=%s | %s", call_type, self.model, last_error)
        raise DecisionError(str(last_error))


def _summarise(answers: dict[str, Answer]) -> str:
    """Compact `name=value` list for log lines."""
    parts = []
    for qid, a in answers.items():
        if a.type == "choice":
            parts.append(f"{qid}={a.value}({a.confidence:.2f})" if a.confidence is not None else f"{qid}={a.value}")
        else:
            parts.append(f"{qid}={float(a.value):.2f}")
    return " ".join(parts)


# ── Process-wide instance ─────────────────────────────────────────────────

_instance: DecisionService | None = None


def configure(service: DecisionService | None) -> None:
    global _instance
    _instance = service


def get() -> DecisionService | None:
    return _instance
