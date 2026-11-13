"""Token and cost accounting for LLM calls.

Providers call `record()` after every request with the real token counts the API
returned. Whoever wants a total opens a meter around the work:

    with usage.meter() as m:
        ...  # any LLM calls on this thread
    m.cost_usd, m.input_tokens, ...

Meters live in a ContextVar, so they follow the thread (or task) that opened them and
nest: an inner meter's usage is also added to every outer meter. A task's background
thread opens one meter for the whole task, which is how per-task budgets are enforced.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

_active: ContextVar[tuple["Meter", ...]] = ContextVar("usage_meters", default=())
_prices: dict[str, tuple[float, float]] = {}  # model_id -> ($/1M input, $/1M output)
_prices_lock = threading.Lock()


def set_price(model_id: str, input_per_mtok: float, output_per_mtok: float) -> None:
    with _prices_lock:
        _prices[model_id] = (float(input_per_mtok), float(output_per_mtok))


CACHE_READ_RATE = 0.1    # cached input reads: ~90% off (Anthropic and OpenAI)
CACHE_WRITE_RATE = 1.25  # Anthropic bills writing a prompt-cache entry (5-minute TTL) at 125% of input


def cost(model_id: str, input_tokens: int, output_tokens: int, cached_input_tokens: int = 0,
         cache_write_tokens: int = 0) -> float:
    """Dollar cost of one call. `input_tokens` is the whole prompt, including cache reads and writes."""
    pin, pout = _prices.get(model_id, (0.0, 0.0))
    fresh = max(0, input_tokens - cached_input_tokens - cache_write_tokens)
    return (fresh * pin + cached_input_tokens * pin * CACHE_READ_RATE + cache_write_tokens * pin * CACHE_WRITE_RATE
            + output_tokens * pout) / 1_000_000


@dataclass
class Meter:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0
    by_model: dict[str, dict] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, model_id: str, input_tokens: int, output_tokens: int, cached_input_tokens: int = 0,
            cache_write_tokens: int = 0) -> None:
        c = cost(model_id, input_tokens, output_tokens, cached_input_tokens, cache_write_tokens)
        with self._lock:
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.cached_input_tokens += cached_input_tokens
            self.cost_usd += c
            self.calls += 1
            m = self.by_model.setdefault(model_id, {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "calls": 0})
            m["input_tokens"] += input_tokens
            m["output_tokens"] += output_tokens
            m["cost_usd"] += c
            m["calls"] += 1

    def full_snapshot(self) -> dict:
        """snapshot() plus the per-model breakdown, as stored for the usage dashboard."""
        with self._lock:
            by_model = {k: dict(v) for k, v in self.by_model.items()}
        return {**self.snapshot(), "by_model": by_model}

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "cached_input_tokens": self.cached_input_tokens,
                "cost_usd": round(self.cost_usd, 6),
                "calls": self.calls,
            }


def record(model_id: str, input_tokens: int, output_tokens: int, cached_input_tokens: int = 0,
           cache_write_tokens: int = 0) -> None:
    """Add one call's usage to every meter open on this context. No-op when none is open."""
    for m in _active.get():
        m.add(model_id, int(input_tokens or 0), int(output_tokens or 0), int(cached_input_tokens or 0),
              int(cache_write_tokens or 0))


@contextmanager
def meter(existing: Meter | None = None):
    m = existing or Meter()
    token = _active.set(_active.get() + (m,))
    try:
        yield m
    finally:
        _active.reset(token)


def current() -> tuple[Meter, ...]:
    """The meters open on this context — pass to worker threads with `attach()`."""
    return _active.get()


@contextmanager
def attach(meters: tuple[Meter, ...]):
    """Re-open a parent's meters on another thread (ContextVars don't cross thread pools)."""
    token = _active.set(meters)
    try:
        yield
    finally:
        _active.reset(token)
