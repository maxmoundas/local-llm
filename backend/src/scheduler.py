"""Scheduled tasks: a prompt that runs on a schedule, each run posted to the schedule's own chat.

    data/schedules.json   [{id, name, prompt, model_id, project_id, kind, time, days, every_minutes,
                            enabled, chat_id, created_at, last_run_at, next_run_at, last_task_id}]

Kinds:
    daily     every day at `time` ("HH:MM", local time)
    weekly    on `days` (0 = Monday … 6 = Sunday) at `time`
    interval  every `every_minutes` minutes (at least 5)

A background thread checks every few seconds and starts due runs through `start_run`, which the
server supplies (it starts an ordinary chat turn as a background task). Runs missed while the
server was down run once at startup, then the schedule continues from now — no catch-up storm.
"""
from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

from src.logger import get_logger

_log = get_logger(__name__)

SCHEDULES_PATH = Path("data/schedules.json")
KINDS = ("daily", "weekly", "interval")
MIN_INTERVAL = 5
_lock = threading.RLock()


def _load() -> list[dict]:
    try:
        return json.loads(SCHEDULES_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []


def _save(items: list[dict]) -> None:
    SCHEDULES_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = SCHEDULES_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(items, indent=2), encoding="utf-8")
    tmp.replace(SCHEDULES_PATH)


def _parse_time(value: str) -> tuple[int, int]:
    try:
        h, m = (int(x) for x in str(value).split(":", 1))
    except ValueError as exc:
        raise ValueError(f"time must be HH:MM, got {value!r}") from exc
    if not (0 <= h < 24 and 0 <= m < 60):
        raise ValueError(f"time must be HH:MM, got {value!r}")
    return h, m


def validate(s: dict) -> None:
    if s.get("kind") not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    if not str(s.get("prompt", "")).strip():
        raise ValueError("prompt is required")
    if s["kind"] in ("daily", "weekly"):
        _parse_time(s.get("time", ""))
    if s["kind"] == "weekly":
        days = s.get("days") or []
        if not days or any(not isinstance(d, int) or not 0 <= d <= 6 for d in days):
            raise ValueError("weekly schedules need days: a list of 0 (Mon) … 6 (Sun)")
    if s["kind"] == "interval" and int(s.get("every_minutes") or 0) < MIN_INTERVAL:
        raise ValueError(f"every_minutes must be at least {MIN_INTERVAL}")


def next_run(s: dict, after: datetime) -> datetime:
    """The first run time strictly after `after` (naive local time)."""
    if s["kind"] == "interval":
        return after + timedelta(minutes=max(MIN_INTERVAL, int(s["every_minutes"])))
    h, m = _parse_time(s["time"])
    days = set(range(7)) if s["kind"] == "daily" else set(s["days"])
    candidate = after.replace(hour=h, minute=m, second=0, microsecond=0)
    for _ in range(8):
        if candidate > after and candidate.weekday() in days:
            return candidate
        candidate = (candidate + timedelta(days=1)).replace(hour=h, minute=m)
    raise ValueError("no valid run time")  # unreachable for validated schedules


def describe(s: dict) -> str:
    if s["kind"] == "interval":
        n = int(s["every_minutes"])
        return f"Every {n // 60} h" if n % 60 == 0 else f"Every {n} min"
    if s["kind"] == "daily":
        return f"Daily at {s['time']}"
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    days = sorted(s["days"])
    label = "Weekdays" if days == [0, 1, 2, 3, 4] else ", ".join(names[d] for d in days)
    return f"{label} at {s['time']}"


def _public(s: dict) -> dict:
    return {**s, "description": describe(s)}


# ── CRUD ────────────────────────────────────────────────────────────────

_FIELDS = ("name", "prompt", "model_id", "project_id", "kind", "time", "days", "every_minutes", "enabled")


def list_schedules() -> list[dict]:
    with _lock:
        return [_public(s) for s in _load()]


def get(schedule_id: str) -> dict | None:
    with _lock:
        return next((_public(s) for s in _load() if s["id"] == schedule_id), None)


def create(data: dict, now: datetime | None = None) -> dict:
    s = {k: data.get(k) for k in _FIELDS}
    s["enabled"] = data.get("enabled", True) is not False
    s["name"] = (s.get("name") or "").strip() or str(s.get("prompt", ""))[:40]
    validate(s)
    now = now or datetime.now()
    s.update(id=uuid.uuid4().hex[:12], chat_id=None, created_at=now.isoformat(timespec="seconds"),
             last_run_at=None, last_task_id=None, next_run_at=next_run(s, now).isoformat(timespec="seconds"))
    with _lock:
        items = _load()
        items.append(s)
        _save(items)
    return _public(s)


def update(schedule_id: str, data: dict, now: datetime | None = None) -> dict | None:
    with _lock:
        items = _load()
        s = next((x for x in items if x["id"] == schedule_id), None)
        if s is None:
            return None
        merged = {**s, **{k: v for k, v in data.items() if k in _FIELDS}}
        validate(merged)
        if any(k in data for k in ("kind", "time", "days", "every_minutes", "enabled")):
            merged["next_run_at"] = next_run(merged, now or datetime.now()).isoformat(timespec="seconds")
        s.clear()
        s.update(merged)
        _save(items)
        return _public(s)


def delete(schedule_id: str) -> bool:
    with _lock:
        items = _load()
        kept = [s for s in items if s["id"] != schedule_id]
        if len(kept) == len(items):
            return False
        _save(kept)
        return True


def mark_run(schedule_id: str, chat_id: str | None, task_id: str | None, now: datetime | None = None) -> None:
    now = now or datetime.now()
    with _lock:
        items = _load()
        for s in items:
            if s["id"] == schedule_id:
                s["last_run_at"] = now.isoformat(timespec="seconds")
                s["next_run_at"] = next_run(s, now).isoformat(timespec="seconds")
                if chat_id:
                    s["chat_id"] = chat_id
                s["last_task_id"] = task_id
        _save(items)


def due(now: datetime) -> list[dict]:
    with _lock:
        return [s for s in _load() if s.get("enabled") and s.get("next_run_at")
                and datetime.fromisoformat(s["next_run_at"]) <= now]


# ── Runner ──────────────────────────────────────────────────────────────

class Scheduler:
    """Starts due schedules on a background thread.

    `start_run(schedule) -> (chat_id, task_id)` starts one run (the server's implementation posts
    the prompt to the schedule's chat as a background task) or raises to skip it.
    """

    def __init__(self, start_run: Callable[[dict], tuple[str | None, str | None]], interval: float = 15.0) -> None:
        self._start_run = start_run
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True, name="scheduler")
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def tick(self, now: datetime | None = None) -> list[str]:
        """Start every due schedule once. Returns the ids started."""
        now = now or datetime.now()
        started = []
        for s in due(now):
            try:
                chat_id, task_id = self._start_run(s)
                started.append(s["id"])
                _log.info("Scheduled run started: %s (%s)", s["name"], s["id"])
            except Exception as exc:
                chat_id, task_id = None, None
                _log.warning("Scheduled run %s skipped: %s", s["id"], exc)
            mark_run(s["id"], chat_id, task_id, now)
        return started

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self.tick()
            except Exception:
                _log.exception("Scheduler tick failed")
