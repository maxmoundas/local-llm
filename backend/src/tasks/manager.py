"""Background task runtime: tasks run on their own threads, independent of any HTTP request.

  start(chat_id, model_id, fn)  run fn(ctx) in the background, return the task id
  subscribe(task_id, after)     iterate the task's events from seq > after; blocks for new ones
                                until the task ends (reconnects pass the last seq they saw)
  cancel(task_id)               ask the task to stop at its next checkpoint
  reply(task_id, payload)       answer a pending question / approval

Every event gets a seq. Live tasks keep all events in memory (including token deltas, so a
reconnect mid-answer replays the partial text); non-token events are also written to SQLite.
Finished tasks stay in memory for a few minutes, then are served from the database.
"""
from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

from src import usage
from src.logger import get_logger
from src.tasks import store

_log = get_logger(__name__)

_KEEP_FINISHED_S = 600
_UNPERSISTED = {"token"}
TERMINAL = "task_end"


class Cancelled(Exception):
    """Raised inside a task when it is cancelled while waiting for the user."""


@dataclass
class _Live:
    task_id: str
    chat_id: str | None
    events: list[dict] = field(default_factory=list)
    cond: threading.Condition = field(default_factory=threading.Condition)
    cancel: threading.Event = field(default_factory=threading.Event)
    finished_at: float | None = None
    pending: dict | None = None             # open question/approval request
    reply: dict | None = None
    replied: threading.Event = field(default_factory=threading.Event)
    on_close: list[Callable[[], None]] = field(default_factory=list)


class TaskContext:
    """What a running task sees: emit events, check for cancel, ask the user, track spend."""

    def __init__(self, live: _Live, manager: "TaskManager") -> None:
        self._live = live
        self._mgr = manager
        self.meter: usage.Meter = usage.Meter()

    @property
    def task_id(self) -> str:
        return self._live.task_id

    @property
    def chat_id(self) -> str | None:
        return self._live.chat_id

    @property
    def cancelled(self) -> bool:
        return self._live.cancel.is_set()

    def emit(self, type_: str, **data) -> int:
        return self._mgr._emit(self._live, type_, data)

    def on_close(self, fn: Callable[[], None]) -> None:
        """Run fn when the task ends (e.g. stop its sandbox container)."""
        self._live.on_close.append(fn)

    def ask(self, request: dict, timeout: float | None = None) -> dict:
        """Pause until the user replies (or the task is cancelled). Returns the reply payload."""
        live = self._live
        request = {"id": uuid.uuid4().hex[:10], **request}
        live.reply, live.pending = None, request
        live.replied.clear()
        store.set_status(live.task_id, "waiting")
        self.emit("input_request", request=request)
        deadline = time.monotonic() + timeout if timeout else None
        while not live.replied.wait(0.25):
            if live.cancel.is_set():
                live.pending = None
                raise Cancelled()
            if deadline and time.monotonic() > deadline:
                break
        reply, live.pending = live.reply or {}, None
        store.set_status(live.task_id, "running")
        self.emit("input_resolved", request_id=request["id"], reply=reply)
        return reply


class TaskManager:
    def __init__(self) -> None:
        self._live: dict[str, _Live] = {}
        self._lock = threading.Lock()

    def recover(self) -> list[str]:
        """Mark tasks orphaned by a previous process as interrupted."""
        ids = store.mark_interrupted()
        for task_id in ids:
            seq = max((e["seq"] for e in store.events(task_id)), default=-1) + 1
            store.append_event(task_id, seq, TERMINAL, {"type": TERMINAL, "status": "interrupted"})
        if ids:
            _log.info("Marked %d unfinished task(s) as interrupted", len(ids))
        return ids

    # ── lifecycle ─────────────────────────────────────────────────────────

    def start(self, chat_id: str | None, model_id: str | None, fn: Callable[[TaskContext], None]) -> str:
        self._sweep()
        task_id = uuid.uuid4().hex
        live = _Live(task_id, chat_id)
        with self._lock:
            self._live[task_id] = live
        store.create(task_id, chat_id, model_id)
        ctx = TaskContext(live, self)
        self._emit(live, "task_start", {"task_id": task_id, "chat_id": chat_id})
        threading.Thread(target=self._run, args=(live, ctx, fn), daemon=True, name=f"task-{task_id[:8]}").start()
        return task_id

    def _run(self, live: _Live, ctx: TaskContext, fn: Callable[[TaskContext], None]) -> None:
        status, error = "done", None
        with usage.meter(ctx.meter):
            try:
                fn(ctx)
                if live.cancel.is_set():
                    status = "cancelled"
            except Cancelled:
                status = "cancelled"
            except Exception as exc:
                _log.exception("Task %s failed", live.task_id)
                status, error = "failed", str(exc)
                self._emit(live, "error", {"message": str(exc)})
        for fn_close in live.on_close:
            try:
                fn_close()
            except Exception as exc:
                _log.warning("Task cleanup failed: %s", exc)
        spent = ctx.meter.snapshot()
        self._emit(live, "usage", spent)
        store.set_status(live.task_id, status, error=error, usage=spent)
        # Mark finished in the same critical section as the terminal event, so a client that
        # reacts to task_end (e.g. by sending the next message) never sees the task as active.
        with live.cond:
            live.finished_at = time.monotonic()
            self._emit(live, TERMINAL, {"status": status})

    def cancel(self, task_id: str) -> bool:
        live = self._live.get(task_id)
        if not live or live.finished_at is not None:
            return False
        live.cancel.set()
        with live.cond:
            live.cond.notify_all()
        return True

    def reply(self, task_id: str, payload: dict) -> bool:
        live = self._live.get(task_id)
        if not live or live.pending is None:
            return False
        if payload.get("request_id") and payload["request_id"] != live.pending.get("id"):
            return False
        live.reply = payload
        live.replied.set()
        return True

    # ── reading ───────────────────────────────────────────────────────────

    def subscribe(self, task_id: str, after: int = -1, heartbeat: float = 15.0) -> Iterator[dict]:
        """Yield events with seq > after; {"type": "ping"} every `heartbeat` seconds of quiet."""
        live = self._live.get(task_id)
        if live is None:
            yield from store.events(task_id, after)
            return
        i = 0
        while True:
            with live.cond:
                while i < len(live.events) and live.events[i]["seq"] <= after:
                    i += 1
                if i >= len(live.events):
                    if live.finished_at is not None:
                        return
                    live.cond.wait(heartbeat)
                    if i >= len(live.events):
                        if live.finished_at is not None:
                            return
                        batch = [{"type": "ping"}]
                    else:
                        batch = []
                else:
                    batch = live.events[i:]
                    i = len(live.events)
            for evt in batch:
                yield evt
                if evt.get("seq") is not None:
                    after = evt["seq"]
                if evt.get("type") == TERMINAL:
                    return

    def is_active(self, task_id: str) -> bool:
        live = self._live.get(task_id)
        return bool(live and live.finished_at is None)

    def active_for_chat(self, chat_id: str) -> dict | None:
        with self._lock:
            lives = [l for l in self._live.values() if l.chat_id == chat_id and l.finished_at is None]
        if not lives:
            return None
        live = lives[-1]
        return {"task_id": live.task_id, "pending": live.pending}

    # ── internals ─────────────────────────────────────────────────────────

    def _emit(self, live: _Live, type_: str, data: dict) -> int:
        with live.cond:
            seq = len(live.events)
            evt = {"seq": seq, "type": type_, **data}
            live.events.append(evt)
            live.cond.notify_all()
        if type_ not in _UNPERSISTED:
            store.append_event(live.task_id, seq, type_, {"type": type_, **data})
        return seq

    def _sweep(self) -> None:
        now = time.monotonic()
        with self._lock:
            for task_id in [t for t, l in self._live.items()
                            if l.finished_at is not None and now - l.finished_at > _KEEP_FINISHED_S]:
                del self._live[task_id]
