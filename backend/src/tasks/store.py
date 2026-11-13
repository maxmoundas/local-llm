"""SQLite persistence for background tasks and their event logs.

tasks  — one row per task (a chat turn running in the background)
events — append-only log per task, ordered by seq. Token deltas are not stored (the final
         answer goes to the chat store); everything else is, so a task's activity can be
         rebuilt after a reload or restart.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

DB_PATH = Path("data/tasks.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id         TEXT PRIMARY KEY,
    chat_id    TEXT,
    status     TEXT NOT NULL,          -- running | waiting | done | failed | cancelled | interrupted
    model_id   TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    usage      TEXT,
    error      TEXT
);
CREATE INDEX IF NOT EXISTS tasks_chat ON tasks(chat_id, created_at);
CREATE TABLE IF NOT EXISTS events (
    task_id TEXT NOT NULL,
    seq     INTEGER NOT NULL,
    type    TEXT NOT NULL,
    data    TEXT NOT NULL,
    ts      REAL NOT NULL,
    PRIMARY KEY (task_id, seq)
);
"""

ACTIVE = ("running", "waiting")

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None
_conn_path: Path | None = None


def _db() -> sqlite3.Connection:
    global _conn, _conn_path
    if _conn is None or _conn_path != DB_PATH:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.executescript(_SCHEMA)
        _conn_path = DB_PATH
    return _conn


def create(task_id: str, chat_id: str | None, model_id: str | None) -> None:
    now = time.time()
    with _lock:
        _db().execute("INSERT INTO tasks (id, chat_id, status, model_id, created_at, updated_at) VALUES (?,?,?,?,?,?)",
                      (task_id, chat_id, "running", model_id, now, now))
        _db().commit()


def set_status(task_id: str, status: str, error: str | None = None, usage: dict | None = None) -> None:
    with _lock:
        _db().execute(
            "UPDATE tasks SET status=?, updated_at=?, error=COALESCE(?, error), usage=COALESCE(?, usage) WHERE id=?",
            (status, time.time(), error, json.dumps(usage) if usage is not None else None, task_id))
        _db().commit()


def append_event(task_id: str, seq: int, type_: str, data: dict) -> None:
    with _lock:
        _db().execute("INSERT OR REPLACE INTO events (task_id, seq, type, data, ts) VALUES (?,?,?,?,?)",
                      (task_id, seq, type_, json.dumps(data), time.time()))
        _db().commit()


def get(task_id: str) -> dict | None:
    with _lock:
        row = _db().execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
    return _row(row) if row else None


def events(task_id: str, after: int = -1) -> list[dict]:
    with _lock:
        rows = _db().execute("SELECT seq, type, data FROM events WHERE task_id=? AND seq>? ORDER BY seq",
                             (task_id, after)).fetchall()
    return [{"seq": r["seq"], **json.loads(r["data"])} for r in rows]


def for_chat(chat_id: str, limit: int = 50) -> list[dict]:
    with _lock:
        rows = _db().execute("SELECT * FROM tasks WHERE chat_id=? ORDER BY created_at DESC LIMIT ?",
                             (chat_id, limit)).fetchall()
    return [_row(r) for r in rows]


def mark_interrupted() -> list[str]:
    """On startup: tasks left running by a previous process can't continue in this one."""
    with _lock:
        rows = _db().execute(f"SELECT id FROM tasks WHERE status IN ({','.join('?' * len(ACTIVE))})", ACTIVE).fetchall()
        ids = [r["id"] for r in rows]
        if ids:
            _db().execute(f"UPDATE tasks SET status='interrupted', updated_at=? WHERE id IN ({','.join('?' * len(ids))})",
                          (time.time(), *ids))
            _db().commit()
    return ids


def usage_since(since: float) -> list[dict]:
    """Finished tasks created after `since` (epoch seconds) that recorded usage — for the usage dashboard."""
    with _lock:
        rows = _db().execute("SELECT id, chat_id, model_id, created_at, usage FROM tasks "
                             "WHERE created_at>=? AND usage IS NOT NULL ORDER BY created_at", (since,)).fetchall()
    return [_row(r) for r in rows]


def delete_for_chat(chat_id: str) -> None:
    """Forget a deleted chat's activity. Task rows stay (detached) so past spending still counts."""
    with _lock:
        _db().execute("DELETE FROM events WHERE task_id IN (SELECT id FROM tasks WHERE chat_id=?)", (chat_id,))
        _db().execute("UPDATE tasks SET chat_id=NULL WHERE chat_id=?", (chat_id,))
        _db().commit()


def _row(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["usage"] = json.loads(d["usage"]) if d.get("usage") else None
    return d
