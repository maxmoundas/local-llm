"""Full-text search index over chats (SQLite FTS5, trigram tokenizer).

Lives next to the chats (chats/search.db). The trigram tokenizer matches any substring of 3+
characters case-insensitively — the same semantics as the old linear scan, without reading every
chat file per query. Shorter queries, or a SQLite without FTS5/trigram, fall back to the scan in
chat_store.search.

Kept in sync by chat_store: every write re-indexes that chat, every delete removes it, and
`sync()` reconciles against file mtimes, so edits made while the server was down are picked up.

The `chats` table also keeps each chat's sidebar summary (JSON from `summarize`), so listing chats
reads one table instead of parsing every chat file.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Callable
from pathlib import Path

from src.logger import get_logger

_log = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chats (
    chat_id    TEXT PRIMARY KEY,
    updated_at TEXT NOT NULL,
    mtime      REAL NOT NULL,
    summary    TEXT
);
CREATE VIRTUAL TABLE IF NOT EXISTS entries USING fts5(
    chat_id UNINDEXED, idx UNINDEXED, role UNINDEXED, content, tokenize='trigram'
);
"""
MIN_QUERY = 3  # trigram needs at least 3 characters

_lock = threading.RLock()
_conn: sqlite3.Connection | None = None
_conn_path: Path | None = None
_broken = False

# Set by chat_store: chat dict -> the summary dict list_chats returns. None stores no summary.
summarize: Callable[[dict], dict] | None = None


def _db(chats_dir: Path) -> sqlite3.Connection | None:
    global _conn, _conn_path, _broken
    path = chats_dir / "search.db"
    if _conn is not None and _conn_path == path:
        return _conn
    try:
        chats_dir.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_SCHEMA)
        if "summary" not in {r["name"] for r in conn.execute("PRAGMA table_info(chats)")}:
            conn.execute("ALTER TABLE chats ADD COLUMN summary TEXT")  # index from before chat summaries
    except sqlite3.Error as exc:
        if not _broken:
            _log.warning("Search index unavailable (%s); search will scan chat files", exc)
        _broken = True
        return None
    if _conn is not None:
        _conn.close()
    _conn, _conn_path, _broken = conn, path, False
    return conn


def available(chats_dir: Path) -> bool:
    with _lock:
        return _db(chats_dir) is not None


def _rows(chat: dict) -> list[tuple]:
    rows = [(chat["id"], -1, None, chat.get("name") or "")]
    for i, m in enumerate(chat.get("messages", [])):
        text = m.get("content")
        if isinstance(text, str) and text and m.get("role") in ("user", "assistant"):
            rows.append((chat["id"], i, m["role"], text))
    return rows


def _index(conn: sqlite3.Connection, chat: dict, mtime: float) -> None:
    conn.execute("DELETE FROM entries WHERE chat_id=?", (chat["id"],))
    conn.executemany("INSERT INTO entries (chat_id, idx, role, content) VALUES (?,?,?,?)", _rows(chat))
    try:
        summary = json.dumps(summarize(chat)) if summarize else None
    except Exception as exc:  # a malformed chat file stays searchable but out of the list
        _log.warning("Could not summarize chat %s: %s", chat.get("id"), exc)
        summary = None
    conn.execute("INSERT OR REPLACE INTO chats (chat_id, updated_at, mtime, summary) VALUES (?,?,?,?)",
                 (chat["id"], chat.get("updated_at") or "", mtime, summary))


def index_chat(chats_dir: Path, chat: dict, mtime: float) -> None:
    with _lock:
        conn = _db(chats_dir)
        if conn is None:
            return
        try:
            _index(conn, chat, mtime)
            conn.commit()
        except sqlite3.Error as exc:
            conn.rollback()
            _log.warning("Could not index chat %s: %s", chat.get("id"), exc)


def remove_chat(chats_dir: Path, chat_id: str) -> None:
    with _lock:
        conn = _db(chats_dir)
        if conn is None:
            return
        conn.execute("DELETE FROM entries WHERE chat_id=?", (chat_id,))
        conn.execute("DELETE FROM chats WHERE chat_id=?", (chat_id,))
        conn.commit()


def sync(chats_dir: Path) -> int:
    """Re-index chats whose file changed since it was indexed; drop chats whose file is gone."""
    with _lock:
        conn = _db(chats_dir)
        if conn is None:
            return 0
        # A row without a summary (indexed before summaries existed) counts as stale.
        known = {r["chat_id"]: (r["mtime"] if r["summary"] is not None or summarize is None else None)
                 for r in conn.execute("SELECT chat_id, mtime, summary FROM chats")}
        on_disk: set[str] = set()
        changed = 0
        for p in chats_dir.glob("*.json"):
            chat_id = p.stem
            on_disk.add(chat_id)
            mtime = p.stat().st_mtime
            if known.get(chat_id) == mtime:
                continue
            try:
                chat = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            _index(conn, chat, mtime)
            changed += 1
        for chat_id in set(known) - on_disk:
            conn.execute("DELETE FROM entries WHERE chat_id=?", (chat_id,))
            conn.execute("DELETE FROM chats WHERE chat_id=?", (chat_id,))
            changed += 1
        conn.commit()
    if changed:
        _log.info("Search index: updated %d chat(s)", changed)
    return changed


def summaries(chats_dir: Path) -> list[dict] | None:
    """Every chat's summary, most recently written first (the order of the files' mtimes).

    Syncs first — that only stats the files and re-reads the ones that changed — so chats written
    outside chat_store (a restore, a hand edit) still show. None when the index is unavailable.
    """
    if summarize is None:
        return None
    sync(chats_dir)
    with _lock:
        conn = _db(chats_dir)
        if conn is None:
            return None
        try:
            rows = conn.execute("SELECT summary FROM chats WHERE summary IS NOT NULL ORDER BY mtime DESC").fetchall()
        except sqlite3.Error as exc:
            _log.warning("Chat list query failed (%s); reading chat files instead", exc)
            return None
    return [json.loads(r["summary"]) for r in rows]


def search(chats_dir: Path, query: str, limit: int = 50) -> list[tuple[str, int, str | None, str]] | None:
    """Matching (chat_id, index, role, content), one per chat — the earliest match (title counts as
    index -1) — newest chat first. None when the index can't answer (short query, no FTS5)."""
    q = query.strip()
    if len(q) < MIN_QUERY:
        return None
    phrase = '"' + q.replace('"', '""') + '"'
    with _lock:
        conn = _db(chats_dir)
        if conn is None:
            return None
        try:
            rows = conn.execute(
                "SELECT e.chat_id, CAST(e.idx AS INTEGER) AS idx, e.role, e.content FROM entries e "
                "JOIN chats c ON c.chat_id = e.chat_id WHERE entries MATCH ? "
                "ORDER BY c.updated_at DESC, CAST(e.idx AS INTEGER) ASC", (phrase,)).fetchall()
        except sqlite3.Error as exc:
            _log.warning("Search index query failed (%s); scanning instead", exc)
            return None
    # Per chat, the first matching message wins; a title match is used only when no message matches.
    best: dict[str, tuple[str, int, str | None, str]] = {}
    for r in rows:
        cur = best.get(r["chat_id"])
        if cur is None or (cur[1] == -1 and r["idx"] >= 0):
            best[r["chat_id"]] = (r["chat_id"], r["idx"], r["role"], r["content"])
    return list(best.values())[:limit]  # dicts keep insertion order: newest chat first
