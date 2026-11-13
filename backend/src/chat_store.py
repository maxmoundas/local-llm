"""Local JSON-file persistence for chat sessions.

    chats/<id>.json          the chat (see README for the schema)
    chats/<id>/files/        files attached to its messages (see attachments.py)

Versions (regenerate / edit): `chat["messages"]` is always the path currently shown. A message
that has alternatives carries them as whole conversation tails:

    {"role": "assistant", "content": "...", "versions": [[msg, msg, ...], ...], "version_index": 1}

`versions` holds the *other* tails starting at this position, in order; `version_index` is where
the shown one sits among all of them. Tails nest, so this is a tree stored as its visible path.
"""
from __future__ import annotations

import functools
import json
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src import search_index

CHATS_DIR = Path("chats")

# Background tasks and API requests write chats concurrently; every read-modify-write holds this.
_lock = threading.RLock()

_VERSION_KEYS = ("versions", "version_index")


def _locked(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with _lock:
            return fn(*args, **kwargs)
    return wrapper


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _path(chat_id: str) -> Path:
    return CHATS_DIR / f"{chat_id}.json"


class InvalidChatId(ValueError):
    """A chat id that isn't a UUID (never a real chat; refuse before touching the filesystem)."""


def _valid_id(chat_id: str) -> bool:
    try:
        uuid.UUID(chat_id)
        return True
    except (ValueError, TypeError, AttributeError):
        return False


def files_dir(chat_id: str) -> Path:
    if not _valid_id(chat_id):
        raise InvalidChatId(f"Invalid chat id: {chat_id!r}")
    return CHATS_DIR / chat_id / "files"


def _read(chat_id: str) -> dict | None:
    if not _valid_id(chat_id):
        return None
    p = _path(chat_id)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _write(chat: dict) -> None:
    CHATS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _path(chat["id"]).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(chat, indent=2), encoding="utf-8")
    tmp.replace(_path(chat["id"]))  # atomic: readers never see a half-written file
    search_index.index_chat(CHATS_DIR, chat, _path(chat["id"]).stat().st_mtime)


def _preview(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m["role"] in ("user", "assistant"):
            text = m["content"][:80]
            return text + ("…" if len(m["content"]) > 80 else "")
    return ""


def _summary(data: dict) -> dict:
    return {
        "id": data["id"],
        "name": data["name"],
        "created_at": data["created_at"],
        "updated_at": data["updated_at"],
        "preview": _preview(data.get("messages", [])),
        "model_id": data.get("model_id"),
        "pinned": data.get("pinned", False),
        "project_id": data.get("project_id"),
        "schedule_id": data.get("schedule_id"),
    }


def _all() -> list[dict]:
    CHATS_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for p in sorted(CHATS_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass
    return out


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def list_chats(project_id: str | None = None) -> list[dict]:
    chats = [_summary(c) for c in _all()]
    return [c for c in chats if c["project_id"] == project_id] if project_id else chats


@_locked
def create_chat(project_id: str | None = None, name: str = "New Chat", schedule_id: str | None = None) -> dict:
    chat = {
        "id": str(uuid.uuid4()),
        "name": name,
        "created_at": _now(),
        "updated_at": _now(),
        "messages": [],
    }
    if project_id:
        chat["project_id"] = project_id
    if schedule_id:
        chat["schedule_id"] = schedule_id
    _write(chat)
    return chat


@_locked
def get_chat(chat_id: str) -> dict | None:
    return _read(chat_id)


@_locked
def update_chat(chat_id: str, **fields) -> dict | None:
    """Set top-level fields (settings, project_id, context_summary…). None removes a field."""
    chat = _read(chat_id)
    if chat is None:
        return None
    for k, v in fields.items():
        if v is None:
            chat.pop(k, None)
        else:
            chat[k] = v
    _write(chat)
    return chat


@_locked
def rename_chat(chat_id: str, name: str) -> dict | None:
    chat = _read(chat_id)
    if chat is None:
        return None
    chat["name"] = name.strip() or "New Chat"
    chat["updated_at"] = _now()
    _write(chat)
    return chat


@_locked
def append_messages(
    chat_id: str, role: str, content: str, model_id: str | None = None, extra: dict | None = None
) -> dict | None:
    """Append a message. `extra` adds display-only fields (research steps, sources, attachments…).

    If a fork is pending at this position (regenerate / edit), the new message carries the
    earlier versions so the user can switch between them.
    """
    chat = _read(chat_id)
    if chat is None:
        return None
    msg = {"role": role, "content": content, **(extra or {})}
    fork = chat.get("pending_fork")
    if fork and fork["index"] == len(chat["messages"]):
        msg["versions"] = fork["versions"]
        msg["version_index"] = len(fork["versions"])
        chat.pop("pending_fork")
    chat["messages"].append(msg)
    chat["updated_at"] = _now()
    if role == "user" and model_id:
        chat["model_id"] = model_id
    _write(chat)
    return chat


@_locked
def toggle_pin(chat_id: str) -> dict | None:
    chat = _read(chat_id)
    if chat is None:
        return None
    chat["pinned"] = not chat.get("pinned", False)
    chat["updated_at"] = _now()
    _write(chat)
    return chat


@_locked
def delete_chat(chat_id: str) -> bool:
    if _read(chat_id) is None:
        return False
    _path(chat_id).unlink()
    shutil.rmtree(CHATS_DIR / chat_id, ignore_errors=True)
    search_index.remove_chat(CHATS_DIR, chat_id)
    return True


# ── Versions (regenerate / edit) ────────────────────────────────────────

def _strip(msg: dict) -> dict:
    return {k: v for k, v in msg.items() if k not in _VERSION_KEYS}


def _ordered_versions(messages: list[dict], index: int) -> tuple[list[list[dict]], int]:
    """Every version of the tail starting at `index`, in order, and which one is shown."""
    head = messages[index]
    others = head.get("versions", [])
    shown = min(head.get("version_index", len(others)), len(others))
    current = [_strip(head), *messages[index + 1:]]
    return [*others[:shown], current, *others[shown:]], shown


def _show(prefix: list[dict], ordered: list[list[dict]], k: int) -> list[dict]:
    tail = ordered[k]
    others = ordered[:k] + ordered[k + 1:]
    head = {**_strip(tail[0]), "versions": others, "version_index": k}
    return [*prefix, head, *tail[1:]]


@_locked
def fork(chat_id: str, index: int) -> dict | None:
    """Cut the shown path at `index` (keeping it as a version) so a new reply or edit can go there.

    The next message appended at `index` becomes a new version; `restore_fork` undoes the cut
    if the turn produces nothing.
    """
    chat = _read(chat_id)
    if chat is None or not 0 <= index < len(chat["messages"]):
        return None
    ordered, shown = _ordered_versions(chat["messages"], index)
    chat["messages"] = chat["messages"][:index]
    chat["pending_fork"] = {"index": index, "versions": ordered, "shown": shown}
    _write(chat)
    return chat


@_locked
def restore_fork(chat_id: str) -> bool:
    """Put back the version that was shown before an unfinished fork."""
    chat = _read(chat_id)
    fork_ = chat.get("pending_fork") if chat else None
    if not fork_:
        return False
    i, ordered = fork_["index"], fork_["versions"]
    k = min(fork_.get("shown", len(ordered) - 1), len(ordered) - 1)
    chat["messages"] = _show(chat["messages"][:i], ordered, k)
    if len(ordered) == 1:
        chat["messages"][i] = _strip(chat["messages"][i])
    chat.pop("pending_fork")
    _write(chat)
    return True


@_locked
def switch_version(chat_id: str, index: int, version: int) -> dict | None:
    chat = _read(chat_id)
    if chat is None or not 0 <= index < len(chat["messages"]) or chat.get("pending_fork"):
        return None
    ordered, _ = _ordered_versions(chat["messages"], index)
    if not 0 <= version < len(ordered):
        return None
    chat["messages"] = _show(chat["messages"][:index], ordered, version)
    chat["updated_at"] = _now()
    _write(chat)
    return chat


# ── Search & export ─────────────────────────────────────────────────────

def _snippet(text: str, pos: int, qlen: int, width: int = 70) -> str:
    start = max(0, pos - width)
    end = min(len(text), pos + qlen + width)
    s = " ".join(text[start:end].split())
    return ("…" if start > 0 else "") + s + ("…" if end < len(text) else "")


def search(query: str, limit: int = 50) -> list[dict]:
    """Chats whose title or any message contains `query` (case-insensitive), newest first.

    One result per chat: the first matching message, with a snippet around the match.
    """
    q = query.strip().lower()
    if not q:
        return []
    indexed = search_index.search(CHATS_DIR, query, limit)
    if indexed is not None:
        return _indexed_results(indexed, q)
    results = []
    for chat in _all():
        hit = None
        if q in chat["name"].lower():
            hit = {"index": None, "role": None, "snippet": _preview(chat.get("messages", []))}
        for i, m in enumerate(chat.get("messages", [])):
            text = m.get("content") or ""
            pos = text.lower().find(q)
            if pos != -1:
                hit = {"index": i, "role": m["role"], "snippet": _snippet(text, pos, len(q))}
                break
        if hit:
            results.append({**_summary(chat), **hit})
            if len(results) >= limit:
                break
    return results


def _indexed_results(hits: list[tuple[str, int, str | None, str]], q: str) -> list[dict]:
    results = []
    for chat_id, idx, role, text in hits:
        chat = _read(chat_id)
        if chat is None:
            continue
        if idx < 0:
            hit = {"index": None, "role": None, "snippet": _preview(chat.get("messages", []))}
        else:
            hit = {"index": idx, "role": role, "snippet": _snippet(text, max(text.lower().find(q), 0), len(q))}
        results.append({**_summary(chat), **hit})
    return results


def sync_search_index() -> int:
    """Bring the search index up to date with the chat files (run at startup)."""
    return search_index.sync(CHATS_DIR)


def export_markdown(chat: dict) -> str:
    lines = [f"# {chat['name']}", "", f"_Exported {datetime.now():%Y-%m-%d %H:%M} · created {chat['created_at'][:10]}_", ""]
    for m in chat.get("messages", []):
        if m["role"] == "system":
            continue
        who = "You" if m["role"] == "user" else "Assistant"
        lines += [f"## {who}", ""]
        for f in m.get("attachments", []):
            lines.append(f"> 📎 {f['name']}")
        if m.get("attachments"):
            lines.append("")
        lines += [m.get("content") or "_(no text)_", ""]
        if m.get("sources"):
            lines += ["**Sources**", ""] + [f"- [{s['title']}]({s['url']})" for s in m["sources"]] + [""]
    return "\n".join(lines).rstrip() + "\n"


def export_json(chat: dict) -> dict:
    return {k: v for k, v in chat.items() if k not in ("pending_fork", "context_summary")}
