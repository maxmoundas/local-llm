"""Local JSON-file persistence for chat sessions."""
from __future__ import annotations

import functools
import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

CHATS_DIR = Path("chats")

# Background tasks and API requests write chats concurrently; every read-modify-write holds this.
_lock = threading.RLock()


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


def _read(chat_id: str) -> dict | None:
    p = _path(chat_id)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _write(chat: dict) -> None:
    CHATS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _path(chat["id"]).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(chat, indent=2), encoding="utf-8")
    tmp.replace(_path(chat["id"]))  # atomic: readers never see a half-written file


def _preview(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m["role"] in ("user", "assistant"):
            text = m["content"][:80]
            return text + ("…" if len(m["content"]) > 80 else "")
    return ""


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def list_chats() -> list[dict]:
    CHATS_DIR.mkdir(parents=True, exist_ok=True)
    result = []
    for p in sorted(CHATS_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            result.append({
                "id": data["id"],
                "name": data["name"],
                "created_at": data["created_at"],
                "updated_at": data["updated_at"],
                "preview": _preview(data.get("messages", [])),
                "model_id": data.get("model_id"),
                "pinned": data.get("pinned", False),
            })
        except Exception:
            pass
    return result


@_locked
def create_chat() -> dict:
    chat = {
        "id": str(uuid.uuid4()),
        "name": "New Chat",
        "created_at": _now(),
        "updated_at": _now(),
        "messages": [],
    }
    _write(chat)
    return chat


@_locked
def get_chat(chat_id: str) -> dict | None:
    return _read(chat_id)


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
    """Append a message. `extra` adds display-only fields (research steps, sources, stopped)."""
    chat = _read(chat_id)
    if chat is None:
        return None
    chat["messages"].append({"role": role, "content": content, **(extra or {})})
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
    p = _path(chat_id)
    if not p.exists():
        return False
    p.unlink()
    return True
