"""Persistent plain-text store for the user memory profile."""
from __future__ import annotations

import os
import threading
from collections.abc import Callable
from pathlib import Path

MEMORY_PATH = Path("memory.txt")

# Serializes read-modify-write cycles: background extraction runs one thread per message,
# so two quick messages would otherwise each save their own edit and drop the other's.
_lock = threading.RLock()


def load(path: Path | None = None) -> str:
    """Return current memory content, or empty string if none saved.

    `path` selects another memory file (a project's); the default is the global profile."""
    path = path or MEMORY_PATH
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""
    # Older versions could save the extractor's empty-profile placeholder verbatim.
    return "" if text.lower() in ("(none)", "(empty)", "none") else text


def save(content: str, path: Path | None = None) -> None:
    """Write memory content to disk atomically (strips surrounding whitespace)."""
    path = path or MEMORY_PATH
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(content.strip(), encoding="utf-8")
        os.replace(tmp, path)


def update(fn: Callable[[str], str | None], path: Path | None = None) -> dict | None:
    """Apply fn to the current memory under the lock; fn returns new content or None for no change.

    Returns {"added": [...], "removed": [...]} (changed facts, bullet markers stripped) if memory
    was written, else None.
    """
    with _lock:
        old = load(path)
        updated = fn(old)
        if updated is None:
            return None
        save(updated, path)
        return diff(old, updated.strip())


def diff(old: str, new: str) -> dict:
    """Facts added and removed between two memory texts."""
    def facts(text: str) -> list[str]:
        return [ln.strip().lstrip("-*• ").strip() for ln in text.splitlines() if ln.strip()]
    before, after = facts(old), facts(new)
    return {"added": [f for f in after if f not in before], "removed": [f for f in before if f not in after]}
