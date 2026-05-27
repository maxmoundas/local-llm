"""Persistent plain-text store for the user memory profile."""
from __future__ import annotations

from pathlib import Path

MEMORY_PATH = Path("memory.txt")


def load() -> str:
    """Return current memory content, or empty string if none saved."""
    if not MEMORY_PATH.exists():
        return ""
    try:
        return MEMORY_PATH.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def save(content: str) -> None:
    """Write memory content to disk (strips surrounding whitespace)."""
    MEMORY_PATH.write_text(content.strip(), encoding="utf-8")
