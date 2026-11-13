"""Projects: a named group of chats that share instructions, knowledge files, and memory.

    projects/<id>.json         {id, name, description, instructions, files: [meta], created_at, updated_at}
    projects/<id>/files/       knowledge files (see attachments.py)
    projects/<id>/memory.txt   facts learned in this project's chats (memory_store format)
"""
from __future__ import annotations

import json
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src import attachments

PROJECTS_DIR = Path("projects")
_lock = threading.RLock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _path(project_id: str) -> Path:
    return PROJECTS_DIR / f"{project_id}.json"


def files_dir(project_id: str) -> Path:
    return PROJECTS_DIR / project_id / "files"


def memory_path(project_id: str) -> Path:
    return PROJECTS_DIR / project_id / "memory.txt"


def _read(project_id: str) -> dict | None:
    try:
        uuid.UUID(project_id)
    except (ValueError, TypeError):
        return None
    p = _path(project_id)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _write(project: dict) -> None:
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _path(project["id"]).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(project, indent=2), encoding="utf-8")
    tmp.replace(_path(project["id"]))


def list_projects() -> list[dict]:
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for p in PROJECTS_DIR.glob("*.json"):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass
    return sorted(out, key=lambda x: x.get("updated_at", ""), reverse=True)


def get(project_id: str) -> dict | None:
    with _lock:
        return _read(project_id)


def create(name: str, description: str = "", instructions: str = "") -> dict:
    project = {
        "id": str(uuid.uuid4()), "name": name.strip() or "Untitled project", "description": description.strip(),
        "instructions": instructions.strip(), "files": [], "created_at": _now(), "updated_at": _now(),
    }
    with _lock:
        _write(project)
    return project


def update(project_id: str, **fields) -> dict | None:
    allowed = {"name", "description", "instructions"}
    with _lock:
        project = _read(project_id)
        if project is None:
            return None
        for k, v in fields.items():
            if k in allowed and v is not None:
                project[k] = v.strip() if isinstance(v, str) else v
        project["updated_at"] = _now()
        _write(project)
        return project


def add_file(project_id: str, name: str, b64: str, mime: str | None = None) -> dict | None:
    with _lock:
        project = _read(project_id)
        if project is None:
            return None
        meta = attachments.save_base64(files_dir(project_id), name, b64, mime)
        project["files"].append(meta)
        project["updated_at"] = _now()
        _write(project)
        return meta


def remove_file(project_id: str, file_id: str) -> bool:
    with _lock:
        project = _read(project_id)
        if project is None or not any(f["id"] == file_id for f in project["files"]):
            return False
        attachments.delete(files_dir(project_id), file_id)
        project["files"] = [f for f in project["files"] if f["id"] != file_id]
        project["updated_at"] = _now()
        _write(project)
        return True


def delete(project_id: str) -> bool:
    with _lock:
        if _read(project_id) is None:
            return False
        _path(project_id).unlink()
        shutil.rmtree(PROJECTS_DIR / project_id, ignore_errors=True)
        return True


def knowledge(project: dict) -> list[tuple[dict, str]]:
    """(file meta, extracted text) for each document in the project."""
    root = files_dir(project["id"])
    return [(f, attachments.text_of(root, f)) for f in project.get("files", []) if f.get("kind") == "document"]
