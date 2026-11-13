"""Backup and restore everything the user made: chats (with their files), projects (files, memory),
the memory profile, settings, and scheduled tasks — as one zip.

Not included: API keys (.env), connectors.yaml (may hold tokens), the task/usage database, agent
workspaces, and model weights.

    manifest.json
    chats/<uuid>.json            chats/<uuid>/files/<file>
    projects/<uuid>.json         projects/<uuid>/files/<file>   projects/<uuid>/memory.txt
    memory.txt   settings.json   schedules.json

Restore modes:
  merge    add chats, projects, and schedules that aren't here yet; keep everything that is. Memory
           and settings are taken from the backup only if this install has none.
  replace  the backup wins: matching chats/projects are overwritten, memory, settings and schedules
           replaced. Chats and projects that exist only here are kept.

Archive member names are checked against the layout above before anything is written, so a crafted
zip can't write outside these folders (zip-slip), and sizes are capped (zip bombs).
"""
from __future__ import annotations

import json
import re
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from src import chat_store, memory_store, project_store, scheduler, settings_store

FORMAT_VERSION = 1
MAX_MEMBERS = 200_000
MAX_UNCOMPRESSED = 8 * 1024**3  # 8 GB

_UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
_FILE = r"[0-9A-Za-z_-][0-9A-Za-z_.-]{0,200}"  # no leading dot, no slashes
_TOP_LEVEL = {"manifest.json", "memory.txt", "settings.json", "schedules.json"}
_CHAT_DOC = re.compile(rf"^chats/({_UUID})\.json$")
_CHAT_FILE = re.compile(rf"^chats/({_UUID})/files/{_FILE}$")
_PROJECT_DOC = re.compile(rf"^projects/({_UUID})\.json$")
_PROJECT_FILE = re.compile(rf"^projects/({_UUID})/(?:files/{_FILE}|memory\.txt)$")


class BackupError(ValueError):
    """The upload isn't a usable backup."""


# ── Backup ──────────────────────────────────────────────────────────────

def _add_tree(zf: zipfile.ZipFile, base: Path, arc_prefix: str, pattern: re.Pattern) -> int:
    n = 0
    if not base.is_dir():
        return n
    for p in sorted(base.rglob("*")):
        if not p.is_file():
            continue
        arc = f"{arc_prefix}/{p.relative_to(base).as_posix()}"
        if pattern.match(arc):
            zf.write(p, arc)
            n += 1
    return n


def create_backup() -> Path:
    """Write a backup zip to a temporary file and return its path (the caller deletes it)."""
    out = Path(tempfile.mkstemp(prefix="local-llm-backup-", suffix=".zip")[1])
    any_chat = re.compile(rf"^chats/(?:{_UUID}\.json|{_UUID}/files/{_FILE})$")
    any_project = re.compile(rf"^projects/(?:{_UUID}\.json|{_UUID}/(?:files/{_FILE}|memory\.txt))$")
    with chat_store._lock, project_store._lock, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        chats = _add_tree(zf, chat_store.CHATS_DIR, "chats", any_chat)
        projects = _add_tree(zf, project_store.PROJECTS_DIR, "projects", any_project)
        for arc, path in (("memory.txt", memory_store.MEMORY_PATH), ("settings.json", settings_store.SETTINGS_PATH),
                          ("schedules.json", scheduler.SCHEDULES_PATH)):
            if Path(path).is_file():
                zf.write(path, arc)
        zf.writestr("manifest.json", json.dumps({
            "format": FORMAT_VERSION, "app": "local-llm",
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "files": {"chats": chats, "projects": projects},
        }, indent=2))
    return out


# ── Restore ─────────────────────────────────────────────────────────────

def _check(zf: zipfile.ZipFile) -> dict:
    infos = zf.infolist()
    if len(infos) > MAX_MEMBERS:
        raise BackupError("The backup has too many files")
    if sum(i.file_size for i in infos) > MAX_UNCOMPRESSED:
        raise BackupError("The backup is too large once unpacked")
    try:
        manifest = json.loads(zf.read("manifest.json"))
    except KeyError:
        raise BackupError("Not a Local LLM backup (no manifest.json)")
    except ValueError:
        raise BackupError("The backup's manifest.json is unreadable")
    if manifest.get("app") != "local-llm" or int(manifest.get("format", 0)) > FORMAT_VERSION:
        raise BackupError("This backup was made by a different app or a newer version")
    for i in infos:
        name = i.filename
        if i.is_dir():
            continue
        if not (name in _TOP_LEVEL or _CHAT_DOC.match(name) or _CHAT_FILE.match(name)
                or _PROJECT_DOC.match(name) or _PROJECT_FILE.match(name)):
            raise BackupError(f"Unexpected file in backup: {name[:120]}")
    return manifest


def _json(zf: zipfile.ZipFile, name: str) -> object:
    try:
        return json.loads(zf.read(name))
    except ValueError:
        raise BackupError(f"{name} in the backup is not valid JSON")


def _write_group(zf: zipfile.ZipFile, base: Path, item_id: str, doc_name: str, members: list[str],
                 replace: bool) -> None:
    """Write one chat/project: its document plus its folder (files, memory)."""
    folder = base / item_id
    if replace and folder.exists():
        shutil.rmtree(folder)
    for name in members:
        rel = name.split("/", 1)[1]  # strip "chats/" or "projects/"
        target = base / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(name) as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst)
    base.mkdir(parents=True, exist_ok=True)
    tmp = base / f"{item_id}.json.tmp"
    with zf.open(doc_name) as src, open(tmp, "wb") as dst:
        shutil.copyfileobj(src, dst)
    tmp.replace(base / f"{item_id}.json")


def _restore_group(zf: zipfile.ZipFile, base: Path, doc_re: re.Pattern, member_re: re.Pattern,
                   replace: bool) -> tuple[int, int]:
    """Restore every chat (or project) in the archive. Returns (restored, skipped)."""
    names = [i.filename for i in zf.infolist() if not i.is_dir()]
    members: dict[str, list[str]] = {}
    for n in names:
        if m := member_re.match(n):
            members.setdefault(m.group(1), []).append(n)
    restored = skipped = 0
    for n in names:
        m = doc_re.match(n)
        if not m:
            continue
        item_id = m.group(1)
        doc = _json(zf, n)
        if not isinstance(doc, dict) or doc.get("id") != item_id:
            raise BackupError(f"{n} doesn't match its file name")
        if (base / f"{item_id}.json").exists() and not replace:
            skipped += 1
            continue
        _write_group(zf, base, item_id, n, members.get(item_id, []), replace)
        restored += 1
    return restored, skipped


def restore(archive: Path, mode: str = "merge") -> dict:
    """Restore a backup zip. Returns counts of what was restored and skipped."""
    if mode not in ("merge", "replace"):
        raise BackupError(f"Unknown restore mode: {mode}")
    replace = mode == "replace"
    try:
        zf = zipfile.ZipFile(archive)
    except zipfile.BadZipFile:
        raise BackupError("The file is not a zip archive")
    with zf:
        _check(zf)
        names = set(zf.namelist())
        result: dict = {"mode": mode}
        with chat_store._lock:
            result["chats_restored"], result["chats_skipped"] = _restore_group(
                zf, chat_store.CHATS_DIR, _CHAT_DOC, _CHAT_FILE, replace)
        with project_store._lock:
            result["projects_restored"], result["projects_skipped"] = _restore_group(
                zf, project_store.PROJECTS_DIR, _PROJECT_DOC, _PROJECT_FILE, replace)

        if "memory.txt" in names and (replace or not memory_store.load().strip()):
            memory_store.save(zf.read("memory.txt").decode("utf-8", "replace"))
            result["memory_restored"] = True

        if "settings.json" in names and (replace or not Path(settings_store.SETTINGS_PATH).exists()):
            data = _json(zf, "settings.json")
            if not isinstance(data, dict):
                raise BackupError("settings.json in the backup is not an object")
            known = settings_store.load().keys()
            settings_store.save({k: v for k, v in data.items() if k in known})
            result["settings_restored"] = True

        if "schedules.json" in names:
            incoming = _json(zf, "schedules.json")
            if not isinstance(incoming, list):
                raise BackupError("schedules.json in the backup is not a list")
            with scheduler._lock:
                current = scheduler._load()
                if replace:
                    merged = [s for s in incoming if isinstance(s, dict)]
                else:
                    have = {s.get("id") for s in current}
                    merged = current + [s for s in incoming if isinstance(s, dict) and s.get("id") not in have]
                result["schedules_restored"] = len(merged) - (0 if replace else len(current))
                scheduler._save(merged)

    chat_store.sync_search_index()
    return result
