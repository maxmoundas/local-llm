"""Tools bound to one task's workspace: run code, read/write/list files.

Built per task by `workspace_tools(sandbox, task_id)` so every call lands in that task's
directory (and, with Docker, that task's container).
"""
from __future__ import annotations

from pathlib import Path

from src.agent import sandbox as sb
from src.agent.sandbox import LANGUAGES, Sandbox, safe_path, workspace_for
from src.agent.tools import Tool, ToolResult

_READ_LIMIT = 12000
_SKIP = {".local", ".cache", "__pycache__", "node_modules", ".npm"}


def list_workspace(ws: Path, limit: int = 200) -> list[dict]:
    files: list[dict] = []
    if not ws.exists():
        return files
    for p in sorted(ws.rglob("*")):
        rel = p.relative_to(ws)
        if p.is_file() and not any(part in _SKIP for part in rel.parts) and rel.name not in {"main.py", "main.sh", "main.js"}:
            files.append({"path": str(rel), "size": p.stat().st_size})
            if len(files) >= limit:
                break
    return files


def _first_line(code: str) -> str:
    for line in code.splitlines():
        line = line.strip().lstrip("#/").strip()
        if line:
            return line[:70]
    return ""


def workspace_tools(sandbox: Sandbox, task_id: str, timeout: int = 120) -> list[Tool]:
    ws = (sb.WORKSPACES_DIR / task_id).resolve()  # created on first write
    languages = [lang for lang in sandbox.languages if lang in LANGUAGES] or ["python"]

    def run_code(a: dict) -> ToolResult:
        language = a.get("language", "python")
        if language not in languages:
            return ToolResult(text=f"Unsupported language {language!r}. Use one of: {', '.join(languages)}",
                              summary="unsupported", ok=False)
        before = {f["path"]: f["size"] for f in list_workspace(ws)}
        result = sandbox.run(task_id, language, str(a.get("code", "")), timeout=min(int(a.get("timeout") or timeout), 600))
        after = list_workspace(ws)
        changed = [f["path"] for f in after if before.get(f["path"]) != f["size"]]
        text = result.text() + (f"\nfiles written: {', '.join(changed)}" if changed else "")
        summary = "timed out" if result.timed_out else ("ok" if result.exit_code == 0 else f"exit {result.exit_code}")
        return ToolResult(text=text, summary=summary, ok=result.exit_code == 0 and not result.timed_out)

    def write_file(a: dict) -> ToolResult:
        workspace_for(task_id)
        target = safe_path(ws, a.get("path", ""))
        target.parent.mkdir(parents=True, exist_ok=True)
        content = str(a.get("content", ""))
        target.write_text(content, encoding="utf-8")
        return ToolResult(text=f"Wrote {len(content)} characters to {target.relative_to(ws)}",
                          summary=f"{len(content):,} chars")

    def read_file(a: dict) -> ToolResult:
        target = safe_path(ws, a.get("path", ""))
        if not target.is_file():
            return ToolResult(text=f"No such file: {a.get('path')}", summary="not found", ok=False)
        text = target.read_text(encoding="utf-8", errors="replace")
        clipped = text if len(text) <= _READ_LIMIT else text[:_READ_LIMIT] + f"\n… [{len(text) - _READ_LIMIT} more characters]"
        return ToolResult(text=clipped, summary=f"{len(text):,} chars")

    def list_files(a: dict) -> ToolResult:
        files = list_workspace(ws)
        if not files:
            return ToolResult(text="The workspace is empty.", summary="empty")
        return ToolResult(text="\n".join(f"{f['path']} ({f['size']} bytes)" for f in files),
                          summary=f"{len(files)} file{'s' if len(files) != 1 else ''}")

    egress = bool(getattr(sandbox, "network", False))
    net = "with" if egress else "without"
    return [
        Tool(
            name="run_code",
            description=(
                f"Run code in an isolated sandbox ({net} internet access) whose working directory is the task "
                "workspace. Use for calculations, data analysis, file processing, and checking that code works. "
                "Print results to stdout. Files written to the current directory persist for later steps "
                "and are shared with the user. `pip install --user <pkg>` works in python via subprocess "
                "or in bash when internet is available."
            ),
            parameters={"type": "object", "properties": {
                "language": {"type": "string", "enum": languages, "description": "Default python"},
                "code": {"type": "string", "description": "Complete program to run"},
            }, "required": ["code"]},
            run=run_code,
            running=lambda a: f"Running code{': ' + _first_line(a.get('code', '')) if _first_line(a.get('code', '')) else ''}",
            done=lambda a: "Ran code",
            kind="code", read_only=False, egress=egress,
        ),
        Tool(
            name="write_file",
            description="Create or overwrite a text file in the task workspace (shared with the user).",
            parameters={"type": "object", "properties": {
                "path": {"type": "string", "description": "Relative path, e.g. report.md"},
                "content": {"type": "string"},
            }, "required": ["path", "content"]},
            run=write_file,
            running=lambda a: f"Writing {a.get('path', 'a file')}",
            done=lambda a: f"Wrote {a.get('path', 'a file')}",
            kind="file", read_only=False,
        ),
        Tool(
            name="read_file",
            description="Read a text file from the task workspace.",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
            run=read_file,
            running=lambda a: f"Reading {a.get('path', 'a file')}",
            done=lambda a: f"Read {a.get('path', 'a file')}",
            kind="file",
        ),
        Tool(
            name="list_files",
            description="List files in the task workspace.",
            parameters={"type": "object", "properties": {}},
            run=list_files,
            running=lambda a: "Looking at the workspace",
            done=lambda a: "Listed workspace files",
            kind="file",
        ),
    ]
