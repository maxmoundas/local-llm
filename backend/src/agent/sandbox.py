"""Where agent-written code runs.

Each task gets a workspace directory (data/workspaces/<task_id>). Code runs in:

  DockerSandbox — one container per task (python:3.12-slim by default), started on first
                  use and kept until the task ends, so `pip install` persists between runs.
                  Only the workspace is mounted; no host env vars or API keys go in.
                  CPU/memory/pid limits; network on or off by setting.
  LocalSandbox  — a subprocess in the workspace with a scrubbed environment, CPU and file-size
                  limits. NOT isolation — model-written code can read your files. Opt-in only.

`detect_sandbox()` picks Docker when the daemon answers, else nothing (code tools are not offered).

Housekeeping: `remove_workspace()` when a chat is deleted, `sweep_workspaces()` for old ones, and
`DockerSandbox.remove_stale()` at startup for containers a crashed backend left running.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from src.logger import get_logger

_log = get_logger(__name__)

WORKSPACES_DIR = Path("data/workspaces")
OUTPUT_LIMIT = 8000
LANGUAGES = {"python": ("main.py", ["python", "-u"]), "bash": ("main.sh", ["bash"]),
             "javascript": ("main.js", ["node"])}


@dataclass
class RunResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False

    def text(self) -> str:
        parts = [f"exit code: {self.exit_code}" + (" (timed out)" if self.timed_out else "")]
        if self.stdout:
            parts.append("stdout:\n" + _clip(self.stdout))
        if self.stderr:
            parts.append("stderr:\n" + _clip(self.stderr))
        return "\n".join(parts)


def _clip(text: str, limit: int = OUTPUT_LIMIT) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n… [{len(text) - limit} characters cut] …\n{text[-half:]}"


def workspace_for(task_id: str) -> Path:
    path = (WORKSPACES_DIR / task_id).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def remove_workspace(task_id: str) -> None:
    """Delete one task's workspace (its files and anything code wrote there)."""
    if not task_id or "/" in task_id or "\\" in task_id or task_id.startswith("."):
        return
    shutil.rmtree(WORKSPACES_DIR / task_id, ignore_errors=True)


def sweep_workspaces(max_age_days: float, keep: set[str] | frozenset[str] = frozenset()) -> int:
    """Delete workspaces untouched for `max_age_days` (except ids in `keep`). Returns how many went."""
    if not max_age_days or max_age_days <= 0 or not WORKSPACES_DIR.is_dir():
        return 0
    cutoff = time.time() - max_age_days * 86400
    removed = 0
    for d in WORKSPACES_DIR.iterdir():
        if not d.is_dir() or d.name in keep:
            continue
        try:
            newest = max((p.stat().st_mtime for p in d.rglob("*")), default=d.stat().st_mtime)
        except OSError:
            continue
        if newest < cutoff:
            shutil.rmtree(d, ignore_errors=True)
            removed += 1
    if removed:
        _log.info("Removed %d agent workspace(s) older than %g days", removed, max_age_days)
    return removed


def safe_path(workspace: Path, relative: str) -> Path:
    """Resolve a model-supplied path inside the workspace, refusing escapes."""
    rel = str(relative or "").lstrip("/").removeprefix("workspace/")
    target = (workspace / rel).resolve()
    if target != workspace and workspace not in target.parents:
        raise ValueError(f"Path is outside the workspace: {relative}")
    return target


class Sandbox:
    name = "none"
    network = False

    @property
    def languages(self) -> list[str]:
        """Languages this sandbox can actually run (offered to the model as run_code's enum)."""
        return list(LANGUAGES)

    def run(self, task_id: str, language: str, code: str, timeout: int = 60) -> RunResult:
        raise NotImplementedError

    def close(self, task_id: str) -> None:
        pass


class DockerSandbox(Sandbox):
    name = "docker"

    def __init__(self, image: str = "python:3.12-slim", network: bool = True, memory: str = "1g",
                 cpus: str = "2") -> None:
        self.image = image
        self.network = network
        self.memory = memory
        self.cpus = cpus
        self._containers: dict[str, str] = {}
        self._lock = threading.Lock()
        self._image_ready = False
        self._languages: list[str] | None = None

    @property
    def languages(self) -> list[str]:
        """python and bash always; javascript only if the image has node (python:*-slim doesn't)."""
        if self._languages is None:
            langs = ["python", "bash"]
            try:
                self._ensure_image()
                probe = subprocess.run(["docker", "run", "--rm", "--network", "none", "--entrypoint", "sh",
                                        self.image, "-c", "command -v node"], capture_output=True, timeout=60)
                if probe.returncode == 0:
                    langs.append("javascript")
            except Exception as exc:
                _log.warning("Could not probe sandbox image for node: %s", exc)
            self._languages = langs
        return self._languages

    @staticmethod
    def remove_stale() -> int:
        """Remove task containers left running by a previous backend process (crash, kill -9)."""
        try:
            out = subprocess.run(["docker", "ps", "-aq", "--filter", "name=^llm-task-"],
                                 capture_output=True, text=True, timeout=30).stdout.split()
            if out:
                subprocess.run(["docker", "rm", "-f", *out], capture_output=True, timeout=120)
                _log.info("Removed %d stale sandbox container(s)", len(out))
            return len(out)
        except Exception as exc:
            _log.warning("Could not clean up stale sandbox containers: %s", exc)
            return 0

    def _ensure_image(self) -> None:
        if self._image_ready:
            return
        probe = subprocess.run(["docker", "image", "inspect", self.image], capture_output=True)
        if probe.returncode != 0:
            _log.info("Pulling sandbox image %s", self.image)
            subprocess.run(["docker", "pull", self.image], capture_output=True, check=True, timeout=600)
        self._image_ready = True

    def _container(self, task_id: str) -> str:
        with self._lock:
            if task_id in self._containers:
                return self._containers[task_id]
            self._ensure_image()
            name = f"llm-task-{task_id[:12]}-{uuid.uuid4().hex[:6]}"
            ws = workspace_for(task_id)
            cmd = [
                "docker", "run", "-d", "--rm", "--name", name,
                "--memory", self.memory, "--cpus", self.cpus, "--pids-limit", "256",
                "--security-opt", "no-new-privileges", "--cap-drop", "ALL",
                "-v", f"{ws}:/workspace", "-w", "/workspace",
                "-e", "HOME=/workspace", "-e", "PIP_DISABLE_PIP_VERSION_CHECK=1",
            ]
            if not self.network:
                cmd += ["--network", "none"]
            if hasattr(os, "getuid"):
                cmd += ["--user", f"{os.getuid()}:{os.getgid()}", "-e", "PYTHONUSERBASE=/workspace/.local",
                        "-e", "PATH=/workspace/.local/bin:/usr/local/bin:/usr/bin:/bin"]
            cmd += [self.image, "sleep", "infinity"]
            subprocess.run(cmd, capture_output=True, check=True, timeout=120)
            self._containers[task_id] = name
            return name

    def run(self, task_id: str, language: str, code: str, timeout: int = 60) -> RunResult:
        filename, interpreter = LANGUAGES[language]
        ws = workspace_for(task_id)
        (ws / filename).write_text(code, encoding="utf-8")
        name = self._container(task_id)
        cmd = ["docker", "exec", name, "timeout", "-s", "KILL", str(timeout), *interpreter, filename]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 15)
        except subprocess.TimeoutExpired as exc:
            return RunResult(137, _s(exc.stdout), _s(exc.stderr), timed_out=True)
        return RunResult(proc.returncode, proc.stdout, proc.stderr, timed_out=proc.returncode == 137)

    def close(self, task_id: str) -> None:
        with self._lock:
            name = self._containers.pop(task_id, None)
        if name:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)

    @staticmethod
    def available() -> bool:
        if not shutil.which("docker"):
            return False
        try:
            return subprocess.run(["docker", "info"], capture_output=True, timeout=5).returncode == 0
        except Exception:
            return False


class LocalSandbox(Sandbox):
    """Runs code as a plain subprocess. Not isolated — only enable when you accept that risk."""
    name = "local"

    network = True  # a plain subprocess has whatever network access this machine has

    @property
    def languages(self) -> list[str]:
        return [lang for lang, (_, interp) in LANGUAGES.items() if lang == "python" or shutil.which(interp[0])]

    def run(self, task_id: str, language: str, code: str, timeout: int = 60) -> RunResult:
        filename, interpreter = LANGUAGES[language]
        if interpreter[0] == "python":
            interpreter = [sys.executable, "-u"]
        ws = workspace_for(task_id)
        (ws / filename).write_text(code, encoding="utf-8")
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(ws), "LANG": "C.UTF-8",
               "PYTHONUSERBASE": str(ws / ".local")}

        def limits() -> None:  # runs in the child before exec (POSIX only)
            import resource
            resource.setrlimit(resource.RLIMIT_CPU, (timeout, timeout))
            resource.setrlimit(resource.RLIMIT_FSIZE, (200 * 2**20, 200 * 2**20))

        try:
            proc = subprocess.run([*interpreter, filename], cwd=ws, env=env, capture_output=True, text=True,
                                  timeout=timeout, preexec_fn=limits if os.name == "posix" else None)
        except subprocess.TimeoutExpired as exc:
            return RunResult(137, _s(exc.stdout), _s(exc.stderr), timed_out=True)
        return RunResult(proc.returncode, proc.stdout, proc.stderr)


def _s(v) -> str:
    if v is None:
        return ""
    return v.decode("utf-8", "replace") if isinstance(v, bytes) else v


def detect_sandbox(mode: str = "auto", network: bool = True, image: str = "python:3.12-slim") -> Sandbox | None:
    """mode: auto | docker | local | off."""
    if mode == "off":
        return None
    if mode == "local":
        return LocalSandbox()
    if DockerSandbox.available():
        return DockerSandbox(image=image, network=network)
    if mode == "docker":
        _log.warning("Code execution is set to docker but the Docker daemon is not reachable")
    return None
