"""Connectors: MCP (Model Context Protocol) servers exposed to the agent as ordinary tools.

Any MCP server works — email, calendar, GitHub, Slack, databases, your own scripts — and
every model can use them, because tools reach the model through the provider-neutral
tool layer (native tool calling or prompted JSON).

Configure servers in backend/connectors.yaml:

    servers:
      gmail:
        command: npx
        args: ["-y", "@some/gmail-mcp"]
        env: {GMAIL_TOKEN: "${GMAIL_TOKEN}"}       # ${VAR} expands from the environment
        confirm: ["send_*", "delete_*"]           # pause for approval before these tools run
      notes:
        url: https://example.com/mcp               # streamable HTTP server
        headers: {Authorization: "Bearer ${NOTES_TOKEN}"}

Sessions live on one background asyncio loop; the agent's worker threads call into it.
"""
from __future__ import annotations

import asyncio
import fnmatch
import os
import re
import threading
from concurrent.futures import Future
from pathlib import Path

import yaml

from src.agent.tools import Tool, ToolResult
from src.logger import get_logger

_log = get_logger(__name__)

CONNECTORS_PATH = Path(__file__).resolve().parents[2] / "connectors.yaml"
_CALL_TIMEOUT = 120.0
_RESULT_LIMIT = 8000
_ENV_RE = re.compile(r"\$\{(\w+)\}")


def _expand(value):
    if isinstance(value, str):
        return _ENV_RE.sub(lambda m: os.getenv(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def _tool_name(server: str, tool: str) -> str:
    # Provider tool names allow [a-zA-Z0-9_-]{1,64}.
    return re.sub(r"[^a-zA-Z0-9_-]", "_", f"{server}__{tool}")[:64]


def _root_cause(exc: BaseException) -> BaseException:
    """MCP transports raise ExceptionGroups; report the first real error inside."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return exc


def _field(obj, *names):
    """MCP SDK 2.x uses snake_case attributes where 1.x used camelCase."""
    for n in names:
        v = getattr(obj, n, None)
        if v is not None:
            return v
    return None


def _humanize(name: str) -> str:
    return name.replace("_", " ").replace("-", " ").strip().capitalize()


class ConnectorHub:
    def __init__(self, config_path: Path | str | None = None) -> None:
        self._path = Path(config_path) if config_path else CONNECTORS_PATH
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._sessions: dict[str, object] = {}
        self._tools: list[Tool] = []
        self._status: dict[str, str] = {}
        self._stop: asyncio.Event | None = None

    # ── lifecycle ─────────────────────────────────────────────────────────

    def config(self) -> dict:
        if not self._path.exists():
            return {}
        return _expand((yaml.safe_load(self._path.read_text(encoding="utf-8")) or {}).get("servers") or {})

    def start(self, timeout: float = 30.0) -> None:
        servers = self.config()
        if not servers:
            return
        try:
            import mcp  # noqa: F401
        except ImportError:
            _log.warning("connectors.yaml lists servers but the `mcp` package is not installed")
            return
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True, name="connectors")
        self._thread.start()
        futures = {name: self._submit(self._serve(name, spec)) for name, spec in servers.items()}
        ready = {name: self._submit(self._wait_ready(name)) for name in servers}
        for name, f in ready.items():
            try:
                f.result(timeout=timeout)
            except Exception as exc:
                self._status[name] = f"error: {exc}"
                _log.error("Connector %s did not start: %s", name, exc)
                futures[name].cancel()

    def stop(self) -> None:
        if self._loop and self._stop:
            self._loop.call_soon_threadsafe(self._stop.set)

    def _submit(self, coro) -> Future:
        assert self._loop is not None
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    async def _wait_ready(self, name: str) -> None:
        while name not in self._sessions:
            if self._status.get(name, "").startswith("error"):
                raise RuntimeError(self._status[name])
            await asyncio.sleep(0.05)

    async def _serve(self, name: str, spec: dict) -> None:
        """Hold one server's session open until shutdown."""
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        if self._stop is None:
            self._stop = asyncio.Event()
        try:
            if spec.get("url"):
                try:
                    from mcp.client.streamable_http import streamable_http_client as http_client
                except ImportError:  # older SDKs
                    from mcp.client.streamable_http import streamablehttp_client as http_client
                transport = http_client(spec["url"], headers=spec.get("headers") or None)
            else:
                params = StdioServerParameters(command=spec["command"], args=list(spec.get("args") or []),
                                               env={**os.environ, **(spec.get("env") or {})})
                transport = stdio_client(params)
            async with transport as streams:
                read, write = streams[0], streams[1]
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    listed = await session.list_tools()
                    self._register(name, spec, session, listed.tools)
                    self._sessions[name] = session
                    self._status[name] = f"connected ({len(listed.tools)} tools)"
                    _log.info("Connector %s: %d tools", name, len(listed.tools))
                    await self._stop.wait()
        except Exception as exc:
            cause = _root_cause(exc)
            self._status[name] = f"error: {type(cause).__name__}: {cause}"
            _log.error("Connector %s failed: %s", name, self._status[name])
        finally:
            self._sessions.pop(name, None)
            self._tools = [t for t in self._tools if not t.name.startswith(_tool_name(name, ""))]

    def _register(self, server: str, spec: dict, session, tools) -> None:
        confirm_patterns = list(spec.get("confirm") or [])
        allow = spec.get("tools")  # optional allow-list
        for t in tools:
            if allow and t.name not in allow:
                continue
            title = getattr(t, "title", None) or _humanize(t.name)
            raw_schema = _field(t, "input_schema", "inputSchema")
            schema = raw_schema if isinstance(raw_schema, dict) else {"type": "object", "properties": {}}
            schema = {"type": "object", "properties": {}, **schema}
            annotations = getattr(t, "annotations", None)
            read_only_hint = bool(getattr(annotations, "readOnlyHint", False)) if annotations else False
            self._tools.append(Tool(
                name=_tool_name(server, t.name),
                description=f"[{server}] {t.description or title}"[:1024],
                parameters=schema,
                run=self._caller(server, t.name),
                running=lambda a, s=server, ti=title: f"{ti} ({s})",
                done=lambda a, s=server, ti=title: f"{ti} ({s})",
                kind="connector",
                confirm=any(fnmatch.fnmatch(t.name, p) for p in confirm_patterns),
                read_only=False,
                # Results (email bodies, issues, files…) are written by other people; anything not
                # declared read-only by its server may send data out.
                untrusted=True,
                egress=not read_only_hint,
            ))

    def _caller(self, server: str, tool: str):
        def run(args: dict) -> ToolResult:
            session = self._sessions.get(server)
            if session is None:
                return ToolResult(text=f"Connector {server} is not connected.", summary="not connected")
            result = self._submit(session.call_tool(tool, args)).result(timeout=_CALL_TIMEOUT)
            parts = []
            for block in getattr(result, "content", None) or []:
                text = getattr(block, "text", None)
                parts.append(text if text is not None else f"[{getattr(block, 'type', 'content')}]")
            structured = _field(result, "structured_content", "structuredContent")
            if not parts and structured is not None:
                parts.append(str(structured))
            text = "\n".join(parts) or "(no output)"
            if len(text) > _RESULT_LIMIT:
                text = text[:_RESULT_LIMIT] + f"\n… [{len(text) - _RESULT_LIMIT} more characters]"
            failed = bool(_field(result, "is_error", "isError"))
            return ToolResult(text=("Error: " if failed else "") + text, summary="failed" if failed else "done",
                              ok=not failed)
        return run

    # ── queries ───────────────────────────────────────────────────────────

    @property
    def tools(self) -> list[Tool]:
        return list(self._tools)

    def status(self) -> dict[str, str]:
        return dict(self._status)

    @property
    def names(self) -> list[str]:
        return list(self._sessions)
