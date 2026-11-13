"""
FastAPI backend — serves the LLM inference API.

Endpoints
─────────
  GET  /health
  GET  /api/models
  GET  /api/chats
  POST /api/chats
  GET  /api/chats/{id}
  PATCH /api/chats/{id}
  DELETE /api/chats/{id}
  POST /api/chats/{id}/message   ← starts a background task, streams its events (SSE)
  GET  /api/chats/{id}/active    ← running task for a chat (reattach after reload)
  GET  /api/tasks/{id}/events    ← resumable SSE (?after=<seq>)
  POST /api/tasks/{id}/cancel | reply
  GET  /api/tasks/{id}/files/{path}
  POST /api/chats/{id}/regenerate | edit | versions | title | files
  GET  /api/chats/{id}/files/{file_id} | export
  GET  /api/search?q=            ← full-text search over every chat
  /api/projects/*                ← projects: instructions, knowledge files, memory
  /api/schedules/*               ← scheduled tasks
  GET  /api/usage                ← spend by day / model / chat
  POST /api/transcribe           ← local Whisper speech-to-text
  GET  /api/auth/status, POST /api/auth/login | logout
  GET  /api/connectors
  GET/PATCH /api/settings
  GET/PATCH /api/memory
  POST /api/decide               ← Jev passthrough (structured decisions)
  GET  /v1/models                ← OpenAI-compat
  POST /v1/chat/completions      ← OpenAI-compat
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlsplit

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask
from starlette.responses import FileResponse, JSONResponse, Response, StreamingResponse

from src import attachments, backup, chat_store, context, decision_service, memory_store, project_store, scheduler
from src import settings_store, transcribe, usage
from src.agent.connectors import ConnectorHub
from src.agent.router import ResearchRouter
from src.agent.runner import TaskAgent
from src.agent import sandbox as sb
from src.agent.sandbox import Sandbox, detect_sandbox, safe_path
from src.attachments import AttachmentError
from src.agent.tools import default_tools
from src.config import load_config
from src.decision_service import CALL_TYPE_MEMORY, DecisionError, DecisionService, noul
from src.guardrail_service import GuardrailService
from src.llm_service import LLMService
from src.logger import get_logger, setup_logging
from src.model_factory import create_local_llm
from src.providers import ProviderRegistry, build_registry
from src.providers.base import EFFORT_LEVELS
from src.settings_store import DEFAULT_GUARDRAIL_PROMPT, DEFAULT_SYSTEM_PROMPT
from src.tasks import store as task_store
from src.tasks.manager import TaskManager
from src.tasks.turn import TurnDeps, run_turn

if TYPE_CHECKING:  # torch is only needed on the CUDA backend
    from src.model import LocalLLM

logger = get_logger(__name__)
_llm: "LocalLLM | None" = None
_registry: ProviderRegistry | None = None
_llm_service: LLMService | None = None
_decision_svc: DecisionService | None = None
_system_prompt: str = DEFAULT_SYSTEM_PROMPT
_memory_enabled: bool = True
_jev_memory: bool = True
_guardrail_svc: GuardrailService | None = None
_guardrails_input: bool = True
_guardrails_output: bool = False
# Agent loop — None when disabled in settings (or before startup), so messages answer directly.
_router: ResearchRouter | None = None
_agent: TaskAgent | None = None
_sandbox: Sandbox | None = None
_sandbox_key: tuple | None = None
_connectors = ConnectorHub()
_tasks = TaskManager()
_task_limits: dict = {}
_monthly_budget: float | None = None
_whisper_model: str = "base"
_scheduler: scheduler.Scheduler | None = None

# ── Network security ─────────────────────────────────────────────────────
# The server binds to loopback by default. Set ACCESS_TOKEN (backend/.env) before exposing it on a
# network: every /api and /v1 request then needs the token (cookie from /api/auth/login, or an
# `Authorization: Bearer` header for API clients). Without a token, requests must address the server
# by a loopback host name, which blocks DNS-rebinding attacks from web pages.
_SERVER_CFG = load_config().get("server", {})
_ACCESS_TOKEN = os.getenv("ACCESS_TOKEN") or _SERVER_CFG.get("access_token") or ""
_ALLOWED_ORIGINS = set(_SERVER_CFG.get("cors_origins") or [
    "http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:4173", "http://127.0.0.1:4173",
])
_ALLOWED_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]", *(_SERVER_CFG.get("allowed_hosts") or [])}
_AUTH_COOKIE = "llm_session"
_OPEN_PATHS = {"/health", "/api/auth/status", "/api/auth/login", "/api/auth/logout"}

_MEMORY_THRESHOLD = 0.4
_HAS_DURABLE_FACT = noul(
    "The message reveals a lasting personal fact about the sender (their name, job, location, tools or languages "
    "they use, ongoing projects, stated preferences, or recurring constraints) worth remembering for future "
    "conversations, or explicitly asks the assistant to remember, forget, or correct something about them.",
    true="It states something durable about who the sender is, what they work on, or how they like things done, "
         "or asks for something to be remembered or forgotten.",
    false="It is only a question, task, or one-off detail with nothing lasting about the sender.",
)

_MEMORY_SECTION = """

<user_memory>
Facts about the user saved from earlier conversations (may be out of date):
{memory}
</user_memory>
Use these to personalize answers only when relevant. Don't recite or mention them unless asked; \
if the user says something that contradicts them, trust the user."""


def _memory_context() -> str:
    return memory_store.load() if _memory_enabled else ""


def _apply_settings(s: dict) -> None:
    """(Re)build every settings-driven service. Called at startup and after each PATCH."""
    global _system_prompt, _memory_enabled, _jev_memory, _monthly_budget, _whisper_model
    global _guardrail_svc, _guardrails_input, _guardrails_output, _router, _agent, _sandbox, _sandbox_key

    if _llm_service is not None:
        _llm_service.set_cheapest_model(s.get("cheapest_model_id"))
    _system_prompt = s.get("system_prompt") or DEFAULT_SYSTEM_PROMPT
    _memory_enabled = bool(s.get("memory_enabled", True))
    _jev_memory = bool(s.get("jev_memory", True))
    _monthly_budget = float(s["monthly_budget_usd"]) if s.get("monthly_budget_usd") else None
    _whisper_model = s.get("whisper_model") or "base"

    if _decision_svc is not None:
        _decision_svc.enabled = bool(s.get("jev_enabled", True))
        _decision_svc.model = s.get("jev_model") or decision_service.DEFAULT_MODEL

    _guardrails_input = bool(s.get("guardrails_input", True))
    _guardrails_output = bool(s.get("guardrails_output", False))
    if s.get("guardrails_enabled", False) and _llm_service is not None:
        _guardrail_svc = GuardrailService(
            _llm_service,
            prompt=s.get("guardrails_prompt") or DEFAULT_GUARDRAIL_PROMPT,
            denied_topics=s.get("guardrails_topics") or [],
            decisions=_decision_svc if s.get("jev_guardrails", True) else None,
            policy=s.get("guardrails_policy") or "strict",
        )
    else:
        _guardrail_svc = None

    key = (s.get("code_execution") or "auto", bool(s.get("sandbox_network", True)))
    if key != _sandbox_key:  # detecting Docker shells out, so only redo it when these change
        _sandbox, _sandbox_key = detect_sandbox(key[0], network=key[1]), key

    _task_limits.update(
        budget_usd=float(s["task_budget_usd"]) if s.get("task_budget_usd") else None,
        max_seconds=float(s["task_max_minutes"]) * 60 if s.get("task_max_minutes") else None,
        agent_model_id=s.get("agent_model_id") or None,
        guard_untrusted=bool(s.get("agent_guard_untrusted", True)),
        workspace_retention_days=float(s.get("workspace_retention_days") or 0),
    )
    if s.get("agent_enabled", True) and _llm_service is not None:
        _router = ResearchRouter(_llm_service, _decision_svc, use_jev=bool(s.get("jev_routing", True)),
                                 can_run_code=_sandbox is not None, connectors=_connectors.names)
        _agent = TaskAgent(_llm_service, _router, default_tools(), max_steps=int(s.get("agent_max_steps") or 12))
    else:
        _router = _agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _llm, _registry, _llm_service, _decision_svc
    cfg = load_config()
    setup_logging(cfg)
    logger.info("Starting backend")
    _llm = create_local_llm(cfg)
    _registry = build_registry(_llm)
    saved_settings = settings_store.load()
    _llm_service = LLMService(_registry, cheapest_model_id=saved_settings.get("cheapest_model_id"))
    _decision_svc = DecisionService(os.getenv("TYPESAFE_API_KEY"))
    decision_service.configure(_decision_svc)
    _apply_settings(saved_settings)
    _tasks.recover()
    cheapest = _registry.cheapest_model
    logger.info("Server ready — cheapest model: %s (%s) | jev=%s | agent=%s | code=%s",
                cheapest.name, cheapest.id,
                "on" if _decision_svc.available else ("off" if _decision_svc.configured else "no key"),
                "on" if _agent else "off", _sandbox.name if _sandbox else "off")
    threading.Thread(target=_llm.load, daemon=True, name="model-loader").start()
    threading.Thread(target=_start_connectors, daemon=True, name="connectors-start").start()
    threading.Thread(target=_housekeeping, daemon=True, name="housekeeping").start()
    global _scheduler
    _scheduler = scheduler.Scheduler(_start_scheduled_run)
    _scheduler.start()
    yield
    _scheduler.stop()
    _connectors.stop()


_HOUSEKEEPING_INTERVAL_S = 6 * 3600


def _housekeeping() -> None:
    """Startup + periodic cleanup: containers a crashed backend left behind, and old agent workspaces."""
    try:
        chat_store.sync_search_index()
    except Exception:
        logger.exception("Search index sync failed")
    if isinstance(_sandbox, sb.DockerSandbox):
        sb.DockerSandbox.remove_stale()
        _ = _sandbox.languages  # probe the image once now, not during the first task
    while True:
        try:
            sb.sweep_workspaces(_task_limits.get("workspace_retention_days") or 0, keep=_tasks.active_ids())
        except Exception:
            logger.exception("Workspace cleanup failed")
        time.sleep(_HOUSEKEEPING_INTERVAL_S)


def _metered(kind: str, chat_id: str | None, model_id: str | None, fn, *args, **kwargs):
    """Run an LLM call outside a chat turn and record what it cost (usage dashboard, monthly limit)."""
    with usage.attach(()), usage.meter() as m:
        try:
            return fn(*args, **kwargs)
        finally:
            if m.calls:
                try:
                    task_store.record_usage(chat_id, model_id, m.full_snapshot(), kind)
                except Exception:
                    logger.exception("Could not record %s usage", kind)


def _start_connectors() -> None:
    _connectors.start()
    if _connectors.names:
        _apply_settings(settings_store.load())  # let the router know about connected services


app = FastAPI(title="Local LLM API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=sorted(_ALLOWED_ORIGINS),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Task-Id"],
)


def _token_ok(candidate: str | None) -> bool:
    if not candidate:
        return False
    return hmac.compare_digest(hashlib.sha256(candidate.encode()).digest(),
                               hashlib.sha256(_ACCESS_TOKEN.encode()).digest())


def _authorized(request: Request) -> bool:
    if not _ACCESS_TOKEN:
        return True
    header = request.headers.get("authorization", "")
    bearer = header[7:].strip() if header.lower().startswith("bearer ") else None
    return _token_ok(bearer) or _token_ok(request.cookies.get(_AUTH_COOKIE))


def _host_name(host_header: str) -> str:
    return urlsplit(f"//{host_header}").hostname or ""


@app.middleware("http")
async def _security(request: Request, call_next):
    """Host check (DNS rebinding), Origin check (cross-site requests), and the optional access token."""
    path = request.url.path
    host = _host_name(request.headers.get("host", ""))
    if not _ACCESS_TOKEN and host not in _ALLOWED_HOSTS:
        return JSONResponse({"detail": f"Host {host!r} not allowed. Set ACCESS_TOKEN to serve on a network, "
                                       "or add the name to server.allowed_hosts in config.yaml."}, status_code=403)
    origin = request.headers.get("origin")
    if origin and request.method not in ("GET", "HEAD", "OPTIONS"):
        same_origin = urlsplit(origin).netloc == request.headers.get("host", "")
        if not same_origin and origin not in _ALLOWED_ORIGINS:
            return JSONResponse({"detail": "Cross-origin request blocked"}, status_code=403)
    if (path.startswith("/api") or path.startswith("/v1")) and path not in _OPEN_PATHS \
            and request.method != "OPTIONS" and not _authorized(request):
        return JSONResponse({"detail": "Authentication required"}, status_code=401)
    return await call_next(request)


@app.exception_handler(chat_store.InvalidChatId)
async def _invalid_chat_id(request: Request, exc: chat_store.InvalidChatId) -> JSONResponse:
    return JSONResponse({"detail": "Chat not found"}, status_code=404)


# ── Helpers ────────────────────────────────────────────────────────────

def _require_llm() -> LocalLLM:
    if _llm is None or not _llm.is_loaded:
        raise HTTPException(503, "Model not loaded yet — try again in a moment")
    return _llm


def _require_registry() -> ProviderRegistry:
    if _registry is None:
        raise HTTPException(503, "Server not ready — try again in a moment")
    return _registry


def _require_service() -> LLMService:
    if _llm_service is None:
        raise HTTPException(503, "Server not ready — try again in a moment")
    return _llm_service


_PROJECT_SECTION = """

<project name="{name}">
This conversation is part of the user's project "{name}".{description}{instructions}{memory}{knowledge}
</project>"""

KNOWLEDGE_SHARE = 0.4  # at most this share of the context window goes to project knowledge files


def _project_of(chat: dict) -> dict | None:
    return project_store.get(chat["project_id"]) if chat.get("project_id") else None


def _model_or_none(model_id: str | None):
    if not model_id or _registry is None:
        return None
    try:
        return _registry.get_model(model_id)
    except KeyError:
        return None


def _project_section(project: dict, model) -> str:
    desc = f"\nAbout it: {project['description']}" if project.get("description") else ""
    instr = f"\n\nProject instructions (follow these):\n{project['instructions']}" if project.get("instructions") else ""
    mem_text = memory_store.load(project_store.memory_path(project["id"])) if _memory_enabled else ""
    mem = f"\n\nFacts learned in this project's chats:\n{mem_text}" if mem_text else ""
    docs, budget = [], context.budget_for(model) * KNOWLEDGE_SHARE
    for meta, text in project_store.knowledge(project):
        cost = context.estimate_tokens(text)
        if cost > budget:
            text = text[: int(max(budget, 0) * context.CHARS_PER_TOKEN)] + "\n[… truncated to fit the context window]"
            cost = budget
        if budget <= 0:
            docs.append(f'<document name="{meta["name"]}">(omitted: knowledge files exceed the context window)</document>')
            continue
        docs.append(context.document_block(meta["name"], text))
        budget -= cost
    knowledge = ("\n\nProject knowledge files:\n" + "\n\n".join(docs)) if docs else ""
    return _PROJECT_SECTION.format(name=project["name"], description=desc, instructions=instr, memory=mem,
                                   knowledge=knowledge)


def _system_for(chat: dict, model=None) -> str:
    """System prompt: the chat's own prompt (or the global one), date, memory, and its project."""
    now = datetime.now()
    custom = ((chat.get("settings") or {}).get("system_prompt") or "").strip()
    system = f"{custom or _system_prompt}\n\nToday's date is {now:%A, %B} {now.day}, {now.year}."
    mem = _memory_context()
    if mem:
        system += _MEMORY_SECTION.format(memory=mem)
    project = _project_of(chat)
    if project:
        system += _project_section(project, model)
    return system


def _build_messages(chat: dict, model_id: str | None = None) -> list[dict]:
    """Provider-ready history: system prompt first, attachments expanded, fitted to the model's context.

    Empty turns (e.g. a reply stopped before any text) are dropped and consecutive
    same-role turns merged, since some providers require strict user/assistant alternation.
    """
    model = _model_or_none(model_id)
    chat_id = chat.get("id")
    summarize = _llm_service.summarize_history if _llm_service is not None else None
    return context.build(
        chat, _system_for(chat, model), model,
        files_root=chat_store.files_dir(chat_id) if chat_id else None,
        summarize=summarize,
        save_summary=(lambda summary: chat_store.update_chat(chat_id, context_summary=summary)) if chat_id else None,
    )


def _maybe_update_memory(user_message: str, svc: LLMService, prev_reply: str = "",
                         path: Path | None = None) -> dict | None:
    """Background task: extract durable facts from the user message and update memory if changed.

    Returns {"added": [...], "removed": [...]} when memory changed, else None.

    When Jev is on for memory, it first checks whether the message holds anything worth
    remembering, which skips the LLM call for the large majority of messages.
    """
    if _jev_memory and _decision_svc is not None and _decision_svc.available:
        try:
            p = _decision_svc.ask({"message": user_message[:2000]}, {"durable": _HAS_DURABLE_FACT},
                                  call_type=CALL_TYPE_MEMORY)["durable"].noul
            if p < _MEMORY_THRESHOLD:
                return None
        except DecisionError:
            pass  # no gate — let the LLM decide
    # Under the store's lock so concurrent turns can't drop each other's updates.
    change = memory_store.update(lambda current: svc.extract_memory(user_message, current, prev_reply), path)
    if change:
        logger.info("Memory updated (+%d / -%d)", len(change["added"]), len(change["removed"]))
    return change


def _model_dicts(registry: ProviderRegistry) -> list[dict]:
    return [m.public() for m in registry.list_models()]


# ── Health & model discovery ───────────────────────────────────────────

@app.get("/health")
async def health() -> dict[str, Any]:
    loaded = _llm is not None and _llm.is_loaded
    return {
        "status": "ok" if loaded else "loading",
        "model": _llm.repo_id if _llm else None,
        "local_model_loaded": loaded,
        "jev_available": bool(_decision_svc and _decision_svc.available),
    }


@app.get("/api/models")
async def api_list_models() -> list[dict]:
    return _model_dicts(_require_registry())


# ── Chat management (/api/chats/*) ─────────────────────────────────────

class ChatSettings(BaseModel):
    system_prompt: str | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    effort: Literal["low", "medium", "high"] | None = None


class ChatUpdate(BaseModel):
    name: str | None = None
    project_id: str | None = None        # "" removes the chat from its project
    settings: ChatSettings | None = None


class MessageRequest(BaseModel):
    content: str = ""
    image_base64: str | None = None      # legacy single image; prefer `attachments`
    attachments: list[str] | None = None  # ids returned by POST /api/chats/{id}/files
    model_id: str | None = None


class CreateChatRequest(BaseModel):
    project_id: str | None = None


@app.get("/api/chats")
async def api_list_chats(project_id: str | None = None) -> list[dict]:
    return chat_store.list_chats(project_id)


@app.post("/api/chats", status_code=201)
async def api_create_chat(req: CreateChatRequest | None = None) -> dict:
    project_id = req.project_id if req else None
    if project_id and project_store.get(project_id) is None:
        raise HTTPException(404, "Project not found")
    return chat_store.create_chat(project_id=project_id)


@app.get("/api/chats/{chat_id}")
async def api_get_chat(chat_id: str) -> dict:
    chat = chat_store.get_chat(chat_id)
    if chat is None:
        raise HTTPException(404, "Chat not found")
    return chat


@app.patch("/api/chats/{chat_id}")
async def api_update_chat(chat_id: str, req: ChatUpdate) -> dict:
    chat = chat_store.get_chat(chat_id)
    if chat is None:
        raise HTTPException(404, "Chat not found")
    if req.name is not None:
        chat = chat_store.rename_chat(chat_id, req.name)
    if req.project_id is not None:
        if req.project_id and project_store.get(req.project_id) is None:
            raise HTTPException(404, "Project not found")
        chat = chat_store.update_chat(chat_id, project_id=req.project_id or None)
    if req.settings is not None:
        settings = {k: v for k, v in req.settings.model_dump().items() if v not in (None, "")}
        chat = chat_store.update_chat(chat_id, settings=settings or None)
    return chat


@app.delete("/api/chats/{chat_id}", status_code=204, response_class=Response)
async def api_delete_chat(chat_id: str) -> Response:
    if not chat_store.delete_chat(chat_id):
        raise HTTPException(404, "Chat not found")
    for t in task_store.for_chat(chat_id):
        _tasks.cancel(t["id"])
    for task_id in task_store.task_ids_for_chat(chat_id):
        sb.remove_workspace(task_id)
    task_store.delete_for_chat(chat_id)
    return Response(status_code=204)


@app.post("/api/chats/{chat_id}/pin")
async def api_toggle_pin(chat_id: str) -> dict:
    chat = chat_store.toggle_pin(chat_id)
    if chat is None:
        raise HTTPException(404, "Chat not found")
    return {"pinned": chat.get("pinned", False)}


# ── Settings (/api/settings) ───────────────────────────────────────────

class SettingsUpdate(BaseModel):
    cheapest_model_id: str | None = None
    system_prompt: str | None = None
    memory_enabled: bool | None = None
    guardrails_enabled: bool | None = None
    guardrails_input: bool | None = None
    guardrails_output: bool | None = None
    guardrails_prompt: str | None = None
    guardrails_topics: list[str] | None = None
    guardrails_policy: Literal["strict", "permissive"] | None = None
    agent_enabled: bool | None = None
    agent_max_steps: int | None = Field(default=None, ge=1, le=40)
    agent_model_id: str | None = None
    task_budget_usd: float | None = Field(default=None, ge=0, le=100)
    task_max_minutes: float | None = Field(default=None, ge=0, le=240)
    code_execution: Literal["auto", "docker", "local", "off"] | None = None
    sandbox_network: bool | None = None
    agent_guard_untrusted: bool | None = None
    workspace_retention_days: float | None = Field(default=None, ge=0, le=3650)
    jev_enabled: bool | None = None
    jev_model: str | None = None
    jev_routing: bool | None = None
    jev_guardrails: bool | None = None
    jev_memory: bool | None = None
    display_name: str | None = None
    monthly_budget_usd: float | None = Field(default=None, ge=0, le=100_000)
    whisper_model: Literal["tiny", "base", "small", "medium", "large-v3"] | None = None


@app.get("/api/settings")
async def api_get_settings() -> dict:
    registry = _require_registry()
    current = settings_store.load()
    return {
        **current,
        "default_system_prompt": DEFAULT_SYSTEM_PROMPT,
        "default_guardrail_prompt": DEFAULT_GUARDRAIL_PROMPT,
        "jev_configured": bool(_decision_svc and _decision_svc.configured),
        "models": _model_dicts(registry),
        "sandbox_active": _sandbox.name if _sandbox else "off",
        "connectors": _connectors.status(),
        "transcription_available": transcribe.available(),
        "access_token_set": bool(_ACCESS_TOKEN),
    }


@app.patch("/api/settings")
async def api_update_settings(req: SettingsUpdate) -> dict:
    # Only save fields that were explicitly provided in the request body.
    updates = {k: v for k, v in req.model_dump().items() if k in req.model_fields_set}
    updated = settings_store.save(updates)
    _apply_settings(updated)
    logger.info("Settings updated: %s", sorted(updates))
    return updated


# ── Memory (/api/memory) ───────────────────────────────────────────────────

class MemoryUpdate(BaseModel):
    content: str


@app.get("/api/memory")
async def api_get_memory() -> dict:
    return {"content": memory_store.load()}


@app.patch("/api/memory")
async def api_update_memory(req: MemoryUpdate) -> dict:
    # Off the event loop: save() waits if a background extraction is mid-update.
    await run_in_threadpool(memory_store.save, req.content)
    return {"content": req.content.strip()}


# ── Decisions (/api/decide) ─────────────────────────────────────────────

class DecideRequest(BaseModel):
    state: Any
    questions: dict[str, dict]


@app.post("/api/decide")
async def api_decide(req: DecideRequest) -> dict:
    """Ask Jev typed questions about a state. Same request shape as TypeSafe's /v1/systemone."""
    if _decision_svc is None or not _decision_svc.available:
        raise HTTPException(503, "Jev is not configured or is disabled in settings")
    try:
        answers = await run_in_threadpool(_decision_svc.ask, req.state, req.questions)
    except DecisionError as exc:
        raise HTTPException(502, f"Jev request failed: {exc}")
    return {"answers": {k: vars(a) for k, a in answers.items()}}


# ── Sending a message: background tasks + resumable SSE ─────────────────

_SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}


def _sse_stream(task_id: str, after: int):
    """Event stream for a task. `id:` carries the seq so clients can resume with ?after=."""
    for evt in _tasks.subscribe(task_id, after):
        if evt.get("type") == "ping":
            yield ": ping\n\n"
            continue
        yield f"id: {evt['seq']}\ndata: {json.dumps(evt)}\n\n"


def _workspace_files(chat: dict | None) -> list[tuple[str, Path]]:
    """The user's files in this chat (and its project) that the agent may open with code."""
    if not chat:
        return []
    out = []
    root = chat_store.files_dir(chat["id"])
    for m in chat.get("messages", []):
        for f in m.get("attachments") or []:
            p = attachments.path_for(root, f)
            if p:
                out.append((f["name"], p))
    project = _project_of(chat)
    if project:
        proot = project_store.files_dir(project["id"])
        out += [(f["name"], p) for f in project.get("files", []) if (p := attachments.path_for(proot, f))]
    return out


def _turn_deps(svc: LLMService, chat: dict | None = None, unattended: bool = False) -> TurnDeps:
    # Chats in a project learn into that project's memory; other chats into the user's profile.
    mem_path = project_store.memory_path(chat["project_id"]) if chat and chat.get("project_id") else None
    return TurnDeps(
        svc=svc,
        build_messages=_build_messages,
        guardrails=_guardrail_svc,
        guard_input=_guardrails_input,
        guard_output=_guardrails_output,
        update_memory=(lambda text, prev="": _maybe_update_memory(text, svc, prev, mem_path)) if _memory_enabled else None,
        user_memory=_memory_context(),
        router=_router,
        agent=_agent,
        sandbox=_sandbox,
        connectors=_connectors if _connectors.names else None,
        agent_model_id=_task_limits.get("agent_model_id"),
        budget_usd=_task_limits.get("budget_usd"),
        max_seconds=_task_limits.get("max_seconds"),
        guard_untrusted=_task_limits.get("guard_untrusted", True),
        unattended=unattended,
        workspace_files=_workspace_files,
    )


def _month_spend() -> float:
    start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0).timestamp()
    return sum((t["usage"] or {}).get("cost_usd", 0.0) for t in task_store.usage_since(start))


def _start_turn(chat_id: str, model_id: str | None, unattended: bool = False, **turn_kwargs) -> tuple[str, str]:
    """Validate and start a turn as a background task. Returns (task_id, model_id)."""
    registry = _require_registry()
    svc = _require_service()
    chat = chat_store.get_chat(chat_id)
    if chat is None:
        raise HTTPException(404, "Chat not found")
    if _tasks.active_for_chat(chat_id):
        raise HTTPException(409, "This chat is still working on the previous message")
    model_id = model_id or registry.default_model_id
    try:
        registry.get_provider(model_id)  # validate model exists
    except KeyError:
        raise HTTPException(400, f"Unknown model: {model_id}")
    _check_monthly_budget()
    deps = _turn_deps(svc, chat, unattended=unattended)
    content = turn_kwargs.pop("content", "")
    image_base64 = turn_kwargs.pop("image_base64", None)
    task_id = _tasks.start(chat_id, model_id,
                           lambda ctx: run_turn(ctx, chat_id, content, image_base64, model_id, deps, **turn_kwargs))
    return task_id, model_id


def _stream_task(task_id: str) -> StreamingResponse:
    return StreamingResponse(_sse_stream(task_id, -1), media_type="text/event-stream",
                             headers={**_SSE_HEADERS, "X-Task-Id": task_id})


def _resolve_attachments(chat_id: str, ids: list[str]) -> list[dict]:
    root = chat_store.files_dir(chat_id)
    metas = []
    for file_id in ids:
        meta = attachments.load_meta(root, file_id)
        if meta is None:
            raise HTTPException(400, f"Unknown attachment: {file_id}")
        metas.append(meta)
    return metas


@app.post("/api/chats/{chat_id}/message")
async def api_send_message(chat_id: str, req: MessageRequest) -> Any:
    """Start the turn as a background task and stream its events.

    The task keeps running if the client disconnects; reconnect with
    GET /api/tasks/{task_id}/events?after=<last seq>. Stop it with POST /api/tasks/{id}/cancel.
    """
    metas = _resolve_attachments(chat_id, req.attachments) if req.attachments else []
    if not req.content.strip() and not metas and not req.image_base64:
        raise HTTPException(400, "Message is empty")
    task_id, _ = _start_turn(chat_id, req.model_id, content=req.content, image_base64=req.image_base64,
                             attachments=metas)
    return _stream_task(task_id)


class RegenerateRequest(BaseModel):
    index: int | None = None     # the assistant message to replace; None = answer the last user message
    model_id: str | None = None


@app.post("/api/chats/{chat_id}/regenerate")
async def api_regenerate(chat_id: str, req: RegenerateRequest) -> Any:
    chat = chat_store.get_chat(chat_id)
    if chat is None:
        raise HTTPException(404, "Chat not found")
    msgs = chat["messages"]
    if req.index is not None and (not 0 <= req.index < len(msgs) or msgs[req.index]["role"] != "assistant"):
        raise HTTPException(400, "index must point at an assistant message")
    task_id, _ = _start_turn(chat_id, req.model_id, mode="regenerate", index=req.index)
    return _stream_task(task_id)


class EditRequest(BaseModel):
    index: int
    content: str
    model_id: str | None = None
    attachments: list[str] | None = None   # ids; None keeps the original message's attachments


@app.post("/api/chats/{chat_id}/edit")
async def api_edit(chat_id: str, req: EditRequest) -> Any:
    chat = chat_store.get_chat(chat_id)
    if chat is None:
        raise HTTPException(404, "Chat not found")
    msgs = chat["messages"]
    if not 0 <= req.index < len(msgs) or msgs[req.index]["role"] != "user":
        raise HTTPException(400, "index must point at a user message")
    metas = (_resolve_attachments(chat_id, req.attachments) if req.attachments is not None
             else list(msgs[req.index].get("attachments") or []))
    if not req.content.strip() and not metas:
        raise HTTPException(400, "Message is empty")
    task_id, _ = _start_turn(chat_id, req.model_id, content=req.content, attachments=metas, mode="edit",
                             index=req.index)
    return _stream_task(task_id)


class VersionRequest(BaseModel):
    index: int
    version: int


@app.post("/api/chats/{chat_id}/versions")
async def api_switch_version(chat_id: str, req: VersionRequest) -> dict:
    if _tasks.active_for_chat(chat_id):
        raise HTTPException(409, "Wait for the current reply to finish")
    chat = chat_store.switch_version(chat_id, req.index, req.version)
    if chat is None:
        raise HTTPException(400, "No such version")
    return chat


@app.post("/api/chats/{chat_id}/title")
async def api_generate_title(chat_id: str) -> dict:
    chat = chat_store.get_chat(chat_id)
    if chat is None:
        raise HTTPException(404, "Chat not found")
    title = await run_in_threadpool(_metered, "title", chat_id, None,
                                    _require_service().generate_title_from_history, chat["messages"])
    return chat_store.rename_chat(chat_id, title)


class FileUpload(BaseModel):
    name: str
    data_base64: str
    mime: str | None = None


@app.post("/api/chats/{chat_id}/files", status_code=201)
async def api_upload_chat_file(chat_id: str, req: FileUpload) -> dict:
    if chat_store.get_chat(chat_id) is None:
        raise HTTPException(404, "Chat not found")
    try:
        return await run_in_threadpool(attachments.save_base64, chat_store.files_dir(chat_id), req.name,
                                       req.data_base64, req.mime)
    except AttachmentError as exc:
        raise HTTPException(400, str(exc))


@app.get("/api/chats/{chat_id}/files/{file_id}")
async def api_chat_file(chat_id: str, file_id: str, download: bool = False) -> Any:
    root = chat_store.files_dir(chat_id)
    meta, path = attachments.load_meta(root, file_id), attachments.path_for(root, file_id)
    if meta is None or path is None:
        raise HTTPException(404, "File not found")
    if meta["kind"] == "image" and not download:
        return FileResponse(path, media_type=meta["mime"])
    return FileResponse(path, filename=meta["name"], media_type=meta["mime"])


@app.get("/api/chats/{chat_id}/export")
async def api_export_chat(chat_id: str, format: Literal["md", "json"] = "md") -> Response:
    chat = chat_store.get_chat(chat_id)
    if chat is None:
        raise HTTPException(404, "Chat not found")
    stem = "".join(c if c.isalnum() or c in " -_" else "_" for c in chat["name"]).strip()[:60] or "chat"
    if format == "json":
        body, media = json.dumps(chat_store.export_json(chat), indent=2), "application/json"
    else:
        body, media = chat_store.export_markdown(chat), "text/markdown; charset=utf-8"
    return Response(body, media_type=media,
                    headers={"Content-Disposition": f'attachment; filename="{stem}.{format}"'})


@app.get("/api/search")
async def api_search(q: str = "", limit: int = 50) -> list[dict]:
    return await run_in_threadpool(chat_store.search, q, max(1, min(limit, 200)))


@app.get("/api/chats/{chat_id}/active")
async def api_active_task(chat_id: str) -> dict | None:
    """The task currently running for this chat, if any (used to reattach after a reload)."""
    return _tasks.active_for_chat(chat_id)


@app.get("/api/tasks/{task_id}/events")
async def api_task_events(task_id: str, after: int = -1) -> Any:
    if task_store.get(task_id) is None:
        raise HTTPException(404, "Task not found")
    return StreamingResponse(_sse_stream(task_id, after), media_type="text/event-stream", headers=_SSE_HEADERS)


@app.get("/api/tasks/{task_id}")
async def api_get_task(task_id: str) -> dict:
    task = task_store.get(task_id)
    if task is None:
        raise HTTPException(404, "Task not found")
    return {**task, "active": _tasks.is_active(task_id)}


@app.post("/api/tasks/{task_id}/cancel")
async def api_cancel_task(task_id: str) -> dict:
    return {"cancelled": _tasks.cancel(task_id)}


class TaskReply(BaseModel):
    request_id: str | None = None
    answer: str | None = None
    approved: bool | None = None


@app.post("/api/tasks/{task_id}/reply")
async def api_reply_task(task_id: str, req: TaskReply) -> dict:
    payload = {k: v for k, v in req.model_dump().items() if v is not None}
    if not _tasks.reply(task_id, payload):
        raise HTTPException(409, "This task is not waiting for input")
    return {"ok": True}


@app.get("/api/tasks/{task_id}/files/{path:path}")
async def api_task_file(task_id: str, path: str) -> Any:
    """Download a file the agent created in this task's workspace."""
    if task_store.get(task_id) is None:
        raise HTTPException(404, "Task not found")
    workspace = (sb.WORKSPACES_DIR / task_id).resolve()
    try:
        target = safe_path(workspace, path)
    except ValueError:
        raise HTTPException(400, "Invalid path")
    if not target.is_file():
        raise HTTPException(404, "File not found")
    return FileResponse(target, filename=target.name)


@app.get("/api/connectors")
async def api_connectors() -> dict:
    return {"connectors": _connectors.status(), "tools": [t.name for t in _connectors.tools]}


# ── Projects (/api/projects) ──────────────────────────────────────────

class ProjectCreate(BaseModel):
    name: str
    description: str = ""
    instructions: str = ""


class ProjectUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    instructions: str | None = None


def _project_or_404(project_id: str) -> dict:
    project = project_store.get(project_id)
    if project is None:
        raise HTTPException(404, "Project not found")
    return project


def _project_out(project: dict) -> dict:
    chats = chat_store.list_chats(project["id"])
    return {**project, "chat_count": len(chats)}


@app.get("/api/projects")
async def api_list_projects() -> list[dict]:
    return [_project_out(p) for p in project_store.list_projects()]


@app.post("/api/projects", status_code=201)
async def api_create_project(req: ProjectCreate) -> dict:
    return _project_out(project_store.create(req.name, req.description, req.instructions))


@app.get("/api/projects/{project_id}")
async def api_get_project(project_id: str) -> dict:
    project = _project_or_404(project_id)
    return {**_project_out(project), "chats": chat_store.list_chats(project_id),
            "memory": memory_store.load(project_store.memory_path(project_id))}


@app.patch("/api/projects/{project_id}")
async def api_update_project(project_id: str, req: ProjectUpdate) -> dict:
    _project_or_404(project_id)
    return _project_out(project_store.update(project_id, **req.model_dump()))


@app.delete("/api/projects/{project_id}", status_code=204, response_class=Response)
async def api_delete_project(project_id: str) -> Response:
    _project_or_404(project_id)
    for c in chat_store.list_chats(project_id):  # chats survive, outside any project
        chat_store.update_chat(c["id"], project_id=None)
    project_store.delete(project_id)
    return Response(status_code=204)


@app.post("/api/projects/{project_id}/files", status_code=201)
async def api_add_project_file(project_id: str, req: FileUpload) -> dict:
    _project_or_404(project_id)
    try:
        return await run_in_threadpool(project_store.add_file, project_id, req.name, req.data_base64, req.mime)
    except AttachmentError as exc:
        raise HTTPException(400, str(exc))


@app.delete("/api/projects/{project_id}/files/{file_id}", status_code=204, response_class=Response)
async def api_remove_project_file(project_id: str, file_id: str) -> Response:
    if not project_store.remove_file(project_id, file_id):
        raise HTTPException(404, "File not found")
    return Response(status_code=204)


@app.get("/api/projects/{project_id}/files/{file_id}")
async def api_project_file(project_id: str, file_id: str) -> Any:
    root = project_store.files_dir(project_id)
    meta, path = attachments.load_meta(root, file_id), attachments.path_for(root, file_id)
    if meta is None or path is None:
        raise HTTPException(404, "File not found")
    return FileResponse(path, filename=meta["name"], media_type=meta["mime"])


@app.patch("/api/projects/{project_id}/memory")
async def api_update_project_memory(project_id: str, req: MemoryUpdate) -> dict:
    _project_or_404(project_id)
    await run_in_threadpool(memory_store.save, req.content, project_store.memory_path(project_id))
    return {"content": req.content.strip()}


# ── Usage (/api/usage) ─────────────────────────────────────────────────

@app.get("/api/usage")
async def api_usage(days: int = 30) -> dict:
    """Spending and tokens by day, model, and chat over the last `days` days, plus month to date."""
    days = max(1, min(days, 365))
    now = datetime.now()
    since = (now - timedelta(days=days - 1)).replace(hour=0, minute=0, second=0, microsecond=0)
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    rows = await run_in_threadpool(task_store.usage_since, min(since, month_start).timestamp())
    names = {m.id: m.name for m in _registry.list_models()} if _registry else {}
    chat_names = {c["id"]: c["name"] for c in chat_store.list_chats()}

    per_day = {(since + timedelta(days=i)).strftime("%Y-%m-%d"): {"cost_usd": 0.0, "input_tokens": 0,
                                                                   "output_tokens": 0, "turns": 0}
               for i in range(days)}
    per_model: dict[str, dict] = {}
    per_chat: dict[str, dict] = {}
    total = {"cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0, "turns": 0}
    month = 0.0
    for t in rows:
        u = t["usage"] or {}
        cost = float(u.get("cost_usd") or 0.0)
        if t["created_at"] >= month_start.timestamp():
            month += cost
        if t["created_at"] < since.timestamp():
            continue
        day = datetime.fromtimestamp(t["created_at"]).strftime("%Y-%m-%d")
        d = per_day.setdefault(day, {"cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0, "turns": 0})
        for bucket in (d, total):
            bucket["cost_usd"] += cost
            bucket["input_tokens"] += int(u.get("input_tokens") or 0)
            bucket["output_tokens"] += int(u.get("output_tokens") or 0)
            bucket["turns"] += 1 if t.get("kind", "turn") == "turn" else 0
        by_model = u.get("by_model") or ({t["model_id"]: u} if t.get("model_id") else {})
        for mid, mu in by_model.items():
            m = per_model.setdefault(mid, {"model_id": mid, "name": names.get(mid, mid), "cost_usd": 0.0,
                                           "input_tokens": 0, "output_tokens": 0, "calls": 0})
            m["cost_usd"] += float(mu.get("cost_usd") or 0.0)
            m["input_tokens"] += int(mu.get("input_tokens") or 0)
            m["output_tokens"] += int(mu.get("output_tokens") or 0)
            m["calls"] += int(mu.get("calls") or 0)
        key = t.get("chat_id") or ""
        c = per_chat.setdefault(key, {"chat_id": t.get("chat_id"),
                                      "name": chat_names.get(key, "Deleted chats") if key else "Deleted chats",
                                      "cost_usd": 0.0, "turns": 0})
        c["cost_usd"] += cost
        c["turns"] += 1 if t.get("kind", "turn") == "turn" else 0

    def rounded(d: dict) -> dict:
        return {**d, "cost_usd": round(d["cost_usd"], 6)}

    return {
        "days": [{"date": k, **rounded(v)} for k, v in sorted(per_day.items())],
        "models": sorted((rounded(m) for m in per_model.values()), key=lambda m: -m["cost_usd"]),
        "chats": sorted((rounded(c) for c in per_chat.values()), key=lambda c: -c["cost_usd"])[:15],
        "total": rounded(total),
        "month_to_date_usd": round(month, 6),
        "monthly_budget_usd": _monthly_budget,
    }


# ── Backup & restore (/api/backup, /api/restore) ───────────────────────

MAX_RESTORE_BYTES = 4 * 1024**3


@app.get("/api/backup")
async def api_backup() -> Any:
    """Download chats, projects, memory, settings, and schedules as one zip (no API keys)."""
    path = await run_in_threadpool(backup.create_backup)
    name = f"local-llm-backup-{datetime.now():%Y-%m-%d}.zip"
    return FileResponse(path, filename=name, media_type="application/zip",
                        background=BackgroundTask(path.unlink, missing_ok=True))


@app.post("/api/restore")
async def api_restore(request: Request, mode: Literal["merge", "replace"] = "merge") -> dict:
    """Restore a backup zip sent as the raw request body. See src/backup.py for the merge/replace rules."""
    import tempfile
    fd, tmp_name = tempfile.mkstemp(prefix="local-llm-restore-", suffix=".zip")
    tmp = Path(tmp_name)
    try:
        size = 0
        with os.fdopen(fd, "wb") as f:
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_RESTORE_BYTES:
                    raise HTTPException(413, "Backup is too large")
                f.write(chunk)
        if size == 0:
            raise HTTPException(400, "Send the backup zip as the request body")
        if _tasks.active_ids():
            raise HTTPException(409, "Wait for running chats to finish before restoring")
        try:
            result = await run_in_threadpool(backup.restore, tmp, mode)
        except backup.BackupError as exc:
            raise HTTPException(400, str(exc))
        if result.get("settings_restored"):
            _apply_settings(settings_store.load())
        logger.info("Backup restored: %s", result)
        return result
    finally:
        tmp.unlink(missing_ok=True)


# ── Scheduled tasks (/api/schedules) ───────────────────────────────────

class ScheduleIn(BaseModel):
    name: str | None = None
    prompt: str | None = None
    model_id: str | None = None
    project_id: str | None = None
    kind: Literal["daily", "weekly", "interval"] | None = None
    time: str | None = None
    days: list[int] | None = None
    every_minutes: int | None = None
    enabled: bool | None = None


def _start_scheduled_run(s: dict) -> tuple[str | None, str | None]:
    """Post a schedule's prompt to its chat (created on first run) as an unattended background turn."""
    chat = chat_store.get_chat(s["chat_id"]) if s.get("chat_id") else None
    if chat is None:
        project_id = s.get("project_id") if s.get("project_id") and project_store.get(s["project_id"]) else None
        chat = chat_store.create_chat(project_id=project_id, name=s["name"], schedule_id=s["id"])
    task_id, _ = _start_turn(chat["id"], s.get("model_id"), unattended=True, content=s["prompt"])
    return chat["id"], task_id


@app.get("/api/schedules")
async def api_list_schedules() -> list[dict]:
    return scheduler.list_schedules()


@app.post("/api/schedules", status_code=201)
async def api_create_schedule(req: ScheduleIn) -> dict:
    try:
        return scheduler.create(req.model_dump())
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.patch("/api/schedules/{schedule_id}")
async def api_update_schedule(schedule_id: str, req: ScheduleIn) -> dict:
    try:
        updated = scheduler.update(schedule_id, {k: v for k, v in req.model_dump().items() if k in req.model_fields_set})
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    if updated is None:
        raise HTTPException(404, "Schedule not found")
    return updated


@app.delete("/api/schedules/{schedule_id}", status_code=204, response_class=Response)
async def api_delete_schedule(schedule_id: str) -> Response:
    if not scheduler.delete(schedule_id):
        raise HTTPException(404, "Schedule not found")
    return Response(status_code=204)


@app.post("/api/schedules/{schedule_id}/run")
async def api_run_schedule(schedule_id: str) -> dict:
    s = scheduler.get(schedule_id)
    if s is None:
        raise HTTPException(404, "Schedule not found")
    chat_id, task_id = _start_scheduled_run(s)
    scheduler.mark_run(schedule_id, chat_id, task_id)
    return {"chat_id": chat_id, "task_id": task_id}


# ── Voice input (/api/transcribe) ──────────────────────────────────────

class TranscribeRequest(BaseModel):
    audio_base64: str
    mime: str | None = None


_AUDIO_SUFFIX = {"audio/webm": ".webm", "audio/ogg": ".ogg", "audio/mp4": ".mp4", "audio/mpeg": ".mp3",
                 "audio/wav": ".wav", "audio/x-wav": ".wav"}


@app.post("/api/transcribe")
async def api_transcribe(req: TranscribeRequest) -> dict:
    if not transcribe.available():
        raise HTTPException(503, "Voice input needs faster-whisper: pip install faster-whisper")
    try:
        audio = base64.b64decode(req.audio_base64.split(",", 1)[-1])
    except Exception:
        raise HTTPException(400, "audio_base64 is not valid base64")
    if len(audio) > 25 * 1024 * 1024:
        raise HTTPException(413, "Recording is too long")
    suffix = _AUDIO_SUFFIX.get((req.mime or "").split(";")[0], ".webm")
    try:
        text = await run_in_threadpool(transcribe.transcribe, audio, suffix, _whisper_model)
    except Exception as exc:
        logger.warning("Transcription failed: %s", exc)
        raise HTTPException(500, f"Transcription failed: {exc}")
    return {"text": text}


# ── Access token (/api/auth) ───────────────────────────────────────────

class LoginRequest(BaseModel):
    token: str


@app.get("/api/auth/status")
async def api_auth_status(request: Request) -> dict:
    return {"required": bool(_ACCESS_TOKEN), "authenticated": _authorized(request)}


@app.post("/api/auth/login")
async def api_login(req: LoginRequest, request: Request) -> Response:
    if not _ACCESS_TOKEN:
        return JSONResponse({"ok": True})
    if not _token_ok(req.token):
        time.sleep(0.5)  # slow down guessing
        raise HTTPException(401, "Wrong access token")
    resp = JSONResponse({"ok": True})
    resp.set_cookie(_AUTH_COOKIE, req.token, httponly=True, samesite="strict", max_age=60 * 60 * 24 * 30,
                    secure=request.url.scheme == "https")
    return resp


@app.post("/api/auth/logout")
async def api_logout() -> Response:
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(_AUTH_COOKIE)
    return resp


# ── OpenAI-compatible (/v1/*) ──────────────────────────────────────────

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    # A string, or OpenAI content parts ([{"type": "text", "text": ...}, ...]) — text parts are joined.
    content: str | list[dict] | None = ""

    def text(self) -> str:
        if isinstance(self.content, list):
            return "\n".join(str(p.get("text", "")) for p in self.content if p.get("type") in ("text", "input_text"))
        return self.content or ""


class ChatCompletionRequest(BaseModel):
    model: str = "local"
    messages: list[ChatMessage]
    max_tokens: int | None = Field(default=None)
    temperature: float | None = None
    top_p: float | None = None
    stream: bool = False


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Choice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage


class OAIModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "local"


class OAIModelListResponse(BaseModel):
    object: str = "list"
    data: list[OAIModelInfo]


class VisionRequest(BaseModel):
    image_base64: str
    prompt: str
    max_new_tokens: int = 128


@app.post("/v1/vision")
def vision_completion(req: VisionRequest) -> dict[str, str]:
    import base64
    from io import BytesIO
    from PIL import Image as PILImage
    llm = _require_llm()
    img_bytes = base64.b64decode(req.image_base64)
    image = PILImage.open(BytesIO(img_bytes)).convert("RGB")
    response = llm.generate_with_image(image, req.prompt, req.max_new_tokens)
    logger.info("Vision request complete | response=%s", response[:120])
    return {"response": response}


@app.get("/v1/models", response_model=OAIModelListResponse)
async def list_models() -> OAIModelListResponse:
    """Every model the app can use — the in-process one plus cloud and local-server models."""
    created = int(time.time())
    data: list[OAIModelInfo] = []
    if _llm is not None and _llm.is_loaded:
        data.append(OAIModelInfo(id=_llm.repo_id, created=created))
    if _registry is not None:
        for m in _registry.list_models():
            if m.provider == "local":
                continue  # the in-process model is listed above, by its repo id, once it has loaded
            data.append(OAIModelInfo(id=m.id, created=created, owned_by=m.provider))
    if not data:
        raise HTTPException(503, "No models are ready yet — try again in a moment")
    return OAIModelListResponse(data=data)


def _is_local_target(model: str) -> bool:
    """'local', an empty name, or the in-process model's repo id → the direct in-process path."""
    if model in ("", "local"):
        return True
    if _llm is not None and model == _llm.repo_id:
        return True
    reg_model = _model_or_none(model)
    return reg_model is not None and reg_model.provider == "local"


def _check_monthly_budget() -> None:
    if _monthly_budget is not None:
        spent = _month_spend()
        if spent >= _monthly_budget:
            raise HTTPException(402, f"Monthly spending limit reached (${spent:.2f} of ${_monthly_budget:.2f}). "
                                     "Raise it in Settings → Usage.")


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest) -> Any:
    messages = [{"role": m.role, "content": m.text()} for m in req.messages]
    if not _is_local_target(req.model):
        return await _registry_completion(req, messages)

    llm = _require_llm()
    if req.stream:
        return StreamingResponse(
            _sse_openai(req, messages, llm),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
        )

    completion = llm.generate(messages, max_new_tokens=req.max_tokens,
                              temperature=req.temperature, top_p=req.top_p)
    prompt_tokens = sum(llm.count_tokens(m["content"]) for m in messages)
    completion_tokens = llm.count_tokens(completion)

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=llm.repo_id,
        choices=[Choice(index=0, message=ChatMessage(role="assistant", content=completion))],
        usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens),
    )


def _sse_openai(req: ChatCompletionRequest, messages: list[dict], llm: LocalLLM):
    cid = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())
    for token in llm.stream(messages, max_new_tokens=req.max_tokens,
                            temperature=req.temperature, top_p=req.top_p):
        chunk = {"id": cid, "object": "chat.completion.chunk", "created": created,
                 "model": llm.repo_id,
                 "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}]}
        yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"


_STREAM_END = object()


def _registry_stream(model_id: str, messages: list[dict], options: dict | None):
    """Run a registry model's stream on its own thread (one meter, one thread) and yield its pieces.

    Yields text chunks, then a final dict: {"usage": meter snapshot} or {"error": message}.
    Usage is recorded as kind "api" so /v1 spend shows on the dashboard and counts toward the monthly limit.
    """
    import queue
    q: queue.Queue = queue.Queue()
    svc = _require_service()

    def work() -> None:
        with usage.attach(()), usage.meter() as m:
            try:
                for chunk in svc.stream(messages, model_id, call_type="api", options=options):
                    q.put(chunk)
            except Exception as exc:
                q.put({"error": str(exc)})
        if m.calls:
            try:
                task_store.record_usage(None, model_id, m.full_snapshot(), "api")
            except Exception:
                logger.exception("Could not record /v1 usage")
        q.put({"usage": m.snapshot()})
        q.put(_STREAM_END)

    threading.Thread(target=work, daemon=True, name="v1-stream").start()
    while (item := q.get()) is not _STREAM_END:
        yield item


async def _registry_completion(req: ChatCompletionRequest, messages: list[dict]) -> Any:
    model = _model_or_none(req.model)
    if model is None:
        raise HTTPException(404, f"Unknown model: {req.model}. GET /v1/models lists the available ones.")
    _check_monthly_budget()
    options = {"temperature": req.temperature} if req.temperature is not None and model.temperature else None
    cid, created = f"chatcmpl-{uuid.uuid4().hex[:8]}", int(time.time())

    if req.stream:
        def sse():
            for item in _registry_stream(model.id, messages, options):
                if isinstance(item, str):
                    delta = {"content": item}
                    chunk = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model.id,
                             "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
                elif "error" in item:
                    chunk = {"error": {"message": item["error"], "type": "provider_error"}}
                else:
                    chunk = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model.id,
                             "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(sse(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"})

    def collect() -> tuple[str, dict]:
        text, info = [], {}
        for item in _registry_stream(model.id, messages, options):
            if isinstance(item, str):
                text.append(item)
            else:
                info.update(item)
        return "".join(text), info

    text, info = await run_in_threadpool(collect)
    if "error" in info:
        raise HTTPException(502, info["error"])
    u = info.get("usage") or {}
    prompt_tokens, completion_tokens = int(u.get("input_tokens") or 0), int(u.get("output_tokens") or 0)
    return ChatCompletionResponse(
        id=cid, created=created, model=model.id,
        choices=[Choice(index=0, message=ChatMessage(role="assistant", content=text))],
        usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens),
    )


# ── Built frontend (single-process mode) ────────────────────────────────
# After `npm run build` in frontend/, the backend serves the app itself at http://<host>:8000 — no Vite
# dev server needed (e.g. to use it from a phone with ACCESS_TOKEN set). Unknown paths fall back to
# index.html so client-side routes work. Override the folder with server.frontend_dist in config.yaml.

_FRONTEND_DIST = Path(_SERVER_CFG.get("frontend_dist") or Path(__file__).resolve().parents[2] / "frontend" / "dist")


@app.get("/{path:path}", include_in_schema=False)
async def frontend(path: str) -> Any:
    if path.startswith(("api/", "v1/")) or path in ("api", "v1"):
        raise HTTPException(404, "Not found")
    index = _FRONTEND_DIST / "index.html"
    if not index.is_file():
        raise HTTPException(404, "Frontend not built. Run `npm run build` in frontend/, or use the dev server "
                                 "at http://localhost:5173.")
    root = _FRONTEND_DIST.resolve()
    target = (root / path).resolve()
    if path and target.is_file() and root in target.parents:
        # Vite fingerprints asset names, so they can be cached forever; index.html must not be.
        cache = "public, max-age=31536000, immutable" if "assets" in target.relative_to(root).parts else "no-cache"
        return FileResponse(target, headers={"Cache-Control": cache})
    return FileResponse(index, headers={"Cache-Control": "no-cache"})


# ── Entry point ────────────────────────────────────────────────────────

def run_server() -> None:
    cfg = load_config()
    host = cfg["server"].get("host", "127.0.0.1")
    if host not in ("127.0.0.1", "localhost", "::1") and not _ACCESS_TOKEN:
        logger.warning("Serving on %s without ACCESS_TOKEN: only requests addressed to localhost are accepted. "
                       "Set ACCESS_TOKEN in backend/.env to use the app from other devices.", host)
    uvicorn.run(app, host=host, port=cfg["server"].get("port", 8000), log_level="warning")
