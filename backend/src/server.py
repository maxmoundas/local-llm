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
  GET  /api/connectors
  GET/PATCH /api/settings
  GET/PATCH /api/memory
  POST /api/decide               ← Jev passthrough (structured decisions)
  GET  /v1/models                ← OpenAI-compat
  POST /v1/chat/completions      ← OpenAI-compat
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Literal

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.responses import FileResponse, Response, StreamingResponse

from src import chat_store, decision_service, memory_store, settings_store
from src.agent.connectors import ConnectorHub
from src.agent.router import ResearchRouter
from src.agent.runner import TaskAgent
from src.agent import sandbox as sb
from src.agent.sandbox import Sandbox, detect_sandbox, safe_path
from src.agent.tools import default_tools
from src.config import load_config
from src.decision_service import CALL_TYPE_MEMORY, DecisionError, DecisionService, noul
from src.guardrail_service import GuardrailService
from src.llm_service import LLMService
from src.logger import get_logger, setup_logging
from src.model_factory import create_local_llm
from src.providers import ProviderRegistry, build_registry
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

_MEMORY_THRESHOLD = 0.4
_HAS_DURABLE_FACT = noul(
    "The message reveals a lasting personal fact about the sender (their name, job, location, tools or languages "
    "they use, ongoing projects, stated preferences, or recurring constraints) worth remembering for future "
    "conversations.",
    true="It states something durable about who the sender is, what they work on, or how they like things done.",
    false="It is only a question, task, or one-off detail with nothing lasting about the sender.",
)


def _apply_settings(s: dict) -> None:
    """(Re)build every settings-driven service. Called at startup and after each PATCH."""
    global _system_prompt, _memory_enabled, _jev_memory
    global _guardrail_svc, _guardrails_input, _guardrails_output, _router, _agent, _sandbox, _sandbox_key

    if _llm_service is not None:
        _llm_service.set_cheapest_model(s.get("cheapest_model_id"))
    _system_prompt = s.get("system_prompt") or DEFAULT_SYSTEM_PROMPT
    _memory_enabled = bool(s.get("memory_enabled", True))
    _jev_memory = bool(s.get("jev_memory", True))

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
    yield
    _connectors.stop()


def _start_connectors() -> None:
    _connectors.start()
    if _connectors.names:
        _apply_settings(settings_store.load())  # let the router know about connected services


app = FastAPI(title="Local LLM API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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


def _build_messages(chat: dict) -> list[dict]:
    """Provider-ready history: system prompt first, display-only fields stripped.

    Empty turns (e.g. a reply stopped before any text) are dropped and consecutive
    same-role turns merged, since some providers require strict user/assistant alternation.
    """
    system = _system_prompt
    if _memory_enabled:
        mem = memory_store.load()
        if mem:
            system = f"{system}\n\n[User Memory]\n{mem}"
    out: list[dict] = [{"role": "system", "content": system}]
    for m in chat.get("messages", []):
        if m["role"] == "system" or not m.get("content"):
            continue
        if out[-1]["role"] == m["role"]:
            out[-1] = {"role": m["role"], "content": f"{out[-1]['content']}\n\n{m['content']}"}
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


def _maybe_update_memory(user_message: str, svc: LLMService) -> None:
    """Background task: extract durable facts from the user message and update memory if changed.

    When Jev is on for memory, it first checks whether the message holds anything worth
    remembering, which skips the LLM call for the large majority of messages.
    """
    if _jev_memory and _decision_svc is not None and _decision_svc.available:
        try:
            p = _decision_svc.ask({"message": user_message[:2000]}, {"durable": _HAS_DURABLE_FACT},
                                  call_type=CALL_TYPE_MEMORY)["durable"].noul
            if p < _MEMORY_THRESHOLD:
                return
        except DecisionError:
            pass  # no gate — let the LLM decide
    current = memory_store.load()
    updated = svc.extract_memory(user_message, current)
    if updated is not None:
        memory_store.save(updated)
        logger.info("Memory updated")


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

class RenameRequest(BaseModel):
    name: str


class MessageRequest(BaseModel):
    content: str
    image_base64: str | None = None
    model_id: str | None = None


@app.get("/api/chats")
async def api_list_chats() -> list[dict]:
    return chat_store.list_chats()


@app.post("/api/chats", status_code=201)
async def api_create_chat() -> dict:
    return chat_store.create_chat()


@app.get("/api/chats/{chat_id}")
async def api_get_chat(chat_id: str) -> dict:
    chat = chat_store.get_chat(chat_id)
    if chat is None:
        raise HTTPException(404, "Chat not found")
    return chat


@app.patch("/api/chats/{chat_id}")
async def api_rename_chat(chat_id: str, req: RenameRequest) -> dict:
    chat = chat_store.rename_chat(chat_id, req.name)
    if chat is None:
        raise HTTPException(404, "Chat not found")
    return chat


@app.delete("/api/chats/{chat_id}", status_code=204, response_class=Response)
async def api_delete_chat(chat_id: str) -> Response:
    if not chat_store.delete_chat(chat_id):
        raise HTTPException(404, "Chat not found")
    for t in task_store.for_chat(chat_id):
        _tasks.cancel(t["id"])
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
    jev_enabled: bool | None = None
    jev_model: str | None = None
    jev_routing: bool | None = None
    jev_guardrails: bool | None = None
    jev_memory: bool | None = None
    display_name: str | None = None


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
    memory_store.save(req.content)
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


def _turn_deps(svc: LLMService) -> TurnDeps:
    return TurnDeps(
        svc=svc,
        build_messages=_build_messages,
        guardrails=_guardrail_svc,
        guard_input=_guardrails_input,
        guard_output=_guardrails_output,
        update_memory=(lambda text: _maybe_update_memory(text, svc)) if _memory_enabled else None,
        router=_router,
        agent=_agent,
        sandbox=_sandbox,
        connectors=_connectors if _connectors.names else None,
        agent_model_id=_task_limits.get("agent_model_id"),
        budget_usd=_task_limits.get("budget_usd"),
        max_seconds=_task_limits.get("max_seconds"),
    )


@app.post("/api/chats/{chat_id}/message")
async def api_send_message(chat_id: str, req: MessageRequest) -> Any:
    """Start the turn as a background task and stream its events.

    The task keeps running if the client disconnects; reconnect with
    GET /api/tasks/{task_id}/events?after=<last seq>. Stop it with POST /api/tasks/{id}/cancel.
    """
    registry = _require_registry()
    svc = _require_service()
    if chat_store.get_chat(chat_id) is None:
        raise HTTPException(404, "Chat not found")
    if _tasks.active_for_chat(chat_id):
        raise HTTPException(409, "This chat is still working on the previous message")

    model_id = req.model_id or registry.default_model_id
    try:
        registry.get_provider(model_id)  # validate model exists
    except KeyError:
        raise HTTPException(400, f"Unknown model: {model_id}")

    deps = _turn_deps(svc)
    task_id = _tasks.start(chat_id, model_id,
                           lambda ctx: run_turn(ctx, chat_id, req.content, req.image_base64, model_id, deps))
    return StreamingResponse(_sse_stream(task_id, -1), media_type="text/event-stream",
                             headers={**_SSE_HEADERS, "X-Task-Id": task_id})


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


# ── OpenAI-compatible (/v1/*) ──────────────────────────────────────────

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


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
    llm = _require_llm()
    return OAIModelListResponse(data=[OAIModelInfo(id=llm.repo_id, created=int(time.time()))])


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest) -> Any:
    llm = _require_llm()
    messages = [m.model_dump() for m in req.messages]

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


# ── Entry point ────────────────────────────────────────────────────────

def run_server() -> None:
    cfg = load_config()
    uvicorn.run(app, host=cfg["server"].get("host", "0.0.0.0"),
                port=cfg["server"].get("port", 8000), log_level="warning")
