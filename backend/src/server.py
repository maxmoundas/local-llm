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
  POST /api/chats/{id}/message   ← SSE streaming response
  GET  /v1/models                ← OpenAI-compat
  POST /v1/chat/completions      ← OpenAI-compat
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Literal

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from src import chat_store, memory_store, settings_store
from src.config import load_config
from src.guardrail_service import GuardrailService
from src.llm_service import CALL_TYPE_CHAT, LLMService
from src.logger import get_logger, setup_logging
from src.model import LocalLLM
from src.providers import ProviderRegistry, build_registry
from src.settings_store import DEFAULT_GUARDRAIL_PROMPT, DEFAULT_SYSTEM_PROMPT

logger = get_logger(__name__)
_llm: LocalLLM | None = None
_registry: ProviderRegistry | None = None
_llm_service: LLMService | None = None
_system_prompt: str = DEFAULT_SYSTEM_PROMPT
_memory_enabled: bool = True
_guardrail_svc: GuardrailService | None = None
_guardrails_input: bool = True
_guardrails_output: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _llm, _registry, _llm_service, _system_prompt, _memory_enabled
    global _guardrail_svc, _guardrails_input, _guardrails_output
    cfg = load_config()
    setup_logging(cfg)
    logger.info("Starting backend")
    _llm = LocalLLM(cfg)
    _registry = build_registry(_llm)
    saved_settings = settings_store.load()
    _llm_service = LLMService(_registry, cheapest_model_id=saved_settings.get("cheapest_model_id"))
    _system_prompt = saved_settings.get("system_prompt") or DEFAULT_SYSTEM_PROMPT
    _memory_enabled = bool(saved_settings.get("memory_enabled", True))
    _guardrails_input = bool(saved_settings.get("guardrails_input", True))
    _guardrails_output = bool(saved_settings.get("guardrails_output", False))
    if saved_settings.get("guardrails_enabled", False):
        _guardrail_svc = GuardrailService(
            _llm_service,
            prompt=saved_settings.get("guardrails_prompt") or DEFAULT_GUARDRAIL_PROMPT,
            denied_topics=saved_settings.get("guardrails_topics") or [],
        )
    cheapest = _registry.cheapest_model
    logger.info("Server ready — cheapest model: %s (%s)", cheapest.name, cheapest.id)
    threading.Thread(target=_llm.load, daemon=True, name="model-loader").start()
    yield


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
    msgs = chat.get("messages", [])
    system = _system_prompt
    if _memory_enabled:
        mem = memory_store.load()
        if mem:
            system = f"{system}\n\n[User Memory]\n{mem}"
    if not msgs or msgs[0]["role"] != "system":
        return [{"role": "system", "content": system}] + msgs
    return msgs


def _maybe_update_memory(user_message: str, svc: LLMService) -> None:
    """Background task: extract durable facts from the user message and update memory if changed."""
    current = memory_store.load()
    updated = svc.extract_memory(user_message, current)
    if updated is not None:
        memory_store.save(updated)
        logger.info("Memory updated")


# ── Health & model discovery ───────────────────────────────────────────

@app.get("/health")
async def health() -> dict[str, Any]:
    loaded = _llm is not None and _llm.is_loaded
    return {
        "status": "ok" if loaded else "loading",
        "model": _llm.repo_id if _llm else None,
        "local_model_loaded": loaded,
    }


@app.get("/api/models")
async def api_list_models() -> list[dict]:
    registry = _require_registry()
    return [
        {"id": m.id, "name": m.name, "provider": m.provider, "is_cheapest": m.is_cheapest}
        for m in registry.list_models()
    ]


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


@app.delete("/api/chats/{chat_id}", status_code=204)
async def api_delete_chat(chat_id: str) -> None:
    if not chat_store.delete_chat(chat_id):
        raise HTTPException(404, "Chat not found")


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


@app.get("/api/settings")
async def api_get_settings() -> dict:
    registry = _require_registry()
    current = settings_store.load()
    return {
        **current,
        "default_system_prompt": DEFAULT_SYSTEM_PROMPT,
        "default_guardrail_prompt": DEFAULT_GUARDRAIL_PROMPT,
        "models": [
            {"id": m.id, "name": m.name, "provider": m.provider, "is_cheapest": m.is_cheapest}
            for m in registry.list_models()
        ],
    }


@app.patch("/api/settings")
async def api_update_settings(req: SettingsUpdate) -> dict:
    global _system_prompt, _memory_enabled, _guardrail_svc, _guardrails_input, _guardrails_output
    # Only save fields that were explicitly provided in the request body.
    updates = {k: v for k, v in req.model_dump().items() if k in req.model_fields_set}
    updated = settings_store.save(updates)
    if _llm_service is not None:
        _llm_service.set_cheapest_model(updated.get("cheapest_model_id"))
    _system_prompt = updated.get("system_prompt") or DEFAULT_SYSTEM_PROMPT
    _memory_enabled = bool(updated.get("memory_enabled", True))
    _guardrails_input = bool(updated.get("guardrails_input", True))
    _guardrails_output = bool(updated.get("guardrails_output", False))
    if updated.get("guardrails_enabled", False) and _llm_service is not None:
        _guardrail_svc = GuardrailService(
            _llm_service,
            prompt=updated.get("guardrails_prompt") or DEFAULT_GUARDRAIL_PROMPT,
            denied_topics=updated.get("guardrails_topics") or [],
        )
    else:
        _guardrail_svc = None
    logger.info("Settings updated: %s", updated)
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


_SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}


def _sse_blocked(target: str, reason: str | None):
    """Minimal SSE stream that emits a single 'blocked' event."""
    data: dict = {"type": "blocked", "target": target}
    if reason:
        data["reason"] = reason
    yield f"data: {json.dumps(data)}\n\n"


@app.post("/api/chats/{chat_id}/message")
async def api_send_message(
    chat_id: str, req: MessageRequest, background_tasks: BackgroundTasks
) -> Any:
    registry = _require_registry()
    svc = _require_service()
    chat = chat_store.get_chat(chat_id)
    if chat is None:
        raise HTTPException(404, "Chat not found")

    model_id = req.model_id or registry.default_model_id
    try:
        registry.get_provider(model_id)  # validate model exists
    except KeyError:
        raise HTTPException(400, f"Unknown model: {model_id}")

    # Input guardrail check — runs before appending so blocked messages aren't stored
    if _guardrail_svc is not None and _guardrails_input:
        guard = _guardrail_svc.check(req.content)
        if guard.blocked:
            logger.info("Input blocked by guardrail: %s", guard.reason)
            return StreamingResponse(
                _sse_blocked("input", guard.reason),
                media_type="text/event-stream",
                headers=_SSE_HEADERS,
            )

    # Capture before appending so we can detect the first exchange
    is_new_chat = chat["name"] == "New Chat"

    chat_store.append_messages(chat_id, "user", req.content, model_id=model_id)
    messages = _build_messages(chat_store.get_chat(chat_id))

    if _memory_enabled:
        background_tasks.add_task(_maybe_update_memory, req.content, svc)

    return StreamingResponse(
        _sse_generator(chat_id, messages, model_id, req.image_base64, svc, is_new_chat, req.content),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


def _sse_generator(
    chat_id: str,
    messages: list[dict],
    model_id: str,
    image_base64: str | None,
    svc: LLMService,
    is_new_chat: bool,
    first_user_content: str,
):
    full_response = ""
    try:
        for token in svc.stream(
            messages, model_id,
            image_base64=image_base64,
            call_type=CALL_TYPE_CHAT,
            context_id=chat_id,
        ):
            full_response += token
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
    except Exception as exc:
        # LLMService already logged the error; just surface it to the client.
        yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
        return

    # Output guardrail check — runs after full response is received
    if _guardrail_svc is not None and _guardrails_output:
        guard = _guardrail_svc.check(full_response)
        if guard.blocked:
            logger.info("Output blocked by guardrail: %s", guard.reason)
            data: dict = {"type": "blocked", "target": "output"}
            if guard.reason:
                data["reason"] = guard.reason
            yield f"data: {json.dumps(data)}\n\n"
            return  # do not store the blocked response

    chat_store.append_messages(chat_id, "assistant", full_response)

    # On the first exchange, generate a descriptive title via the cheapest model.
    if is_new_chat:
        title = svc.generate_title(first_user_content)
        chat_store.rename_chat(chat_id, title)

    updated = chat_store.get_chat(chat_id)
    yield f"data: {json.dumps({'type': 'done', 'name': updated['name']})}\n\n"


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
