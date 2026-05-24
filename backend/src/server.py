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
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from src import chat_store
from src.config import load_config
from src.logger import get_logger, setup_logging
from src.model import LocalLLM
from src.providers import BaseProvider, ProviderRegistry, build_registry

logger = get_logger(__name__)
_llm: LocalLLM | None = None
_registry: ProviderRegistry | None = None

DEFAULT_SYSTEM = (
    "You are a helpful, accurate, and thoughtful AI assistant. "
    "Be concise unless the user asks for detail."
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _llm, _registry
    cfg = load_config()
    setup_logging(cfg)
    logger.info("Starting backend")
    _llm = LocalLLM(cfg)
    _registry = build_registry(_llm)
    logger.info("Server ready — cloud providers available; local model loading in background")
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


def _build_messages(chat: dict) -> list[dict]:
    msgs = chat.get("messages", [])
    if not msgs or msgs[0]["role"] != "system":
        return [{"role": "system", "content": DEFAULT_SYSTEM}] + msgs
    return msgs


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
    return [{"id": m.id, "name": m.name, "provider": m.provider} for m in registry.list_models()]


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


_SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}


@app.post("/api/chats/{chat_id}/message")
async def api_send_message(chat_id: str, req: MessageRequest) -> Any:
    registry = _require_registry()
    chat = chat_store.get_chat(chat_id)
    if chat is None:
        raise HTTPException(404, "Chat not found")

    model_id = req.model_id or registry.default_model_id
    try:
        provider = registry.get_provider(model_id)
    except KeyError:
        raise HTTPException(400, f"Unknown model: {model_id}")

    chat_store.append_messages(chat_id, "user", req.content, model_id=model_id)
    messages = _build_messages(chat_store.get_chat(chat_id))

    logger.info("Chat %s | model=%s | prompt_tokens≈%d", chat_id[:8], model_id, provider.count_tokens(req.content))
    return StreamingResponse(
        _sse_generator(chat_id, messages, provider, model_id, req.image_base64),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


def _sse_generator(
    chat_id: str,
    messages: list[dict],
    provider: BaseProvider,
    model_id: str,
    image_base64: str | None = None,
):
    full_response = ""
    try:
        for token in provider.stream(messages, model_id, image_base64=image_base64):
            full_response += token
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
    except Exception as exc:
        logger.error("SSE stream error [%s]: %s", model_id, exc)
        yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
        return

    chat_store.append_messages(chat_id, "assistant", full_response)
    updated = chat_store.get_chat(chat_id)
    logger.info("Chat %s | done | model=%s | tokens≈%d", chat_id[:8], model_id, provider.count_tokens(full_response))
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
