"""
FastAPI backend — serves the LLM inference API.

Endpoints
─────────
  GET  /health
  GET  /api/chats
  POST /api/chats
  GET  /api/chats/{id}
  PATCH /api/chats/{id}
  DELETE /api/chats/{id}
  POST /api/chats/{id}/message   ← SSE streaming response
  GET  /v1/models                ← OpenAI-compat (pipe into other projects)
  POST /v1/chat/completions      ← OpenAI-compat
"""
from __future__ import annotations

import base64
import json
import time
import uuid
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Any, Literal

from PIL import Image as PILImage

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from src import chat_store
from src.config import load_config
from src.logger import get_logger, setup_logging
from src.model import LocalLLM

logger = get_logger(__name__)
_llm: LocalLLM | None = None

DEFAULT_SYSTEM = (
    "You are a helpful, accurate, and thoughtful AI assistant. "
    "Be concise unless the user asks for detail."
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _llm
    cfg = load_config()
    setup_logging(cfg)
    logger.info("Starting backend")
    _llm = LocalLLM(cfg)
    _llm.load()
    logger.info("Model ready — backend listening")
    yield


app = FastAPI(title="Local LLM API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ────────────────────────────────────────────────────────────

def _require_loaded() -> LocalLLM:
    if _llm is None or not _llm.is_loaded:
        raise HTTPException(503, "Model not loaded yet — try again in a moment")
    return _llm


def _build_messages(chat: dict) -> list[dict]:
    msgs = chat.get("messages", [])
    if not msgs or msgs[0]["role"] != "system":
        return [{"role": "system", "content": DEFAULT_SYSTEM}] + msgs
    return msgs


# ── Health ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict[str, Any]:
    loaded = _llm is not None and _llm.is_loaded
    return {"status": "ok" if loaded else "loading", "model": _llm.repo_id if _llm else None}


# ── Chat management (/api/chats/*) ─────────────────────────────────────

class RenameRequest(BaseModel):
    name: str

class MessageRequest(BaseModel):
    content: str
    image_base64: str | None = None


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


_SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}


@app.post("/api/chats/{chat_id}/message")
async def api_send_message(chat_id: str, req: MessageRequest) -> Any:
    llm = _require_loaded()
    chat = chat_store.get_chat(chat_id)
    if chat is None:
        raise HTTPException(404, "Chat not found")

    chat_store.append_messages(chat_id, "user", req.content)
    messages = _build_messages(chat_store.get_chat(chat_id))

    if req.image_base64:
        logger.info("Chat %s | vision generate | prompt_tokens≈%d", chat_id[:8], llm.count_tokens(req.content))
        img_bytes = base64.b64decode(req.image_base64)
        image = PILImage.open(BytesIO(img_bytes)).convert("RGB")
        return StreamingResponse(
            _sse_generator_vision(chat_id, image, messages, llm),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )

    logger.info("Chat %s | generating | prompt_tokens≈%d", chat_id[:8], llm.count_tokens(req.content))
    return StreamingResponse(
        _sse_generator(chat_id, messages, llm),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


def _sse_generator_vision(chat_id: str, image: PILImage.Image, messages: list[dict], llm: LocalLLM):
    full_response = ""
    try:
        for token in llm.stream_with_image(image, messages):
            full_response += token
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
    except Exception as exc:
        logger.error("Vision SSE stream error: %s", exc)
        yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
        return

    chat_store.append_messages(chat_id, "assistant", full_response)
    updated = chat_store.get_chat(chat_id)
    logger.info("Vision chat %s | done | completion_tokens≈%d", chat_id[:8], llm.count_tokens(full_response))
    yield f"data: {json.dumps({'type': 'done', 'name': updated['name']})}\n\n"


def _sse_generator(chat_id: str, messages: list[dict], llm: LocalLLM):
    """
    Sync generator — Starlette's StreamingResponse runs this in a thread pool,
    so blocking queue.get() calls inside llm.stream() don't block the event loop.
    Each yielded chunk is flushed immediately to the client.
    """
    full_response = ""
    try:
        for token in llm.stream(messages):
            full_response += token
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
    except Exception as exc:
        logger.error("SSE stream error: %s", exc)
        yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
        return

    chat_store.append_messages(chat_id, "assistant", full_response)
    updated = chat_store.get_chat(chat_id)
    logger.info("Chat %s | done | completion_tokens≈%d", chat_id[:8], llm.count_tokens(full_response))
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

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "local"

class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


class VisionRequest(BaseModel):
    image_base64: str
    prompt: str
    max_new_tokens: int = 128


@app.post("/v1/vision")
def vision_completion(req: VisionRequest) -> dict[str, str]:
    """Single-turn vision inference: base64 image + prompt → raw model response text.

    FastAPI runs sync route handlers in a thread pool, so this does not block
    the event loop despite being a blocking GPU call.
    """
    llm = _require_loaded()
    img_bytes = base64.b64decode(req.image_base64)
    image = PILImage.open(BytesIO(img_bytes)).convert("RGB")
    response = llm.generate_with_image(image, req.prompt, req.max_new_tokens)
    logger.info("Vision request complete | response=%s", response[:120])
    return {"response": response}


@app.get("/v1/models", response_model=ModelListResponse)
async def list_models() -> ModelListResponse:
    llm = _require_loaded()
    return ModelListResponse(data=[ModelInfo(id=llm.repo_id, created=int(time.time()))])


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest) -> Any:
    llm = _require_loaded()
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
