# Local LLM

A self-hosted chat interface for running Google Gemma 4 locally, with support for Anthropic and OpenAI cloud models through a unified provider system. The backend exposes an OpenAI-compatible API so it can drop into any project already using the OpenAI SDK.

**Default model**: `google/gemma-4-E4B-it` (8B parameters, ~5 GB at 4-bit quantization, fits on an RTX 4070 SUPER with headroom).

---

## Architecture

```
Browser (React/Vite)
        |
        | HTTP + SSE
        v
FastAPI backend (:8000)
        |
   ProviderRegistry
   /       |       \
Local   Anthropic   OpenAI
(GPU)    (API key)  (API key)
```

### Backend

The backend is a FastAPI application (`backend/serve.py`) running on port 8000.

**`src/model.py`** -- `LocalLLM` class. Loads the model via HuggingFace Transformers with 4-bit NF4 quantization (bitsandbytes). Supports text-only and vision (image + text) inputs. Exposes `generate()` and `stream()` methods; vision variants use `generate_with_image()` and `stream_with_image()`.

**`src/providers/`** -- Provider abstraction layer.

- `BaseProvider` (ABC): defines `stream(messages, model_id, image_base64)` and `count_tokens(text)`.
- `LocalProvider`: wraps `LocalLLM`, always registered. Returns a descriptive error if a request arrives before the model finishes loading.
- `AnthropicProvider`: wraps the Anthropic SDK, registers Haiku 4.5 and Sonnet 4.6 if `ANTHROPIC_API_KEY` is set.
- `OpenAIProvider`: wraps the OpenAI SDK, registers GPT-5.4 Mini if `OPENAI_API_KEY` is set.
- `ProviderRegistry`: maps model IDs to their provider. `build_registry()` auto-discovers providers at startup based on available environment variables. The first registered model (local) is the default.

**`src/server.py`** -- FastAPI routes and streaming. On startup the provider registry is built immediately (cloud providers available at once) and the local model loads in a background thread — the server accepts requests within seconds regardless of how long the model takes to load. When a message is sent, the server resolves the provider for the requested `model_id`, calls `provider.stream()`, and yields each token as a Server-Sent Event. The full response and the model used are written to the chat store after streaming completes. Also exposes OpenAI-compatible `/v1/chat/completions` and `/v1/models` endpoints.

**`src/chat_store.py`** -- Chat persistence (see below).

**`src/config.py`** -- Loads `backend/config.yaml`. Any key can be overridden with environment variables (`MODEL_REPO_ID`, `MODEL_QUANTIZATION`, `SERVER_PORT`, `LOG_LEVEL`).

**`src/logger.py`** -- Rotating file logger (10 MB per file, 5 backups) with optional console output, written to `backend/logs/`.

### Frontend

React + TypeScript + Vite. Vite proxies `/api`, `/health`, and `/v1` to the backend, so there are no CORS issues and the frontend uses relative paths throughout.

On load, the app fetches `GET /api/models` to populate the model selector and polls `GET /health` every 3 seconds until `local_model_loaded` is true. While the local model is loading, a banner is shown and the local model option in the selector is disabled — cloud models are immediately usable. When the user sends a message, the frontend calls `POST /api/chats/{id}/message` and reads the response body as a `ReadableStream`, parsing SSE events line by line. Tokens are appended to the streaming message bubble in real time. A `done` event carries the auto-generated chat title; an `error` event (including API billing errors) surfaces the error message directly in the chat bubble.

Key components:

- `App.tsx` -- top-level state: chat list, current chat, model list, selected model, local model load status. Polls `/health` on startup.
- `ChatView.tsx` -- message list and SSE stream consumer. Notifies the parent of the stored model when a chat is loaded so the selector stays in sync.
- `InputBar.tsx` -- text input with image attachment support.
- `ModelSelector.tsx` -- dropdown to switch between available models (switching creates a new chat). Disables and labels the local model as `(loading...)` while it is not yet ready.
- `Sidebar.tsx` -- full-featured conversation list (see below).
- `api/client.ts` -- thin typed wrappers over `fetch`.

### Sidebar

The sidebar provides:

- **Search** — filters by chat name or preview text in real time.
- **Date grouping** — chats are bucketed into Today / Yesterday / Previous 7 Days / Older based on last activity.
- **Pin / star** — `☆`/`★` button on hover pins a chat to a permanent "Pinned" section above the date groups. The star stays visible on pinned chats.
- **Inline rename** — double-click any chat name to edit it in place; Enter commits, Escape cancels.
- **Delete on hover** — `✕` button appears alongside the pin button on hover.
- **Model badge** — small colored pill showing the last model used in each chat (Local, Haiku, Sonnet, GPT mini, etc.).

### Chat persistence

Each conversation is stored as a JSON file in `backend/chats/`, named by UUID (`{chat_id}.json`). There is no database. The schema per file:

```json
{
  "id": "uuid",
  "name": "Chat title",
  "created_at": "2026-01-01T00:00:00+00:00",
  "updated_at": "2026-01-01T00:00:00+00:00",
  "model_id": "claude-haiku-4-5-20251001",
  "pinned": false,
  "messages": [
    { "role": "system",    "content": "..." },
    { "role": "user",      "content": "..." },
    { "role": "assistant", "content": "..." }
  ]
}
```

`model_id` records the last model used in the chat and drives the sidebar badge and model selector sync when switching chats. `pinned` is set by the pin toggle in the sidebar.

The chat title is auto-generated from the first user message: the first sentence is used if it is 60 characters or fewer; otherwise the text is truncated at a word boundary with `…`. Titles can also be renamed inline in the sidebar (double-click) or in the chat header (click). Chats survive server restarts.

### Streaming (SSE)

`POST /api/chats/{id}/message` returns `text/event-stream`. Each event is a JSON object on a `data:` line:

| Event type | Payload | Description |
|---|---|---|
| `token` | `{ "type": "token", "content": "..." }` | One token from the model |
| `done` | `{ "type": "done", "name": "..." }` | Stream complete, carries the chat title |
| `error` | `{ "type": "error", "message": "..." }` | Generation failed (shown in the chat bubble) |

The response is written to the chat store only after the full stream completes (on `done`). If the stream errors, the partial response is discarded and the error message is displayed to the user.

### Multi-provider system

The model selector in the UI lists all available models. Providers register at startup based on which API keys are present in `backend/.env`. The local model is always available regardless of network or API keys.

| Provider | Models | Activation |
|---|---|---|
| Local (Gemma 4) | `google/gemma-4-E4B-it` | Always on (loads in background) |
| Anthropic | Claude Haiku 4.5, Claude Sonnet 4.6 | `ANTHROPIC_API_KEY` in `.env` |
| OpenAI | GPT-5.4 Mini | `OPENAI_API_KEY` in `.env` |

> **Expensive models disabled**: Claude Opus 4.7, GPT-5.5, and GPT-5.5 Pro are defined in the provider files but commented out. To re-enable any of them, uncomment the relevant `ModelOption` line and its corresponding test assertion.

All providers implement the same `BaseProvider.stream()` interface, so adding a new provider requires only a new file and three lines in `build_registry()`.

---

## Requirements

| Dependency | Version |
|---|---|
| Python | 3.12+ |
| Node.js | 18+ |
| CUDA | 12+ |
| VRAM | 8 GB minimum, 12 GB recommended |

Python packages are in `backend/requirements.txt`.

> **Note**: Gemma 4 support requires `transformers` from source until it ships in a stable release:
> ```
> pip install git+https://github.com/huggingface/transformers.git
> ```

---

## Project layout

```
local-llm/
├── backend/
│   ├── src/
│   │   ├── model.py              # LocalLLM: HF Transformers, 4-bit quant, streaming
│   │   ├── server.py             # FastAPI routes, SSE streaming, OpenAI endpoint
│   │   ├── chat_store.py         # JSON-file chat persistence
│   │   ├── config.py             # config.yaml loader + env overrides
│   │   ├── logger.py             # Rotating file + console logger
│   │   └── providers/
│   │       ├── base.py           # BaseProvider ABC, ModelOption dataclass
│   │       ├── local.py          # LocalProvider (wraps LocalLLM)
│   │       ├── anthropic_provider.py
│   │       ├── openai_provider.py
│   │       └── registry.py       # ProviderRegistry + build_registry()
│   ├── tests/                    # Unit tests (mock-based) + integration tests
│   ├── scripts/
│   │   └── download_model.py
│   ├── config.yaml
│   ├── .env.example
│   ├── requirements.txt
│   ├── pyproject.toml
│   └── serve.py                  # Entry point: python serve.py
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── api/client.ts
│   │   ├── components/
│   │   │   ├── ChatView.tsx
│   │   │   ├── Message.tsx
│   │   │   ├── ModelSelector.tsx
│   │   │   ├── InputBar.tsx
│   │   │   ├── Sidebar.tsx
│   │   │   └── EmptyState.tsx
│   │   ├── styles/global.css
│   │   └── types.ts
│   ├── package.json
│   └── vite.config.ts            # Proxies /api, /health, /v1 to :8000
├── models/                       # Downloaded model weights (gitignored)
└── README.md
```

---

## First-time setup

### 1. Install Python dependencies

```bash
cd backend
pip install -r requirements.txt
pip install git+https://github.com/huggingface/transformers.git
```

### 2. Configure environment

Copy the example and fill in your tokens:

```bash
cp backend/.env.example backend/.env
```

`backend/.env`:
```
# Required for local model download
HF_TOKEN=hf_your_token_here

# Optional: enables Anthropic models in the UI
ANTHROPIC_API_KEY=sk-ant-...

# Optional: enables OpenAI models in the UI
OPENAI_API_KEY=sk-...
```

You need a HuggingFace account with access granted to `google/gemma-4-E4B-it` at huggingface.co/google/gemma-4-E4B-it.

### 3. Download the model

```bash
cd backend
python scripts/download_model.py
```

Downloads ~5 GB to `models/` at the project root. Only runs once; subsequent starts load from disk.

### 4. Install frontend dependencies

```bash
cd frontend
npm install
```

---

## Running

```bash
# Terminal 1
cd backend && python serve.py

# Terminal 2
cd frontend && npm run dev
```

- Backend: http://localhost:8000
- Frontend: http://localhost:5173

The backend starts accepting requests immediately. The local model loads in the background (typically 1–3 minutes depending on disk speed). Cloud models are usable right away; a loading banner in the UI indicates when the local model is still warming up.

---

## Configuration

Edit `backend/config.yaml`:

```yaml
model:
  repo_id: "google/gemma-4-E4B-it"
  cache_dir: "../models"
  quantization: "4bit"       # none | 4bit | 8bit
  device_map: "auto"
  max_new_tokens: 512
  temperature: 0.7
  top_p: 0.9
  repetition_penalty: 1.1

server:
  host: "0.0.0.0"
  port: 8000
```

Environment variable overrides: `MODEL_REPO_ID`, `MODEL_QUANTIZATION`, `SERVER_PORT`, `LOG_LEVEL`.

---

## API reference

### Chat API (used by the UI)

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Server and model load status |
| `GET` | `/api/models` | List available models across all providers |
| `GET` | `/api/chats` | List all chats (sorted by last activity) |
| `POST` | `/api/chats` | Create a new chat |
| `GET` | `/api/chats/{id}` | Get chat with full message history |
| `POST` | `/api/chats/{id}/message` | Send a message; returns SSE stream |
| `PATCH` | `/api/chats/{id}` | Rename a chat |
| `DELETE` | `/api/chats/{id}` | Delete a chat |
| `POST` | `/api/chats/{id}/pin` | Toggle pin state; returns `{ "pinned": bool }` |

**`GET /health` response:**
```json
{ "status": "ok", "model": "google/gemma-4-E4B-it", "local_model_loaded": true }
```
`status` is `"loading"` and `local_model_loaded` is `false` while the local model is still loading into VRAM. Cloud provider endpoints are fully functional regardless.

**Message request body:**
```json
{
  "content": "Hello",
  "model_id": "claude-sonnet-4-6",
  "image_base64": null
}
```
`model_id` is optional; omitting it uses the local model. `image_base64` is a base64-encoded image for vision requests.

### OpenAI-compatible endpoint

Drop-in replacement for projects already using the OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="local")
response = client.chat.completions.create(
    model="google/gemma-4-E4B-it",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
)
for chunk in response:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

### Vision endpoint (non-streaming, local model only)

```python
import base64, requests

with open("photo.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

resp = requests.post("http://localhost:8000/v1/vision",
                     json={"image_base64": b64, "prompt": "Describe this image.", "max_new_tokens": 128})
print(resp.json()["response"])
```

---

## Adding a provider

1. Create `backend/src/providers/myprovider.py` subclassing `BaseProvider`. Implement `provider_name`, `models`, and `stream()`.
2. In `backend/src/providers/registry.py`, add to `build_registry()`:
   ```python
   my_key = os.getenv("MY_PROVIDER_API_KEY")
   if my_key:
       from src.providers.myprovider import MyProvider
       registry.register(MyProvider(my_key))
   ```
3. Add `MY_PROVIDER_API_KEY=` to `backend/.env`.

The new models appear automatically in `GET /api/models` and the UI model selector.

---

## Tests

```bash
cd backend
pytest                                          # unit tests (no GPU, no API keys)
pytest tests/test_providers_integration.py -m integration -v   # real API calls
```

Unit tests mock the model and all external API clients. Integration tests pass if a valid response is received, skip if the account has insufficient credits, and fail only on authentication errors.
