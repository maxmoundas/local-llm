# Local LLM

A self-hosted, Claude-style chat app for running Google Gemma 4 locally, with Anthropic and OpenAI cloud models through a unified provider system, multi-step web research, and fast structured decisions from TypeSafe's Jev. The backend exposes an OpenAI-compatible API so it can drop into any project already using the OpenAI SDK.

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

### Research (multi-step answers)

For questions that need current or external information, the assistant researches before answering, showing one-line status updates as it goes ("Checking the weather in Nashville", "Searching news for “…”", "Reviewing what I found").

```
message ─► router ── no ──► answer directly
              │ yes
              ▼
        ┌─► planner (chat model, tool calling) ─► tools in parallel ─┐
        │                                                           │
        └──── not enough ◄── sufficiency check (Jev) ◄──────────────┘
                                   │ enough / max steps / planner stops
                                   ▼
                     answer streamed with research notes + sources
```

- **`src/agent/router.py`** -- `ResearchRouter.decide()` asks Jev whether the message needs fresh/external information (threshold 0.5; on a 24-prompt labeled set research prompts scored 0.82–0.98 and others 0.02–0.07). Falls back to a yes/no from the utility model. `enough()` asks Jev whether the notes so far fully answer the question (stop at ≥ 0.75).
- **`src/agent/runner.py`** -- `ResearchAgent.run()` yields `status` / `step` / `done` events. The planner is the user's selected chat model using native tool calling (Anthropic, OpenAI) or a prompted-JSON fallback (local). Up to 4 tool calls run in parallel per step; repeated calls are skipped.
- **`src/agent/tools.py`** -- free, keyless tools: `web_search` and `news_search` (DuckDuckGo via `ddgs`), `fetch_page` (readable text from a URL; refuses private/loopback hosts), `get_weather` (Open-Meteo). Each tool defines its own present- and past-tense status line.

Messages with images skip research. Research can be turned off, and its step limit changed, in **Settings → Research**.

### Jev (TypeSafe System One)

**`src/decision_service.py`** wraps [Jev](https://docs.typesafe.ai/introduction), a model that answers typed questions (`noul` yes/no probability, `choice`, `score`) about a state in a few hundred milliseconds instead of generating text. It is used for the app's quick decisions:

| Use | Jev question(s) | Fallback when Jev is off/unavailable |
|---|---|---|
| Research routing | needs fresh/external info? · notes sufficient? | utility-model yes/no · planner decides when to stop |
| Safety checks | jailbreak · harmful request · self-harm · denied topics (one noul each) + harm-severity score, thresholded by a strict/permissive policy | utility model with the editable JSON prompt |
| Memory gating | does this message hold a lasting fact about the user? (skip extraction below 0.4) | always run extraction |

Self-harm signals route to a supportive response (a system note with crisis resources) rather than a block. Call Jev from anywhere in the backend with `decision_service.get().ask(state, questions)`, or over HTTP with `POST /api/decide` (same request shape as TypeSafe's `/v1/systemone`). Every use can be switched off in **Settings → Jev**; set `TYPESAFE_API_KEY` in `backend/.env` to enable it.

### Frontend

React + TypeScript + Vite, styled after the Claude app (warm neutrals, serif responses, light and dark themes that follow the OS or can be forced in **Settings → General**). Vite proxies `/api`, `/health`, and `/v1` to the backend.

- `App.tsx` -- layout and top-level state. The new-chat screen creates the chat on first send. Switching models keeps the current chat; the last-used model is remembered.
- `ChatView.tsx` -- message list and SSE consumer (`lib/sse.ts`); follows the stream while you're at the bottom; **Stop** aborts the request and the partial reply is kept.
- `Message.tsx` / `Markdown.tsx` -- user bubbles, assistant prose (GFM tables, code blocks with copy).
- `ResearchSteps.tsx` -- the live status line while researching, collapsing to "Researched · N steps · M sources" with the step list and source chips.
- `Composer.tsx` / `ModelMenu.tsx` -- input card with image attach/paste/drop, model menu, send/stop.
- `Sidebar.tsx` -- collapsible; New chat, Search, Starred, Recents, per-chat menu (star, rename, delete).
- `SettingsView.tsx` -- General, Research, Jev, Safety, Memory, Models.

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
    { "role": "user",      "content": "..." },
    { "role": "assistant", "content": "...",
      "steps":   [{ "tool": "web_search", "label": "Searched the web for “…”", "summary": "6 results" }],
      "sources": [{ "title": "...", "url": "https://..." }],
      "stopped": true }
  ]
}
```

`steps`, `sources`, and `stopped` are display-only and stripped before history is sent to a model. `model_id` records the last model used. The title is generated by the utility model after the first exchange and can be renamed from the sidebar menu or by clicking the title.

### Streaming (SSE)

`POST /api/chats/{id}/message` returns `text/event-stream`. Each event is a JSON object on a `data:` line:

| Event type | Payload | Description |
|---|---|---|
| `status` | `{ "type": "status", "text": "..." }` | One-line research status (research only) |
| `step` | `{ "type": "step", "step": { "tool", "label", "summary" } }` | A finished tool call (research only) |
| `sources` | `{ "type": "sources", "sources": [{ "title", "url" }] }` | Sources gathered (research only) |
| `token` | `{ "type": "token", "content": "..." }` | One token from the model |
| `blocked` | `{ "type": "blocked", "target": "input" \| "output", "reason": "..." }` | Stopped by a safety check |
| `done` | `{ "type": "done", "name": "..." }` | Stream complete, carries the chat title |
| `error` | `{ "type": "error", "message": "..." }` | Generation failed (shown in the chat) |

The reply is saved when the stream completes. If the client disconnects (Stop), the partial reply is saved with `"stopped": true`. All blocking work — safety checks, routing, tools, generation — runs in a worker thread, never on the event loop.

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
│   │   ├── llm_service.py        # every LLM call: routing, logging, tool turns
│   │   ├── decision_service.py   # Jev client + DecisionService (typed decisions)
│   │   ├── guardrail_service.py  # safety checks (Jev battery, LLM fallback)
│   │   ├── agent/
│   │   │   ├── router.py         # research? / enough? decisions
│   │   │   ├── runner.py         # ResearchAgent loop (status/step/done events)
│   │   │   └── tools.py          # web_search, news_search, fetch_page, get_weather
│   │   ├── chat_store.py         # JSON-file chat persistence
│   │   ├── memory_store.py / settings_store.py
│   │   ├── config.py             # config.yaml loader + env overrides
│   │   ├── logger.py             # Rotating file + console logger
│   │   └── providers/
│   │       ├── base.py           # BaseProvider ABC, ModelOption, ToolCall/ToolTurn
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
│   │   ├── lib/                  # sse.ts (event parser), theme.ts (light/dark)
│   │   ├── components/           # ChatView, Message, Markdown, ResearchSteps, Composer,
│   │   │                         # ModelMenu, NewChat, Sidebar, SettingsView, Icons
│   │   ├── styles/global.css     # design tokens (light + dark) and all styles
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

# Optional: enables Jev for routing, safety checks, and memory gating
TYPESAFE_API_KEY=apikey_...
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
| `GET` / `PATCH` | `/api/settings` | Read / update settings (research, Jev, safety, memory, …) |
| `GET` / `PATCH` | `/api/memory` | Read / replace the memory profile |
| `POST` | `/api/decide` | Ask Jev typed questions: `{ "state": ..., "questions": {...} }` |

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
