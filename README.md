# Local LLM

A self-hosted, Claude-style chat app for running Google Gemma 4 locally, with Anthropic and OpenAI cloud models through a unified provider system, multi-step web research, and fast structured decisions from TypeSafe's Jev. The backend exposes an OpenAI-compatible API so it can drop into any project already using the OpenAI SDK.

**Default model**: `google/gemma-4-E4B-it` (8B parameters, ~5 GB at 4-bit quantization, fits on an RTX 4070 SUPER with headroom).

## Features at a glance

- **Chat** — streaming Markdown with math (KaTeX) and code highlighting; **regenerate** any reply and **edit-and-resend**
  any message, with ‹ 1 / 2 › switching between versions; export a chat as Markdown or JSON; "Generate title".
- **Attachments** — several images and documents per message (PDF, Word, Excel, CSV, text, code). Images are saved
  with the chat and stay visible to vision models in follow-ups; document text goes to the model, and the files are
  copied into the agent's workspace so code can work on them.
- **Long chats** — history is fitted to each model's context window; older turns are summarized by the utility model.
- **Projects** — groups of chats sharing instructions, knowledge files, and their own memory.
- **Per-chat settings** — a custom system prompt, temperature (models that accept one), and reasoning effort (Claude, GPT-5).
- **Search** — full text over every message, with snippets; jumps to the matching message.
- **Previews** — HTML, SVG, and Mermaid code blocks open in a side panel; files the agent creates (images, CSV as a
  table, text, Markdown, HTML) preview inline instead of only downloading.
- **Scheduled tasks** — prompts that run daily, on chosen weekdays, or every N minutes, each posting to its own chat.
- **Usage** — spend by day, model, and chat, plus an optional monthly limit that pauses new turns.
- **Voice input** — local speech-to-text with Whisper (`faster-whisper`), no cloud calls.
- **Keyboard shortcuts** — ⌘/Ctrl+Shift+O new chat, ⌘K search, ⌘B sidebar, Shift+Esc focus input, ⌘/ for the list.
- **Safe by default on the network** — loopback-only unless you set an access token (see [Security](#security)).

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

- `BaseProvider` (ABC): `stream`, `complete`, and `complete_with_tools` (native tool calling where the provider has it, prompted JSON with lenient parsing and one repair retry otherwise).
- `LocalProvider`: wraps the in-process model (`model_factory.py` picks Transformers/CUDA or MLX/Apple Silicon). Returns a descriptive error if a request arrives before the model finishes loading.
- `AnthropicProvider` / `OpenAICompatibleProvider`: cloud APIs and any OpenAI-compatible server (Ollama, LM Studio, OpenRouter…). Both record real token usage and cache the system prompt where supported.
- `ProviderRegistry`: built from `models.yaml` by `build_registry()`; maps model IDs to providers and picks the cheapest ready model for utility calls.

**`src/server.py`** -- FastAPI routes. On startup the registry is built (cloud and local-server models available at once), the in-process model loads in a background thread, and MCP connectors start. Each message runs as a background task (`src/tasks/`) whose events stream over SSE. Also exposes OpenAI-compatible `/v1/chat/completions` and `/v1/models` endpoints.

**`src/chat_store.py`** -- Chat persistence (see below).

**`src/config.py`** -- Loads `backend/config.yaml`. Any key can be overridden with environment variables (`MODEL_REPO_ID`, `MODEL_QUANTIZATION`, `SERVER_PORT`, `LOG_LEVEL`).

**`src/logger.py`** -- Rotating file logger (10 MB per file, 5 backups) with optional console output, written to `backend/logs/`.

### Agent (research, code, connectors)

When a message needs tools, the assistant works on it step by step as a **background task**: it plans, calls tools (several in parallel), reads the results, and repeats until done, then writes the answer. Each step shows as a one-line update ("Searching the web for “…”", "Running code: load sales.csv", "Reading nytimes.com") in an animated timeline. Tasks keep running if the tab closes; reopening the chat reattaches to the live task.

```
message ─► router ── no ──► answer directly
              │ yes
              ▼
        ┌─► planner (tool calls, any model) ─► tools in parallel ─► work log ─┐
        │                                                                     │
        └── continue ◄── limits: steps · $ budget · time · cancel ◄──────────┘
                                   │ done / limit hit
                                   ▼
                     answer streamed from the work log (+ sources, files)
```

- **`src/agent/runner.py`** — `TaskAgent`. Stateless turns over a rendered work log, so every model (native or prompted tool calling) runs the same loop. Every tool gets an optional `status` argument the model fills with the user-facing line (no extra LLM calls); templates are the fallback. Internal tools: `update_plan` (a private plan) and `ask_user` (pauses for your answer). Old log entries are trimmed, then folded into a summary by the cheapest model when the log outgrows the model's context. Stops on step, spending, and time limits, or after repeated failures.
- **`src/agent/tools.py`** — free, keyless research tools: `web_search`, `news_search`, `fetch_page`, `get_weather`.
- **`src/agent/sandbox.py` + `workspace_tools.py`** — `run_code` (python/bash/node), `write_file`, `read_file`, `list_files` in a per-task workspace. Code runs in a Docker container per task (`python:3.12-slim`, CPU/memory/pid limits, no host env vars, network optional); files the agent creates appear as downloads under the answer. Without Docker, code tools are off unless you choose the unsandboxed "Local" mode in Settings.
- **`src/agent/connectors.py`** — MCP servers from `backend/connectors.yaml` (see `connectors.example.yaml`) become tools for any model: email, calendar, GitHub, files… Tools matching a `confirm` pattern pause for your **Allow / Deny** before running.
- **`src/agent/router.py`** — decides per message whether tools are needed (Jev, or the utility model), and whether research notes are already sufficient.
- **`src/tasks/`** — `TaskManager` runs each turn on its own thread with an append-only event log (SQLite, `data/tasks.db`); `turn.py` is the per-message flow (guardrails → routing → agent → answer → title).

Settings → **Agent**: max steps, spending and time limits per task, a separate (cheap) model for agent steps, code execution mode, and connector status.

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
- `Sidebar.tsx` -- collapsible; New chat, full-text Search, Scheduled, Projects, Starred, Recents, per-chat menu (star, rename, generate title, move to project, export, delete).
- `ProjectView.tsx` / `SchedulesView.tsx` -- project page (instructions, files, memory, chats) and scheduled tasks.
- `PreviewPanel.tsx` -- side panel for HTML/SVG/Mermaid code and agent files (sandboxed iframes; no access to the app).
- `ChatSettingsMenu.tsx` -- per-chat system prompt, temperature, and reasoning effort.
- `SettingsView.tsx` / `UsageTab.tsx` -- General (incl. voice, network status), Usage, Agent, Jev, Safety, Memory, Models.

### Chat persistence

Each conversation is stored as a JSON file in `backend/chats/`, named by UUID (`{chat_id}.json`), with its attached files in `backend/chats/{chat_id}/files/` (original + extracted text + metadata). Projects live in `backend/projects/`, schedules in `backend/data/schedules.json`. There is no database besides the task log. The schema per chat file:

```json
{
  "id": "uuid",
  "name": "Chat title",
  "created_at": "2026-01-01T00:00:00+00:00",
  "updated_at": "2026-01-01T00:00:00+00:00",
  "model_id": "claude-haiku-4-5-20251001",
  "pinned": false,
  "project_id": "uuid (optional)",
  "settings": { "system_prompt": "...", "temperature": 0.7, "effort": "low" },
  "context_summary": { "upto": 40, "fp": "…", "text": "summary of messages[:40]" },
  "messages": [
    { "role": "user",      "content": "...",
      "attachments": [{ "id": "…", "name": "report.pdf", "mime": "application/pdf", "size": 12345, "kind": "document" }] },
    { "role": "assistant", "content": "...",
      "steps":   [{ "tool": "web_search", "label": "Searched the web for “…”", "summary": "6 results" }],
      "sources": [{ "title": "...", "url": "https://..." }],
      "stopped": true }
  ]
}
```

`steps`, `sources`, and `stopped` are display-only and stripped before history is sent to a model.

**Versions.** `messages` is always the path being shown. A message with alternatives (from regenerate or edit) carries
`versions` — the *other* conversation tails starting at that position — and `version_index`, where the shown one sits.
Tails nest, so the full tree is stored as its visible path. `POST /api/chats/{id}/versions` switches between them. `model_id` records the last model used. The title is generated by the utility model after the first exchange and can be renamed from the sidebar menu or by clicking the title.

### Streaming (SSE)

`POST /api/chats/{id}/message` starts the turn as a background task (id in the `X-Task-Id` header) and streams its events. Each event is `id: <seq>` + a JSON `data:` line. Disconnecting does **not** stop the task: resume with `GET /api/tasks/{id}/events?after=<last seq>`, find a chat's running task with `GET /api/chats/{id}/active`, stop it with `POST /api/tasks/{id}/cancel`.

| Event type | Description |
|---|---|
| `task_start` | `{ task_id }` — first event |
| `status` | one-line status between steps ("Thinking", "Writing the answer") |
| `action` | `{ action: { id, tool, kind, label, status, summary, started, ended } }` — a tool call; sent when it starts (`running`/`waiting`) and again when it ends (`done`/`failed`/`denied`). Upsert by `id`. |
| `sources` | `{ sources: [{ title, url }] }` |
| `input_request` / `input_resolved` | the agent is waiting on a question or approval — answer with `POST /api/tasks/{id}/reply` |
| `token` | `{ content }` — answer text (not persisted; a reconnect to a live task replays them) |
| `blocked` / `error` | stopped by a safety check / something failed |
| `done` | `{ name, files }` — chat title and files created |
| `usage` | tokens and `cost_usd` for the whole turn |
| `task_end` | `{ status: done \| failed \| cancelled \| interrupted }` — last event |

The frontend rebuilds a turn by folding these events (`frontend/src/lib/turn.ts`), so live view, reconnects, and reloads all show the same thing.

### Multi-provider system (models.yaml)

Every model comes from **`backend/models.yaml`** — no code changes to add one. Each entry records its provider, tool-calling mode (`native`, `prompted` JSON, or `auto` = try native, fall back), vision, context window (used to fit long chats), max output, price per 1M tokens (used for cost tracking, task budgets, and the monthly limit), and which per-chat options it accepts (`temperature`, `effort`).

| Provider type | Examples | Activation |
|---|---|---|
| `builtin` | Gemma 4 in-process (Transformers on CUDA, MLX on Apple Silicon) | always (loads in background) |
| `anthropic` | Claude Haiku 5.5, Claude Sonnet 5.5 | `ANTHROPIC_API_KEY` |
| `openai` | GPT-5.4 Mini | `OPENAI_API_KEY` |
| `openai_compatible` | **Ollama, LM Studio**, llama.cpp, vLLM, OpenRouter, Groq… | `base_url` (+ optional `api_key_env`); `discover: true` lists the server's models at startup |

New local model: `ollama pull qwen3:8b` and restart — it appears in the model menu. New cloud model: add a few lines to `models.yaml`, then run the contract suite (below) to confirm it works and get the recommended `tools:` setting.

Real token usage is read from every API response; the agent can use a cheaper model for its steps than for the final answer (Settings → Agent).

---

## Requirements

| Dependency | Version |
|---|---|
| Python | 3.12+ |
| Node.js | 18+ |
| CUDA | 12+ |
| VRAM | 8 GB minimum, 12 GB recommended |

Python packages are in `backend/requirements.txt`. Attachments use `pypdf`, `python-docx`, and `openpyxl`; voice
input uses `faster-whisper` (optional — without it the microphone button is hidden; the first use downloads the
Whisper model chosen in Settings → General).

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
  host: "127.0.0.1"      # "0.0.0.0" to serve your network — also set ACCESS_TOKEN (see Security)
  port: 8000
```

Environment variable overrides: `MODEL_REPO_ID`, `MODEL_QUANTIZATION`, `SERVER_HOST`, `SERVER_PORT`, `LOG_LEVEL`, `ACCESS_TOKEN`, `WHISPER_MODEL`.

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
| `POST` | `/api/chats/{id}/message` | Send a message (`attachments`: file ids); returns SSE stream |
| `POST` | `/api/chats/{id}/regenerate` | New version of the reply at `index` (or answer the last message); SSE |
| `POST` | `/api/chats/{id}/edit` | Replace the user message at `index` and answer it; SSE |
| `POST` | `/api/chats/{id}/versions` | Show another version: `{ index, version }` |
| `POST` | `/api/chats/{id}/files` | Upload an attachment `{ name, mime, data_base64 }` → metadata with `id` |
| `GET` | `/api/chats/{id}/files/{file_id}` | Download an attachment |
| `POST` | `/api/chats/{id}/title` | Generate a title from the conversation |
| `GET` | `/api/chats/{id}/export?format=md\|json` | Export |
| `GET` | `/api/search?q=` | Full-text search over all chats |
| `PATCH` | `/api/chats/{id}` | Rename, move to a project (`project_id`), or set per-chat `settings` |
| `DELETE` | `/api/chats/{id}` | Delete a chat |
| `POST` | `/api/chats/{id}/pin` | Toggle pin state; returns `{ "pinned": bool }` |
| `GET` / `PATCH` | `/api/settings` | Read / update settings (research, Jev, safety, memory, …) |
| `GET` / `PATCH` | `/api/memory` | Read / replace the memory profile |
| `POST` | `/api/decide` | Ask Jev typed questions: `{ "state": ..., "questions": {...} }` |
| `GET` / `POST` | `/api/projects` | List / create projects |
| `GET` / `PATCH` / `DELETE` | `/api/projects/{id}` | Project detail (with chats and memory) / update / delete (chats are kept) |
| `POST` / `DELETE` | `/api/projects/{id}/files[/{file_id}]` | Add / remove knowledge files |
| `PATCH` | `/api/projects/{id}/memory` | Replace the project's memory |
| `GET` / `POST` | `/api/schedules` | List / create scheduled tasks |
| `PATCH` / `DELETE` | `/api/schedules/{id}` | Update / delete; `POST /api/schedules/{id}/run` runs now |
| `GET` | `/api/usage?days=30` | Spend and tokens by day, model, and chat; month to date |
| `POST` | `/api/transcribe` | Speech to text: `{ audio_base64, mime }` → `{ text }` |
| `GET` / `POST` | `/api/auth/status`, `/api/auth/login`, `/api/auth/logout` | Access-token sign-in (see Security) |

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

## Security

The backend listens on **127.0.0.1** by default, so only this computer can reach it. Requests must also address
it by a loopback name (`localhost`, `127.0.0.1`), which stops web pages from reaching it through DNS rebinding, and
state-changing requests from other web origins are refused (allowed origins: `server.cors_origins` in `config.yaml`).

To use the app from other devices:

1. Set a long random `ACCESS_TOKEN` in `backend/.env`.
2. Set `server.host: "0.0.0.0"` in `backend/config.yaml` and restart.

The browser then asks for the token once (kept in an HttpOnly, SameSite=Strict cookie). API clients send
`Authorization: Bearer <token>` — including the OpenAI SDK: `OpenAI(base_url=..., api_key="<token>")`.
Without a token, a non-loopback `host` still only accepts requests addressed to localhost.

HTML previews run in sandboxed iframes without same-origin access, so previewed code can't read the app or call its API.

---

## Scheduled tasks

Settings live in the **Scheduled** page. A schedule runs while the backend is running: daily at a time, on chosen
weekdays, or every N minutes (min 5). Each run is an ordinary background turn posted to the schedule's own chat
(created on the first run, optionally inside a project). Nobody is watching, so the agent's questions go unanswered
and tool calls that need approval are declined. Runs missed while the backend was down run once at startup.

---

## Adding a provider

Most providers need no code: anything with an OpenAI-compatible API is an `openai_compatible` entry in `models.yaml`. For a genuinely different API, subclass `BaseProvider` (implement `models`, `stream`, `complete`, and optionally `_native_tool_turn`; prompted tool calling is inherited), record usage with `usage.record(...)`, and add a `type` branch in `build_provider()` in `src/providers/registry.py`.

---

## Tests

```bash
cd backend
pytest                                                   # unit + task-runtime tests (no GPU, no API keys)
pytest tests/test_providers_integration.py -m integration -v   # real API calls
pytest tests/contract -m contract --model claude-haiku-5-5 --model llama3.2:3b -v   # does a model work here?

cd frontend && npm test                                  # turn reducer (vitest)
```

Unit tests mock the model and all external API clients; `tests/fakes.py` has a scripted provider that runs the real agent loop deterministically. The **contract suite** checks a model against what the agent needs (instruction following, streaming + usage, tool calls, status lines, parallel calls, knowing when to stop, a full agent loop) in both native and prompted modes, and prints the recommended `tools:` setting. CI (`.github/workflows/tests.yml`) runs the backend unit tests and the frontend typecheck + tests on every push.
