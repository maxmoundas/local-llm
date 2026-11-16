# Architecture

How Local LLM is put together: the backend, the agent, the frontend, and how data is stored and streamed.
For setup, see the [README](../README.md). For HTTP endpoints, see [api.md](api.md).

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

## Backend

The backend is a FastAPI application (`backend/serve.py`) running on port 8000.

**`src/model.py`** -- `LocalLLM` class. Loads the model via HuggingFace Transformers with 4-bit NF4 quantization (bitsandbytes). Supports text-only and vision (image + text) inputs. Exposes `generate()` and `stream()` methods; vision variants use `generate_with_image()` and `stream_with_image()`.

**`src/providers/`** -- Provider abstraction layer.

- `BaseProvider` (ABC): `stream`, `complete`, and `complete_with_tools` (native tool calling where the provider has it, prompted JSON with lenient parsing and one repair retry otherwise).
- `LocalProvider`: wraps the in-process model (`model_factory.py` picks Transformers/CUDA or MLX/Apple Silicon). Returns a descriptive error if a request arrives before the model finishes loading.
- `AnthropicProvider` / `OpenAICompatibleProvider`: cloud APIs and any OpenAI-compatible server (Ollama, LM Studio, OpenRouter…). Both record real token usage and cache the system prompt where supported.
- `ProviderRegistry`: built from `models.yaml` by `build_registry()`; maps model IDs to providers and picks the cheapest ready model for utility calls and the default model for new chats. Automatic picks skip the in-process local model; it runs only when chosen explicitly, or when no other model is configured.

**`src/server.py`** -- FastAPI routes. On startup the registry is built (cloud and local-server models available at once), the in-process model loads in a background thread if `model.enabled` is true in `config.yaml` (off by default: it is never created, so it uses no GPU memory), and MCP connectors start. Each message runs as a background task (`src/tasks/`) whose events stream over SSE. Also exposes OpenAI-compatible `/v1/chat/completions` and `/v1/models` endpoints.

**`src/chat_store.py`** -- Chat persistence (see below).

**`src/config.py`** -- Loads `backend/config.yaml`. Any key can be overridden with environment variables (`MODEL_REPO_ID`, `MODEL_QUANTIZATION`, `SERVER_PORT`, `LOG_LEVEL`).

**`src/logger.py`** -- Rotating file logger (10 MB per file, 5 backups) with optional console output, written to `backend/logs/`.

## Agent (research, code, connectors)

When a message needs tools, the assistant works on it step by step as a **background task**: it plans, calls tools (several in parallel), reads the results, and repeats until done, then writes the answer. Images can be attached to such messages too: the planner can't see them, but they're copied into the workspace for code, and the vision model that writes the answer sees them. Each step shows as a one-line update ("Searching the web for “…”", "Running code: load sales.csv", "Reading nytimes.com") in an animated timeline. Tasks keep running if the tab closes; reopening the chat reattaches to the live task.

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
- **`src/agent/sandbox.py` + `workspace_tools.py`** — `run_code` (python, bash, and javascript where the sandbox has Node — the default `python:3.12-slim` image doesn't, so Docker offers python and bash), `write_file`, `read_file`, `list_files` in a per-task workspace. Code runs in a Docker container per task (CPU/memory/pid limits, no host env vars, network optional); files the agent creates appear as downloads under the answer. Without Docker, code tools are off unless you choose the unsandboxed "Local" mode in Settings. Workspaces are deleted with their chat and after 30 days (Settings → Agent → *Keep files the agent creates*); containers left behind by a crash are removed at startup.
- **Prompt-injection guard** — web pages and connector data (email, issues…) can carry instructions aimed at the model. Once a task has read any, tools that can send data out — `run_code` with internet access, and connector tools whose server doesn't mark them read-only — pause for your **Allow / Deny**, with the reason shown. Scheduled runs decline them. Toggle: Settings → Agent → *Ask before sending data out*. (Web searches and page fetches themselves are not gated.)
- **`src/agent/connectors.py`** — MCP servers from `backend/connectors.yaml` (see `connectors.example.yaml`) become tools for any model: email, calendar, GitHub, files… Tools matching a `confirm` pattern pause for your **Allow / Deny** before running.
- **`src/agent/router.py`** — decides per message whether tools are needed (Jev, or the utility model), and whether research notes are already sufficient.
- **`src/tasks/`** — `TaskManager` runs each turn on its own thread with an append-only event log (SQLite, `data/tasks.db`); `turn.py` is the per-message flow (guardrails → routing → agent → answer → title).

Settings → **Agent**: max steps, spending and time limits per task, a separate (cheap) model for agent steps, code execution mode, and connector status.

## Jev (TypeSafe System One)

**`src/decision_service.py`** wraps [Jev](https://docs.typesafe.ai/introduction), a model that answers typed questions (`noul` yes/no probability, `choice`, `score`) about a state in a few hundred milliseconds instead of generating text. It is used for the app's quick decisions:

| Use | Jev question(s) | Fallback when Jev is off/unavailable |
|---|---|---|
| Research routing | needs fresh/external info? · notes sufficient? | utility-model yes/no · planner decides when to stop |
| Safety checks | jailbreak · harmful request · self-harm · denied topics (one noul each) + harm-severity score, thresholded by a strict/permissive policy | utility model with the editable JSON prompt |
| Memory gating | does this message hold a lasting fact about the user? (skip extraction below 0.4) | always run extraction |

Self-harm signals route to a supportive response (a system note with crisis resources) rather than a block. Call Jev from anywhere in the backend with `decision_service.get().ask(state, questions)`, or over HTTP with `POST /api/decide` (same request shape as TypeSafe's `/v1/systemone`). Every use can be switched off in **Settings → Jev**; set `TYPESAFE_API_KEY` in `backend/.env` to enable it.

## Frontend

React + TypeScript + Vite, with a warm, minimal look (serif responses, light and dark themes that follow the OS or can be forced in **Settings → General**). Vite proxies `/api`, `/health`, and `/v1` to the backend.

- `App.tsx` -- layout and top-level state. The new-chat screen creates the chat on first send. Switching models keeps the current chat; the last-used model is remembered.
- `ChatView.tsx` -- message list and SSE consumer (`lib/sse.ts`); follows the stream while you're at the bottom; **Stop** aborts the request and the partial reply is kept.
- `Message.tsx` / `Markdown.tsx` -- user bubbles, assistant prose (GFM tables, code blocks with copy).
- `Activity.tsx` -- the live status line while the agent works, the step list and source chips, and the question / approval cards.
- `Composer.tsx` / `ModelMenu.tsx` -- input card with image attach/paste/drop, model menu, send/stop.
- `Sidebar.tsx` -- collapsible; New chat, full-text Search, Scheduled, Projects, Starred, Recents, per-chat menu (star, rename, generate title, move to project, export, delete).
- `ProjectView.tsx` / `SchedulesView.tsx` -- project page (instructions, files, memory, chats) and scheduled tasks.
- `PreviewPanel.tsx` -- side panel for HTML/SVG/Mermaid code and agent files (sandboxed iframes; no access to the app).
- `ChatSettingsMenu.tsx` -- per-chat system prompt, temperature, and reasoning effort.
- `SettingsView.tsx` / `UsageTab.tsx` -- General (incl. voice, network status, backup & restore), Usage, Agent, Jev, Safety, Memory, Models.

## Chat persistence

Each conversation is stored as a JSON file in `backend/chats/`, named by UUID (`{chat_id}.json`), with its attached files in `backend/chats/{chat_id}/files/` (original + extracted text + metadata). Projects live in `backend/projects/`, schedules in `backend/data/schedules.json`. The only databases are the task log (`data/tasks.db`, which also holds usage) and the search index (`chats/search.db`, rebuilt from the chat files if deleted). The schema per chat file:

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

## Streaming (SSE)

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

## Multi-provider system (models.yaml)

Every model comes from **`backend/models.yaml`** — no code changes to add one. Each entry records its provider, tool-calling mode (`native`, `prompted` JSON, or `auto` = try native, fall back), vision, context window (used to fit long chats), max output, price per 1M tokens (used for cost tracking, task budgets, and the monthly limit), and which per-chat options it accepts (`temperature`, `effort`).

| Provider type | Examples | Activation |
|---|---|---|
| `builtin` | Gemma 4 in-process (Transformers on CUDA, MLX on Apple Silicon) | `model.enabled: true` in `config.yaml` (off by default; loads in background) |
| `anthropic` | Claude Haiku 5.5, Claude Sonnet 5.5 | `ANTHROPIC_API_KEY` |
| `openai` | GPT-5.4 Mini | `OPENAI_API_KEY` |
| `openai_compatible` | **Ollama, LM Studio**, llama.cpp, vLLM, OpenRouter, Groq… | `base_url` (+ optional `api_key_env`); `discover: true` lists the server's models at startup |

New local model: `ollama pull qwen3:8b` and restart — it appears in the model menu. New cloud model: add a few lines to `models.yaml`, then run the contract suite (below) to confirm it works and get the recommended `tools:` setting.

Real token usage is read from every API response; the agent can use a cheaper model for its steps than for the final answer (Settings → Agent).

## Project layout

```
local-llm/
├── backend/
│   ├── src/
│   │   ├── server.py             # FastAPI routes, SSE, OpenAI-compatible /v1, serves the built UI
│   │   ├── llm_service.py        # every LLM call: routing, logging, tool turns
│   │   ├── model.py / model_mlx.py / model_factory.py   # in-process model (CUDA or Apple Silicon)
│   │   ├── decision_service.py   # Jev client + DecisionService (typed decisions)
│   │   ├── guardrail_service.py  # safety checks (Jev battery, LLM fallback)
│   │   ├── agent/
│   │   │   ├── router.py         # needs tools? / enough? decisions
│   │   │   ├── runner.py         # TaskAgent loop, approvals, untrusted-content guard
│   │   │   ├── tools.py          # web_search, news_search, fetch_page, get_weather
│   │   │   ├── sandbox.py        # Docker / local code execution, workspace cleanup
│   │   │   ├── workspace_tools.py  # run_code, read/write/list files
│   │   │   └── connectors.py     # MCP servers as tools
│   │   ├── tasks/                # background task runtime (manager, SQLite store, per-turn flow)
│   │   ├── chat_store.py         # JSON-file chat persistence
│   │   ├── search_index.py       # SQLite FTS5 index for search
│   │   ├── backup.py             # backup / restore zip
│   │   ├── project_store.py / memory_store.py / settings_store.py / scheduler.py
│   │   ├── attachments.py / context.py / usage.py / transcribe.py
│   │   ├── config.py / logger.py
│   │   └── providers/            # base, local, anthropic, openai(-compatible), registry
│   ├── tests/                    # unit tests, fakes, contract suite
│   ├── scripts/                  # download_model.py, probe_vision.py
│   ├── config.yaml / models.yaml / connectors.example.yaml / .env.example
│   └── serve.py                  # Entry point: python serve.py
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── api/client.ts
│   │   ├── lib/                  # sse.ts, turn.ts (event reducer), preview.ts, theme.ts
│   │   ├── components/           # ChatView, Message, Markdown, Activity, Composer, ModelMenu, Sidebar,
│   │   │                         # ProjectView, SchedulesView, SettingsView, UsageTab, PreviewPanel, …
│   │   ├── styles/global.css     # design tokens (light + dark) and all styles
│   │   └── types.ts
│   ├── package.json
│   └── vite.config.ts            # Proxies /api, /health, /v1 to :8000
├── start.sh / start.bat          # run both (start.sh --prod: one process)
├── docs/                         # architecture, API reference, screenshots
├── models/                       # Downloaded model weights (gitignored)
└── README.md
```

## Scheduled tasks

Settings live in the **Scheduled** page. A schedule runs while the backend is running: daily at a time, on chosen
weekdays, or every N minutes (min 5). Each run is an ordinary background turn posted to the schedule's own chat
(created on the first run, optionally inside a project). Nobody is watching, so the agent's questions go unanswered
and tool calls that need approval are declined. Runs missed while the backend was down run once at startup.

## Adding a provider

Most providers need no code: anything with an OpenAI-compatible API is an `openai_compatible` entry in `models.yaml`. For a genuinely different API, subclass `BaseProvider` (implement `models`, `stream`, `complete`, and optionally `_native_tool_turn`; prompted tool calling is inherited), record usage with `usage.record(...)`, and add a `type` branch in `build_provider()` in `src/providers/registry.py`.
