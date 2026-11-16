# API reference

The backend (default `http://localhost:8000`) serves the chat API used by the UI and an OpenAI-compatible API for your own scripts. When an `ACCESS_TOKEN` is set, send `Authorization: Bearer <token>` (see [Security](../README.md#security)).

## Chat API (used by the UI)

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
| `GET` | `/api/backup` | Download a backup zip (chats + files, projects, memory, settings, schedules; no API keys) |
| `POST` | `/api/restore?mode=merge\|replace` | Restore a backup zip sent as the raw body (see `src/backup.py`) |

**`GET /health` response:**
```json
{ "status": "ok", "model": "google/gemma-4-E4B-it", "local_model_loaded": true }
```
`status` is `"loading"` and `local_model_loaded` is `false` while the local model is still loading into VRAM. Cloud provider endpoints are fully functional regardless.

**Message request body:**
```json
{
  "content": "Hello",
  "model_id": "claude-sonnet-5-5",
  "attachments": ["<file id from POST /api/chats/{id}/files>"]
}
```
`model_id` is optional; omitting it uses the default model. Upload images and documents first with
`POST /api/chats/{id}/files` and pass their ids in `attachments` (the older single `image_base64` field still works).

## OpenAI-compatible endpoint

Drop-in replacement for projects already using the OpenAI SDK — and a single local gateway to every model in
`models.yaml`. `GET /v1/models` lists them; pass any id as `model` (`"local"` or the repo id means the in-process
model). Calls to cloud models are metered like chat turns: they show in Usage and count toward the monthly limit.

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

# Any other configured model works the same way:
client.chat.completions.create(model="claude-haiku-5-5", messages=[{"role": "user", "content": "Hi"}])
```

## Vision endpoint (non-streaming, local model only)

```python
import base64, requests

with open("photo.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

resp = requests.post("http://localhost:8000/v1/vision",
                     json={"image_base64": b64, "prompt": "Describe this image.", "max_new_tokens": 128})
print(resp.json()["response"])
```

## Streaming events

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
