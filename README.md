# Local LLM

Run Google Gemma 4 locally with a streaming chat interface and an OpenAI-compatible API endpoint for integration into other projects.

**Model**: `google/gemma-4-E4B-it` — 8B parameters, ~5 GB at 4-bit quantization. Fits entirely on an RTX 4070 SUPER (12 GB VRAM) with headroom.

---

## Requirements

| Dependency | Version |
|---|---|
| Python | 3.12+ |
| Node.js | 18+ |
| CUDA | 12+ (RTX 4070 SUPER tested) |
| VRAM | 8 GB minimum, 12 GB recommended |

Python packages are listed in `backend/requirements.txt`.
> **Important**: `transformers` must be installed from the GitHub source (Gemma 4 support is not yet in a stable release):
> ```
> pip install git+https://github.com/huggingface/transformers.git
> ```

---

## Project layout

```
local-llm/
├── backend/               # FastAPI server + model inference
│   ├── src/
│   │   ├── model.py       # Gemma 4 loader, 4-bit quant, streaming
│   │   ├── server.py      # FastAPI app, SSE streaming, OpenAI endpoint
│   │   ├── chat_store.py  # JSON-file chat persistence
│   │   ├── config.py      # Config loader (config.yaml + env overrides)
│   │   └── logger.py      # Rotating file + console logger
│   ├── tests/             # 46 unit/integration tests (pytest)
│   ├── scripts/
│   │   └── download_model.py
│   ├── config.yaml        # Model and server settings
│   ├── requirements.txt
│   ├── pyproject.toml
│   └── serve.py           # Entry point: `python serve.py`
├── frontend/              # React + Vite + TypeScript UI
│   ├── src/
│   │   ├── App.tsx
│   │   ├── api/client.ts
│   │   ├── components/
│   │   │   ├── ChatView.tsx
│   │   │   ├── Message.tsx
│   │   │   ├── InputBar.tsx
│   │   │   ├── Sidebar.tsx
│   │   │   └── EmptyState.tsx
│   │   ├── styles/global.css
│   │   └── types.ts
│   ├── package.json
│   └── vite.config.ts     # Proxies /api → localhost:8000
├── models/                # Downloaded model weights (gitignored)
├── start.bat              # Windows: launch backend + frontend
├── start.sh               # macOS/Linux: launch backend + frontend
└── README.md
```

---

## First-time setup

### 1. Clone and enter the repo

```bash
git clone <repo-url>
cd local-llm
```

### 2. Install Python dependencies

```bash
cd backend
pip install -r requirements.txt
pip install git+https://github.com/huggingface/transformers.git
```

### 3. Set your Hugging Face token

Create `backend/.env` (or set the environment variable directly):

```
HF_TOKEN=hf_your_token_here
```

You need a HuggingFace account with access granted to `google/gemma-4-E4B-it` at huggingface.co/google/gemma-4-E4B-it.

### 4. Download the model

```bash
cd backend
python scripts/download_model.py
```

This downloads ~5 GB to `models/` at the project root. Takes a few minutes on first run; subsequent starts load from cache.

### 5. Install frontend dependencies

```bash
cd frontend
npm install
```

---

## Running

From the project root, use the launch script for your platform:

**Windows**
```bat
start.bat
```

**macOS / Linux**
```bash
chmod +x start.sh
./start.sh
```

The script starts:
- Backend on **http://localhost:8000**
- Frontend dev server on **http://localhost:5173**

Open **http://localhost:5173** in your browser.

To start services manually:
```bash
# Terminal 1 — backend
cd backend && python serve.py

# Terminal 2 — frontend
cd frontend && npm run dev
```

---

## Configuration

Edit `backend/config.yaml` to tune model behavior:

```yaml
model:
  repo_id: "google/gemma-4-E4B-it"
  cache_dir: "../models"         # relative to backend/
  quantization: "4bit"           # none | 4bit | 8bit
  device_map: "auto"
  max_new_tokens: 512
  temperature: 0.7
  top_p: 0.9
  repetition_penalty: 1.1

server:
  host: "0.0.0.0"
  port: 8000
```

Environment variables override config values: `MODEL_REPO_ID`, `QUANTIZATION`, `SERVER_PORT`.

---

## API

The backend exposes two interfaces.

### Chat API (used by the UI)

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/chats` | List all chats |
| `POST` | `/api/chats` | Create a new chat |
| `GET` | `/api/chats/{id}` | Get chat + message history |
| `POST` | `/api/chats/{id}/messages` | Send a message (SSE stream) |
| `PATCH` | `/api/chats/{id}` | Rename a chat |
| `DELETE` | `/api/chats/{id}` | Delete a chat |

The `/messages` endpoint streams tokens as Server-Sent Events:
```
data: {"type": "token", "content": "Hello"}
data: {"type": "done", "name": "Chat title"}
```

### OpenAI-compatible endpoint

Drop-in replacement for projects already using the OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="local")
response = client.chat.completions.create(
    model="gemma-4-E4B-it",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
)
for chunk in response:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

---

## Tests

```bash
cd backend
pytest
```

All 46 tests run against mocked model/processor — no GPU required.

---

## Chat persistence

Conversations are saved as JSON files in `backend/chats/`. Chats survive server restarts. Delete the directory to clear all history.

Logs rotate at 10 MB (5 backups) in `backend/logs/`.
