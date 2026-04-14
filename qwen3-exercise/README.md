# Orion Chat — Claude.ai-style Chatbot

A full-stack, self-hosted AI chatbot with streaming responses, conversation history, multi-turn context, and Docker deployment.

---

## Project Structure

```
chatbot/
├── frontend/
│   └── index.html          ← Complete SPA (no build step needed)
├── backend/
│   ├── main.py             ← FastAPI: streaming SSE, static serving
│   ├── model.py            ← Model loader (warm service, 3 backends)
│   ├── inference.py        ← Pipeline: preprocess → stream → postprocess
│   └── requirements.txt
└── docker/
    ├── Dockerfile
    └── docker-compose.yml
```

---

## Quick Start (Docker — recommended)

### 1. Build and launch

```bash
cd docker
docker compose up --build
```

This starts:
- **Ollama** on port 11434 (local LLM server)
- **model-puller** (auto-pulls qwen3 on first run)
- **Orion** on port 8000 (API + frontend)

### 2. Open in browser

```
http://localhost:8000
```

### 3. Use a different model

```bash
OLLAMA_MODEL=llama3 docker compose up --build
```

---

## Quick Start (Local, no Docker)

### 1. Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3
```

### 2. Install backend deps

```bash
cd backend
pip install -r requirements.txt
```

### 3. Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Open

```
http://localhost:8000
```

---

## API Reference

### `POST /chat` — Streaming chat (SSE)

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Explain transformers in simple terms",
    "history": [],
    "max_tokens": 1024,
    "temperature": 0.7,
    "stream": true
  }'
```

SSE response:
```
data: {"token": "Transform"}
data: {"token": "ers"}
data: {"token": " are"}
...
data: [DONE]
```

### `POST /chat` — Non-streaming

```json
{ "stream": false, "message": "Hi" }
```

Response:
```json
{ "response": "Hello! How can I help?", "elapsed_ms": 430 }
```

### `GET /health`

```json
{ "status": "ok", "model_backend": "ollama" }
```

### `GET /config` / `POST /config`

Switch backend at runtime:
```bash
curl -X POST http://localhost:8000/config \
  -H "Content-Type: application/json" \
  -d '{"backend": "llamacpp"}'
```

---

## Switching Backends

| Backend | Description | Env vars needed |
|---|---|---|
| `ollama` | Ollama server (default) | `OLLAMA_URL`, `OLLAMA_MODEL` |
| `llamacpp` | Local GGUF file | `GGUF_PATH` |
| `transformers` | HuggingFace weights | `HF_MODEL_ID` |

---

## Features

| Feature | Details |
|---|---|
| Streaming responses | SSE token-by-token streaming |
| Multi-turn history | Last 20 turns sent to model |
| Conversation storage | LocalStorage — persists across refreshes |
| Multiple conversations | Create, switch, delete, export |
| Markdown rendering | Code blocks, tables, lists, bold, etc. |
| Copy / Regenerate | Per-message actions |
| Model selector | Switch backend from the UI |
| Health indicator | Live status dot in input bar |
| Mobile responsive | Collapsible sidebar |
| Dark theme | Refined editorial dark UI |

---

## GPU Support (NVIDIA)

Uncomment the `deploy` block in `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

Requires: `nvidia-docker2` and NVIDIA drivers installed on the host.
