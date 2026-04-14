# ── Orion Chatbot — Dockerfile ────────────────────────────────────────────────
FROM python:3.11-slim

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python deps (cached layer) ───────────────────────────────────────
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy backend ──────────────────────────────────────────────────────────────
COPY backend/ ./backend/

# ── Copy frontend (served as static files by FastAPI) ─────────────────────────
COPY frontend/ ./frontend/

# ── Environment defaults ──────────────────────────────────────────────────────
ENV MODEL_BACKEND=ollama
ENV OLLAMA_URL=http://ollama:11434
ENV OLLAMA_MODEL=qwen3

# ── Port ──────────────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# ── Run ───────────────────────────────────────────────────────────────────────
WORKDIR /app/backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
