"""
main.py — Orion Chatbot Backend

Features:
  • POST /chat          — streaming SSE inference with multi-turn history
  • GET  /health        — liveness check
  • GET/POST /config    — read/switch model backend at runtime
  • Static files        — serves the frontend from /frontend
"""

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

import model as model_module
from inference import run_inference_stream, run_inference

# ── State ─────────────────────────────────────────────────────────────────────
_config = {
    "backend": os.getenv("MODEL_BACKEND", "ollama"),
}

# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[startup] Loading model …")
    model_module.load_model(_config["backend"])
    print(f"[startup] Ready. Backend: {_config['backend']}")
    yield
    print("[shutdown] Done.")

app = FastAPI(
    title="Orion Chat API",
    version="1.0.0",
    lifespan=lifespan,
)

# ── Schemas ───────────────────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: str   # "user" | "assistant"
    content: str

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    history: list[ChatMessage] = Field(default_factory=list)
    max_tokens: int = Field(1024, ge=1, le=4096)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    stream: bool = True

    @field_validator("message")
    @classmethod
    def not_blank(cls, v):
        if not v.strip():
            raise ValueError("message must not be blank")
        return v

class ConfigRequest(BaseModel):
    backend: Optional[str] = None

# ── SSE helpers ───────────────────────────────────────────────────────────────
def sse_data(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"

def sse_done() -> str:
    return "data: [DONE]\n\n"

async def token_generator(
    prompt: str,
    history: list[ChatMessage],
    max_tokens: int,
    temperature: float,
) -> AsyncIterator[str]:
    """
    Yield SSE events. Streams tokens from the model.
    Runs blocking inference in a thread pool to keep the event loop free.
    """
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

    def _stream_worker():
        """Runs in a thread — puts tokens onto the async queue."""
        try:
            for token in run_inference_stream(
                message=prompt,
                history=[(m.role, m.content) for m in history],
                max_tokens=max_tokens,
                temperature=temperature,
            ):
                asyncio.run_coroutine_threadsafe(queue.put(token), loop)
        except Exception as exc:
            asyncio.run_coroutine_threadsafe(queue.put(f"__ERROR__:{exc}"), loop)
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)  # sentinel

    # Kick off in thread
    import concurrent.futures
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    executor.submit(_stream_worker)

    while True:
        token = await queue.get()
        if token is None:
            break
        if isinstance(token, str) and token.startswith("__ERROR__:"):
            yield sse_data({"error": token[10:]})
            break
        yield sse_data({"token": token})
        await asyncio.sleep(0)  # yield control

    yield sse_done()

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_backend": _config["backend"],
        "timestamp": time.time(),
    }

@app.get("/config")
async def get_config():
    return {"backend": _config["backend"]}

@app.post("/config")
async def set_config(req: ConfigRequest):
    allowed = {"ollama", "llamacpp", "transformers"}
    if req.backend and req.backend not in allowed:
        raise HTTPException(400, f"backend must be one of {allowed}")
    if req.backend and req.backend != _config["backend"]:
        _config["backend"] = req.backend
        try:
            model_module.load_model(req.backend)
        except Exception as exc:
            raise HTTPException(500, f"Failed to load backend '{req.backend}': {exc}")
    return {"backend": _config["backend"], "status": "ok"}

@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Multi-turn chat with optional SSE streaming.

    Streaming response format (SSE):
        data: {"token": "Hello"}
        data: {"token": " world"}
        data: [DONE]

    Non-streaming:
        {"response": "Hello world", "elapsed_ms": 123}
    """
    if req.stream:
        return StreamingResponse(
            token_generator(
                prompt=req.message.strip(),
                history=req.history,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",       # disable nginx buffering
                "Connection": "keep-alive",
            },
        )
    else:
        t0 = time.perf_counter()
        try:
            response = run_inference(
                message=req.message.strip(),
                history=[(m.role, m.content) for m in req.history],
                max_tokens=req.max_tokens,
                temperature=req.temperature,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        except RuntimeError as exc:
            raise HTTPException(500, str(exc))
        return {
            "response": response,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
        }

# ── Global error handler ──────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def handle_unhandled(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {exc}"},
    )

# ── Serve frontend (must be LAST — catch-all) ─────────────────────────────────
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="static")
