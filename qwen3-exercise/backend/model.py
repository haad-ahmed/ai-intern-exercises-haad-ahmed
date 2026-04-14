"""
model.py — Model loader (warm service pattern).

Supports three backends:
  • ollama       — HTTP to a running Ollama server (default)
  • llamacpp     — llama-cpp-python with local GGUF
  • transformers — HuggingFace AutoModel

Call load_model(backend) once at startup.
Then call generate(...) or generate_stream(...) for inference.
"""

import os
import re
import requests
from typing import Generator, Optional

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_URL   = os.getenv("OLLAMA_URL",   "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3")
GGUF_PATH    = os.getenv("GGUF_PATH",    "models/model.gguf")
HF_MODEL_ID  = os.getenv("HF_MODEL_ID",  "Qwen/Qwen3-0.6B")

# ── Global state ──────────────────────────────────────────────────────────────
_backend:   str      = "ollama"
_llm:       object   = None
_tokenizer: object   = None


def load_model(backend: str = "ollama") -> None:
    global _backend, _llm, _tokenizer
    _backend = backend

    if backend == "ollama":
        try:
            r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
            r.raise_for_status()
            print(f"[model] Ollama ready at {OLLAMA_URL} — model: {OLLAMA_MODEL}")
        except Exception as e:
            print(f"[model] WARNING: Ollama not reachable ({e}). Will retry on first request.")

    elif backend == "llamacpp":
        from llama_cpp import Llama
        print(f"[model] Loading GGUF: {GGUF_PATH}")
        _llm = Llama(model_path=GGUF_PATH, n_ctx=4096, n_threads=os.cpu_count() or 4, verbose=False)
        print("[model] GGUF ready.")

    elif backend == "transformers":
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        print(f"[model] Loading HF model: {HF_MODEL_ID}")
        _tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
        _llm = AutoModelForCausalLM.from_pretrained(
            HF_MODEL_ID,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        print("[model] HF model ready.")

    else:
        raise ValueError(f"Unknown backend: {backend!r}")


def _build_prompt(message: str, history: list[tuple[str, str]]) -> str:
    """Build a simple conversational prompt from history + new message."""
    lines = []
    for role, content in history:
        tag = "User" if role == "user" else "Assistant"
        lines.append(f"{tag}: {content}")
    lines.append(f"User: {message}")
    lines.append("Assistant:")
    return "\n\n".join(lines)


def generate(
    message: str,
    history: list[tuple[str, str]] = None,
    max_tokens: int = 1024,
    temperature: float = 0.7,
) -> str:
    """Non-streaming inference. Returns the full response string."""
    history = history or []
    prompt = _build_prompt(message, history)

    if _backend == "ollama":
        return _ollama_generate(prompt, max_tokens, temperature, stream=False)
    elif _backend == "llamacpp":
        return _llamacpp_generate(prompt, max_tokens, temperature)
    elif _backend == "transformers":
        return _hf_generate(prompt, max_tokens, temperature)
    raise RuntimeError(f"Backend not loaded: {_backend}")


def generate_stream(
    message: str,
    history: list[tuple[str, str]] = None,
    max_tokens: int = 1024,
    temperature: float = 0.7,
) -> Generator[str, None, None]:
    """Streaming inference. Yields token strings one at a time."""
    history = history or []
    prompt = _build_prompt(message, history)

    if _backend == "ollama":
        yield from _ollama_stream(prompt, max_tokens, temperature)
    elif _backend == "llamacpp":
        yield from _llamacpp_stream(prompt, max_tokens, temperature)
    elif _backend == "transformers":
        yield from _hf_stream(prompt, max_tokens, temperature)
    else:
        raise RuntimeError(f"Backend not loaded: {_backend}")


# ── Ollama ────────────────────────────────────────────────────────────────────
def _ollama_generate(prompt: str, max_tokens: int, temperature: float, stream: bool) -> str:
    import json as _json
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": temperature},
    }
    r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=180)
    r.raise_for_status()
    return r.json().get("response", "").strip()


def _ollama_stream(prompt: str, max_tokens: int, temperature: float) -> Generator[str, None, None]:
    import json as _json
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {"num_predict": max_tokens, "temperature": temperature},
    }
    with requests.post(f"{OLLAMA_URL}/api/generate", json=payload, stream=True, timeout=180) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            try:
                obj = _json.loads(line)
                token = obj.get("response", "")
                if token:
                    yield token
                if obj.get("done"):
                    break
            except Exception:
                continue


# ── llama-cpp-python ──────────────────────────────────────────────────────────
def _llamacpp_generate(prompt: str, max_tokens: int, temperature: float) -> str:
    if _llm is None:
        raise RuntimeError("llama-cpp model not loaded.")
    out = _llm(prompt, max_tokens=max_tokens, temperature=temperature, stop=["</s>", "User:"])
    return out["choices"][0]["text"].strip()


def _llamacpp_stream(prompt: str, max_tokens: int, temperature: float) -> Generator[str, None, None]:
    if _llm is None:
        raise RuntimeError("llama-cpp model not loaded.")
    for chunk in _llm(prompt, max_tokens=max_tokens, temperature=temperature,
                      stop=["</s>", "User:"], stream=True):
        yield chunk["choices"][0]["text"]


# ── HuggingFace Transformers ──────────────────────────────────────────────────
def _hf_generate(prompt: str, max_tokens: int, temperature: float) -> str:
    if _llm is None or _tokenizer is None:
        raise RuntimeError("HF model not loaded.")
    import torch
    inputs = _tokenizer(prompt, return_tensors="pt").to(_llm.device)
    with torch.no_grad():
        tokens = _llm.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=_tokenizer.eos_token_id,
        )
    new = tokens[0][inputs["input_ids"].shape[1]:]
    return _tokenizer.decode(new, skip_special_tokens=True).strip()


def _hf_stream(prompt: str, max_tokens: int, temperature: float) -> Generator[str, None, None]:
    """
    Word-by-word streaming for HuggingFace (via TextIteratorStreamer).
    """
    if _llm is None or _tokenizer is None:
        raise RuntimeError("HF model not loaded.")
    import torch
    from transformers import TextIteratorStreamer
    from threading import Thread

    inputs = _tokenizer(prompt, return_tensors="pt").to(_llm.device)
    streamer = TextIteratorStreamer(_tokenizer, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        pad_token_id=_tokenizer.eos_token_id,
        streamer=streamer,
    )
    thread = Thread(target=_llm.generate, kwargs=gen_kwargs)
    thread.start()

    for token in streamer:
        yield token

    thread.join()
