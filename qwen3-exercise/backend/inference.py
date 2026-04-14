"""
inference.py — Inference pipeline.

Validates, preprocesses, calls the model, postprocesses.
Keeps all pipeline logic out of main.py.
"""

import re
from typing import Generator
import model as model_module

MAX_CHARS = 8000
MAX_HISTORY = 20   # keep last N turns to stay within context window


def _preprocess(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text[:MAX_CHARS]


def _postprocess(text: str) -> str:
    text = text.strip()
    # Remove any stray model artefacts
    for tag in ["</s>", "<s>", "[INST]", "[/INST]", "<<SYS>>", "<</SYS>>"]:
        text = text.replace(tag, "")
    # Trim "User:" continuations that the model sometimes generates
    if "\nUser:" in text:
        text = text[:text.index("\nUser:")].strip()
    return text.strip()


def _trim_history(history: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Keep the most recent MAX_HISTORY turns."""
    return history[-MAX_HISTORY:]


def run_inference(
    message: str,
    history: list[tuple[str, str]] = None,
    max_tokens: int = 1024,
    temperature: float = 0.7,
) -> str:
    """
    Non-streaming inference. Returns the full response string.
    """
    if not message or not message.strip():
        raise ValueError("Message must not be empty.")

    history = _trim_history(history or [])
    message = _preprocess(message)

    try:
        raw = model_module.generate(
            message=message,
            history=history,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception as exc:
        raise RuntimeError(f"Inference failed: {exc}") from exc

    return _postprocess(raw)


def run_inference_stream(
    message: str,
    history: list[tuple[str, str]] = None,
    max_tokens: int = 1024,
    temperature: float = 0.7,
) -> Generator[str, None, None]:
    """
    Streaming inference. Yields token strings.
    Postprocesses on the full accumulated text at the end.
    """
    if not message or not message.strip():
        raise ValueError("Message must not be empty.")

    history = _trim_history(history or [])
    message = _preprocess(message)

    accumulated = ""
    try:
        for token in model_module.generate_stream(
            message=message,
            history=history,
            max_tokens=max_tokens,
            temperature=temperature,
        ):
            # Yield raw tokens (postprocess happens client-side / on display)
            # But filter out obvious artefacts inline
            if any(bad in token for bad in ["</s>", "[INST]", "[/INST]"]):
                continue
            accumulated += token

            # Stop if model starts echoing "User:"
            if "\nUser:" in accumulated:
                remaining = accumulated[:accumulated.index("\nUser:")].replace(accumulated[:-len(token)], "")
                if remaining:
                    yield remaining
                break

            yield token

    except GeneratorExit:
        return
    except Exception as exc:
        raise RuntimeError(f"Streaming inference failed: {exc}") from exc
