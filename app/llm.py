"""The only place the app talks to a model: chat() and embed() through the class proxy.

The OpenAI client reads OPENAI_API_KEY / OPENAI_BASE_URL from the environment; the key is
never read from a file in the repo. The client is built lazily so importing the app (and
running the tests) works without a key.
"""
from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

from app.config import DEFAULT_BASE_URL, get_settings

_client: OpenAI | None = None


class LLMNotConfigured(RuntimeError):
    """Raised when there is no API key, with a message that says how to fix it."""


def client() -> OpenAI:
    global _client
    if _client is None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise LLMNotConfigured(
                "OPENAI_API_KEY is not set. Put your course key in .env (see .env.example)."
            )
        os.environ.setdefault("OPENAI_BASE_URL", DEFAULT_BASE_URL)
        _client = OpenAI(timeout=60.0, max_retries=6)  # the class proxy is shared: ride out 429s with backoff
    return _client


def chat(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    temperature: float = 0.0,
    max_tokens: int = 700,
    model: str | None = None,
):
    """One chat turn; returns the assistant message (with .content and .tool_calls).

    temperature defaults to 0: tool choice and SQL are control flow, so they must be
    repeatable (and anything we evaluate must be reproducible).
    """
    kwargs: dict[str, Any] = {
        "model": model or get_settings().chat_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        kwargs["tools"] = tools
    return client().chat.completions.create(**kwargs).choices[0].message


def embed(texts: list[str], *, batch: int = 32, model: str | None = None) -> list[list[float]]:
    """Embed texts in batches; returns one vector per input, in order."""
    out: list[list[float]] = []
    name = model or get_settings().embed_model
    for i in range(0, len(texts), batch):
        part = [t if t.strip() else " " for t in texts[i : i + batch]]  # empty input is rejected
        resp = client().embeddings.create(model=name, input=part)
        out.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
    return out
