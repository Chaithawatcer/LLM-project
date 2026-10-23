"""FastAPI backend: POST /api/ask streams the agent's work as Server-Sent Events.

  GET  /api/health   what is configured (never the key itself)
  POST /api/ask      {question, mode} -> SSE frames  data: {"type": "act"|"obs"|"final"|"error", ...}
  GET  /             the single-page UI in web/

Pydantic rejects a malformed body with a 422 before any of our code runs.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Iterator
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from app import llm
from app.agent import run_agent
from app.chunking import strategy_slug
from app.config import INDEX_DIR, get_settings
from app.index import VectorIndex
from app.runtime import IndexMissing, Runtime, build_runtime
from app.schemas import AskRequest

log = logging.getLogger("cyber-advisor")
app = FastAPI(title="Cyber Advisor")
_lock = threading.Lock()


def get_runtime(application: FastAPI) -> Runtime:
    """Built on first use (so importing the app needs no key or index); tests inject their own."""
    with _lock:
        if getattr(application.state, "runtime", None) is None:
            application.state.runtime = build_runtime()
        return application.state.runtime


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"


@app.get("/api/health")
def health() -> dict:
    s = get_settings()
    index_needed = s.retrieval != "bm25"
    return {
        "llm_key_set": bool(os.environ.get("OPENAI_API_KEY")),
        "chat_model": s.chat_model,
        "retrieval": s.retrieval,
        "chunking": s.chunking,
        "min_score": s.min_score,
        "index_ready": (not index_needed) or VectorIndex.exists(INDEX_DIR / strategy_slug(s.chunking)),
    }


@app.post("/api/ask")
def ask(body: AskRequest, request: Request) -> StreamingResponse:
    def frames() -> Iterator[str]:
        try:
            runtime = get_runtime(request.app)
            chat_fn = getattr(request.app.state, "chat_fn", None) or llm.chat
            for event in run_agent(body.question, runtime.new_context(), mode=body.mode,
                                   snapshot=runtime.snapshot, chat_fn=chat_fn):
                yield _sse(event)
        except (llm.LLMNotConfigured, IndexMissing) as exc:  # actionable, safe to show
            yield _sse({"type": "error", "message": str(exc)})
        except Exception:  # noqa: BLE001 - never leak internals to the browser
            log.exception("agent failed")
            yield _sse({"type": "error", "message": "Something went wrong while answering. Check the server log."})

    return StreamingResponse(frames(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


WEB = Path(__file__).resolve().parent.parent / "web"
app.mount("/", StaticFiles(directory=WEB, html=True), name="web")  # last: it matches everything
