"""Shared test helpers: a fake embedding model and a scripted fake chat model, so the whole
app (retrieval, tools, agent, API) is tested with no network and no API key."""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from app.chunking import build_chunks
from app.config import Settings
from app.index import VectorIndex
from app.kevdb import connect_kev
from app.retrieval import Retriever
from app.runtime import Runtime
from app.sqlguard import GuardedSQL

DIM = 256


def fake_embed(texts: list[str]) -> list[list[float]]:
    """Character-trigram hashing: similar wording -> similar vectors. Deterministic, offline."""
    out = []
    for t in texts:
        v = np.zeros(DIM)
        s = f" {' '.join(t.lower().split())} "
        for i in range(len(s) - 2):
            v[int(hashlib.md5(s[i : i + 3].encode()).hexdigest(), 16) % DIM] += 1.0
        out.append(v.tolist())
    return out


MINI_DOCS = {
    "ir": ["The incident response life cycle has phases: preparation, detection and analysis, containment, recovery.\nIncident handlers should preserve evidence."],
    "patch": ["Emergency patching applies to vulnerabilities that are being exploited in the wild.\nRoutine patching follows the normal maintenance schedule.", "Test patches before rollout when possible."],
    "logs": ["Log management requires retention policies and time synchronisation across systems."],
}


def make_settings(**kw) -> Settings:
    base = dict(chat_model="fake", embed_model="fake", chunking="section:300", retrieval="dense",
                top_k=3, min_score=0.0, max_steps=4)
    return Settings(**{**base, **kw})


@pytest.fixture
def runtime() -> Runtime:
    settings = make_settings()
    chunks = build_chunks(settings.chunking, MINI_DOCS)
    index = VectorIndex.build(chunks, fake_embed)
    manifest = {"ir": {"id": "NIST IR-Demo"}, "patch": {"id": "NIST Patch-Demo"}, "logs": {"id": "NIST Logs-Demo"}}
    return Runtime(Retriever(chunks, index, fake_embed), GuardedSQL(connect_kev()), settings, manifest, "2026-10-20")


def tool_call(name: str, arguments: str, call_id: str = "c1"):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


def assistant(content: str = "", calls: list | None = None):
    return SimpleNamespace(content=content, tool_calls=calls or None)


class ScriptedChat:
    """Replays a fixed list of assistant messages and records what it was sent."""

    def __init__(self, *messages):
        self.messages = list(messages)
        self.sent: list[list[dict]] = []

    def __call__(self, messages, *, tools=None, temperature=0.0, **_):
        self.sent.append([dict(m) for m in messages])
        return self.messages.pop(0) if len(self.messages) > 1 else self.messages[0]
