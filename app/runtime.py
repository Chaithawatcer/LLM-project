"""Wire the pieces together once per process: chunks + index -> Retriever, KEV -> GuardedSQL."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from app import llm
from app.chunking import build_chunks, load_docs, load_manifest, strategy_slug
from app.config import DOCS_DIR, INDEX_DIR, Settings, get_settings
from app.index import VectorIndex
from app.kevdb import connect_kev
from app.retrieval import Retriever
from app.sqlguard import GuardedSQL
from app.tools import ToolContext


class IndexMissing(RuntimeError):
    """The vector index is absent or stale; the message says how to build it."""


@dataclass
class Runtime:
    retriever: Retriever
    sql: GuardedSQL
    settings: Settings
    manifest: dict[str, dict]
    snapshot: str  # newest date_added in the KEV table - the "today" the agent reasons from

    def new_context(self) -> ToolContext:
        """Fresh per-question state (sources, SQL log) so one request can't leak into another."""
        return ToolContext(self.retriever, self.sql, self.settings, self.manifest)


def build_runtime(
    settings: Settings | None = None,
    *,
    embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
) -> Runtime:
    settings = settings or get_settings()
    docs = load_docs(DOCS_DIR)
    if not docs:
        raise RuntimeError("no documents in data/docs - run `uv run python scripts/fetch_data.py`")
    chunks = build_chunks(settings.chunking, docs)

    index = None
    if settings.retrieval != "bm25":  # BM25 needs no embeddings, so no index
        stem = INDEX_DIR / strategy_slug(settings.chunking)
        if not VectorIndex.exists(stem):
            raise IndexMissing(f"no vector index for {settings.chunking!r} - run `uv run python -m app.ingest`")
        index = VectorIndex.load(stem)
        if index.chunks != chunks:
            raise IndexMissing("the vector index is stale (documents or chunking changed) - re-run `uv run python -m app.ingest`")

    sql = GuardedSQL(connect_kev())
    snapshot = sql.run("SELECT MAX(date_added) FROM kev").rows[0][0]
    return Runtime(Retriever(chunks, index, embed_fn or llm.embed), sql, settings, load_manifest(DOCS_DIR), snapshot)
