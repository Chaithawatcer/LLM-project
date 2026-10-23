"""Chunk the documents, embed every chunk, save the vector index.

    uv run python -m app.ingest                          # the strategy in .env / CHUNKING
    uv run python -m app.ingest naive:800:150 section:1200   # several, e.g. for the evaluation

Needs the course API key (embeddings go through the class proxy). Embeddings are cached on disk
(app/embcache.py), so an interrupted run resumes where it stopped. Re-run it whenever the
documents or the chunking strategy change - the app refuses a stale index.
"""
from __future__ import annotations

import sys
import time

from app.chunking import build_chunks, load_docs, strategy_slug
from app.config import DOCS_DIR, INDEX_DIR, get_settings
from app.embcache import EmbeddingCache, embed_cached
from app.index import VectorIndex


def build_index(strategy: str) -> VectorIndex:
    chunks = build_chunks(strategy, load_docs(DOCS_DIR))
    print(f"{strategy}: {len(chunks)} chunks", flush=True)
    started = time.time()
    cache = EmbeddingCache()
    index = VectorIndex.build(chunks, lambda texts: embed_cached(texts, cache=cache))
    index.save(INDEX_DIR / strategy_slug(strategy))
    print(f"{strategy}: saved in {(time.time() - started) / 60:.1f} min -> data/index/{strategy_slug(strategy)}.*", flush=True)
    return index


def main() -> None:
    for strategy in sys.argv[1:] or [get_settings().chunking]:
        build_index(strategy)


if __name__ == "__main__":
    main()
