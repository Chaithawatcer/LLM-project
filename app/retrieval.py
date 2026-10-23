"""Retrieval over the document chunks: dense, BM25, or hybrid (reciprocal rank fusion).

The `min_score` floor is the safety knob: if the best *cosine* similarity is below it we
return nothing and the agent says "I can't answer that" instead of grounding on junk.
Cosine is the only score with a stable scale, so the floor always gates on it - in hybrid
mode too (BM25 scores are not comparable across queries).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from app.bm25 import BM25, tokenize
from app.chunking import Chunk
from app.index import VectorIndex

RRF_K = 60       # the constant from the RRF paper / Lecture 8
CANDIDATES = 50  # how deep each ranker contributes to the fusion


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float          # ranking score for the mode used (cosine / BM25 / RRF)
    dense: float | None   # cosine similarity when it was computed


class Retriever:
    def __init__(
        self,
        chunks: list[Chunk],
        index: VectorIndex | None = None,
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
    ) -> None:
        self.chunks = chunks
        self.index = index
        self.embed_fn = embed_fn
        self._bm25: BM25 | None = None

    @property
    def bm25(self) -> BM25:
        if self._bm25 is None:  # built on first use; BM25 needs no embeddings
            self._bm25 = BM25([tokenize(c.text) for c in self.chunks])
        return self._bm25

    def search(
        self,
        query: str,
        *,
        mode: str = "dense",
        k: int = 5,
        min_score: float = 0.0,
        qvec=None,
    ) -> list[Hit]:
        """Top-k chunks. `qvec` lets the evaluation embed each question once and reuse it."""
        if mode == "bm25":
            s = self.bm25.scores(tokenize(query))
            order = np.argsort(-s, kind="stable")[:k]
            return [Hit(self.chunks[i], float(s[i]), None) for i in order if s[i] > 0]

        if mode not in ("dense", "hybrid"):
            raise ValueError(f"unknown retrieval mode {mode!r}")
        if self.index is None or self.embed_fn is None:
            raise RuntimeError(f"mode {mode!r} needs a vector index - run `python -m app.ingest`")
        if qvec is None:
            qvec = self.embed_fn([query])[0]
        dense = self.index.scores(qvec)
        dense_order = np.argsort(-dense, kind="stable")

        if mode == "dense":
            return [Hit(self.chunks[i], float(dense[i]), float(dense[i])) for i in dense_order[:k] if dense[i] >= min_score]

        if dense[dense_order[0]] < min_score:  # hybrid: gate on cosine, then fuse
            return []
        s = self.bm25.scores(tokenize(query))
        bm25_order = [i for i in np.argsort(-s, kind="stable")[:CANDIDATES] if s[i] > 0]
        fused: dict[int, float] = {}
        for ranking in (dense_order[:CANDIDATES], bm25_order):
            for rank, i in enumerate(ranking, 1):
                fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (RRF_K + rank)
        top = sorted(fused, key=fused.get, reverse=True)[:k]
        return [Hit(self.chunks[i], fused[i], float(dense[i])) for i in top]
