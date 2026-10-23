"""A tiny vector index: normalised embeddings in a NumPy matrix, cosine = one dot product.

For a few thousand chunks, brute force is instant and far easier to read than a database.
(Production swap: pgvector / sqlite-vec - same scores() shape.) The index is derived data:
it is rebuilt by `python -m app.ingest` and is not committed.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import numpy as np

from app.chunking import Chunk


def _unit(m: np.ndarray) -> np.ndarray:
    return m / (np.linalg.norm(m, axis=-1, keepdims=True) + 1e-9)


class VectorIndex:
    def __init__(self, chunks: list[Chunk], matrix: np.ndarray) -> None:
        if len(chunks) != len(matrix):
            raise ValueError("chunks and matrix disagree in length")
        self.chunks = chunks
        self.matrix = _unit(np.asarray(matrix, dtype=np.float32))

    @classmethod
    def build(cls, chunks: list[Chunk], embed_fn: Callable[[list[str]], list[list[float]]]) -> "VectorIndex":
        return cls(chunks, np.asarray(embed_fn([c.text for c in chunks]), dtype=np.float32))

    def scores(self, qvec) -> np.ndarray:
        """Cosine similarity of the query against every chunk."""
        return self.matrix @ _unit(np.asarray(qvec, dtype=np.float32))

    def save(self, stem: Path) -> None:
        stem.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(f"{stem}.npz", matrix=self.matrix)
        Path(f"{stem}.json").write_text(json.dumps([asdict(c) for c in self.chunks]), encoding="utf-8")

    @classmethod
    def exists(cls, stem: Path) -> bool:
        return Path(f"{stem}.npz").exists() and Path(f"{stem}.json").exists()

    @classmethod
    def load(cls, stem: Path) -> "VectorIndex":
        matrix = np.load(f"{stem}.npz")["matrix"]
        chunks = [Chunk(**c) for c in json.loads(Path(f"{stem}.json").read_text(encoding="utf-8"))]
        return cls(chunks, matrix)
