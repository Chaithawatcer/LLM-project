"""An on-disk embedding cache.

The class proxy embeds at roughly 0.4 s per chunk and handles one request at a time (measured:
four parallel batches took exactly as long as four sequential ones), so a full index is tens of
minutes. Caching by (model, text) means a re-run, a resumed run, or a second chunking that shares
text never pays twice - and a crash at minute 25 loses at most one batch.
"""
from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path

import numpy as np

from app import llm
from app.config import INDEX_DIR, get_settings


class EmbeddingCache:
    def __init__(self, path: Path = INDEX_DIR / "embed_cache.db") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.execute("CREATE TABLE IF NOT EXISTS emb(k TEXT PRIMARY KEY, v BLOB NOT NULL)")

    @staticmethod
    def key(model: str, text: str) -> str:
        return hashlib.sha1(f"{model}\0{text}".encode("utf-8")).hexdigest()

    def get_many(self, keys: list[str]) -> dict[str, np.ndarray]:
        found: dict[str, np.ndarray] = {}
        for i in range(0, len(keys), 500):  # SQLite limits the number of bound parameters
            part = keys[i : i + 500]
            marks = ",".join("?" * len(part))
            for k, v in self._db.execute(f"SELECT k, v FROM emb WHERE k IN ({marks})", part):
                found[k] = np.frombuffer(v, dtype=np.float32)
        return found

    def put_many(self, items: list[tuple[str, list[float]]]) -> None:
        self._db.executemany(
            "INSERT OR REPLACE INTO emb(k, v) VALUES(?, ?)",
            [(k, np.asarray(v, dtype=np.float32).tobytes()) for k, v in items],
        )
        self._db.commit()  # every batch is durable


def embed_cached(
    texts: list[str],
    *,
    batch: int = 8,
    cache: EmbeddingCache | None = None,
    model: str | None = None,
    progress: bool = True,
) -> list[np.ndarray]:
    """Like llm.embed, but only the texts not seen before go to the proxy, in small batches."""
    model = model or get_settings().embed_model
    cache = cache or EmbeddingCache()
    keys = [cache.key(model, t) for t in texts]
    have = cache.get_many(list(dict.fromkeys(keys)))
    todo = list({k: t for k, t in zip(keys, texts) if k not in have}.items())  # unique, in order
    started = time.time()
    for i in range(0, len(todo), batch):
        part = todo[i : i + batch]
        vecs = llm.embed([t for _, t in part], batch=batch, model=model)
        cache.put_many([(k, v) for (k, _), v in zip(part, vecs)])
        have.update({k: np.asarray(v, dtype=np.float32) for (k, _), v in zip(part, vecs)})
        if progress and ((i // batch) % 10 == 9 or i + batch >= len(todo)):
            done = min(i + batch, len(todo))
            rate = done / (time.time() - started)
            print(f"  {done}/{len(todo)} new  ({len(texts) - len(todo)} cached)  {rate:.1f}/s  "
                  f"eta {(len(todo) - done) / rate / 60:.0f} min", flush=True)
    return [have[k] for k in keys]
