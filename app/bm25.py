"""BM25 keyword scoring (Lecture 8), implemented from scratch so it is readable.

BM25 is the keyword half of hybrid search and the "BM25-only" baseline. It shines on exact
tokens - CVE ids, standard numbers, product names - where embeddings blur.
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

import numpy as np

# a token is a run of letters/digits that may contain - _ . inside (cve-2023-4966, 800-61r3)
_TOKEN = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")
_SEP = re.compile(r"[-_.]")
_STOP = frozenset(
    "a an and are as at be been by for from has have in is it its of on or that the this to was were will with".split()
)


def tokenize(text: str) -> list[str]:
    """Lower-case tokens; a compound like 'cve-2023-4966' also yields its parts so that a
    query for 'patch management' can still hit 'patch-management'."""
    out: list[str] = []
    for tok in _TOKEN.findall(unicodedata.normalize("NFKC", text).lower()):
        if tok in _STOP:
            continue
        out.append(tok)
        if _SEP.search(tok):
            out.extend(p for p in _SEP.split(tok) if len(p) > 1 and p not in _STOP)
    return out


class BM25:
    def __init__(self, docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.n = len(docs_tokens)
        self.lens = np.array([len(d) for d in docs_tokens], dtype=np.float64)
        self.avg_len = float(self.lens.mean()) if self.n else 1.0
        postings: dict[str, tuple[list[int], list[int]]] = {}
        for i, toks in enumerate(docs_tokens):
            for term, tf in Counter(toks).items():
                ids, tfs = postings.setdefault(term, ([], []))
                ids.append(i)
                tfs.append(tf)
        self.postings = {t: (np.array(ids), np.array(tfs, dtype=np.float64)) for t, (ids, tfs) in postings.items()}
        self.idf = {t: math.log(1 + (self.n - len(ids) + 0.5) / (len(ids) + 0.5)) for t, (ids, _) in self.postings.items()}

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        s = np.zeros(self.n)
        for term in set(query_tokens):
            if term not in self.postings:
                continue
            ids, tf = self.postings[term]
            norm = tf + self.k1 * (1 - self.b + self.b * self.lens[ids] / self.avg_len)
            s[ids] += self.idf[term] * tf * (self.k1 + 1) / norm
        return s
