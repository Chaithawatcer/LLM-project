"""Evaluation metrics, kept free of any LLM or I/O so they are easy to unit-test.

Retrieval is scored at CHUNK level against evidence phrases ("does the retrieved chunk
actually contain the answer-bearing sentence?"), not just at document level. Document-level
relevance cannot see chunking at all - every chunking of the right PDF is "right" - whereas
a naive window that cuts the answer in half scores a miss here, which is the point.
"""
from __future__ import annotations

import math
import re
import unicodedata

_PUNCT = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                        "–": "-", "—": "-", "−": "-", " ": " "})


def norm(text: str) -> str:
    """Case-, whitespace- and typography-insensitive form used for all text matching."""
    return " ".join(unicodedata.normalize("NFKC", text).translate(_PUNCT).lower().split())


def has_evidence(chunk_text: str, evidence: list[str]) -> bool:
    body = norm(chunk_text)
    return any(norm(e) in body for e in evidence)


def retrieval_scores(ranked: list[tuple[str, str]], evidence: list[str], docs: list[str], k: int) -> dict:
    """ranked = [(doc_slug, chunk_text), ...] best first.
    hit@k   1 if any of the top-k chunks contains an evidence phrase
    mrr     1/rank of the first such chunk within the top-k (0 if none)
    doc_recall@k  fraction of the gold documents that appear among the top-k chunks
    """
    top = ranked[:k]
    first = next((i for i, (_, text) in enumerate(top, 1) if has_evidence(text, evidence)), None)
    got_docs = {d for d, _ in top}
    return {
        "hit": 1.0 if first else 0.0,
        "mrr": 1.0 / first if first else 0.0,
        "doc_recall": len(got_docs & set(docs)) / len(docs) if docs else math.nan,
    }


def _cell(v):
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return round(float(v), 4)
    return str(v).strip() if v is not None else None


def rows_match(pred: list[tuple], gold: list[tuple], ordered: bool = False) -> bool:
    """Execution match: do two result sets contain the same values? Column names, aliases and
    (unless `ordered`) row order don't matter - only what the query returns."""
    p = [tuple(_cell(v) for v in r) for r in pred]
    g = [tuple(_cell(v) for v in r) for r in gold]
    return p == g if ordered else sorted(map(repr, p)) == sorted(map(repr, g))


def answer_has_all(answer: str, needles: list[str]) -> bool:
    """True if every needle appears in the answer (thousands separators ignored: 1,754 == 1754)."""
    text = norm(re.sub(r"(?<=\d),(?=\d{3}\b)", "", answer))
    return all(norm(n) in text for n in needles)


def mean(xs: list[float]) -> float:
    xs = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else math.nan


def balanced_accuracy(pos_pass: list[bool], neg_pass: list[bool]) -> float:
    """Mean of the true-positive rate and the true-negative rate (robust to unequal group sizes)."""
    tpr = sum(pos_pass) / len(pos_pass) if pos_pass else math.nan
    tnr = sum(neg_pass) / len(neg_pass) if neg_pass else math.nan
    return mean([tpr, tnr])
