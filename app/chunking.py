"""Turn extracted documents into chunks.

Two strategies, selected by a string so that an index (and an experiment) is fully named
by it:

  naive:800:150   fixed-size character windows with overlap. The obvious default - and the
                  baseline we measure against. It happily cuts words and sentences in half.
  section:1200    boundary-aware: keep lines/sentences whole and pack them up to N chars.

Each chunk remembers its document and the page it starts on, so answers can cite
"NIST SP 800-61 Rev. 3, p.12" and a human can check the claim.
"""
from __future__ import annotations

import bisect
import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    doc: str   # document slug, e.g. "sp800-61r3"
    page: int  # 1-based page where the chunk starts
    idx: int   # position within the document
    text: str

    @property
    def key(self) -> str:
        return f"{self.doc}#{self.idx}"


def load_docs(docs_dir: Path) -> dict[str, list[str]]:
    """slug -> list of page texts (pages are separated by a form-feed in the .txt files)."""
    return {p.stem: p.read_text(encoding="utf-8").split("\f") for p in sorted(docs_dir.glob("*.txt"))}


def load_manifest(docs_dir: Path) -> dict[str, dict]:
    path = docs_dir / "_manifest.json"
    if not path.exists():
        return {}
    return {m["slug"]: m for m in json.loads(path.read_text(encoding="utf-8"))}


def chunk_naive(doc: str, pages: list[str], size: int = 800, overlap: int = 150) -> list[Chunk]:
    if not 0 <= overlap < size:
        raise ValueError("need 0 <= overlap < size")
    norm = [" ".join(p.split()) for p in pages]
    starts, pos = [], 0
    for p in norm:
        starts.append(pos)
        pos += len(p) + 1
    text = " ".join(norm)
    chunks: list[Chunk] = []
    for i in range(0, len(text), size - overlap):
        piece = text[i : i + size]
        if piece.strip():
            chunks.append(Chunk(doc, bisect.bisect_right(starts, i), len(chunks), piece))
    return chunks


_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _split_long(unit: str, max_chars: int) -> list[str]:
    """Break an over-long line at sentence ends, then (last resort) at spaces."""
    if len(unit) <= max_chars:
        return [unit]
    out: list[str] = []
    cur = ""
    for sent in _SENTENCE.split(unit):
        while len(sent) > max_chars:  # one enormous "sentence": a table or a list
            cut = sent.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            if cur:
                out.append(cur)
                cur = ""
            out.append(sent[:cut])
            sent = sent[cut:].lstrip()
        if cur and len(cur) + 1 + len(sent) > max_chars:
            out.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        out.append(cur)
    return out


def chunk_section(doc: str, pages: list[str], max_chars: int = 1200) -> list[Chunk]:
    units: list[tuple[int, str]] = []  # (page, line) with every line <= max_chars
    for pno, page in enumerate(pages, 1):
        for line in re.split(r"\n+", page):
            line = " ".join(line.split())
            if line:
                units.extend((pno, u) for u in _split_long(line, max_chars))

    chunks: list[Chunk] = []
    cur: list[str] = []
    cur_page, cur_len = 1, 0

    def flush() -> None:
        nonlocal cur, cur_len
        if cur:
            chunks.append(Chunk(doc, cur_page, len(chunks), " ".join(cur)))
        cur, cur_len = [], 0

    for pno, unit in units:
        if cur and cur_len + 1 + len(unit) > max_chars:
            flush()
        if not cur:
            cur_page = pno
        cur.append(unit)
        cur_len += len(unit) + 1
    flush()
    return chunks


def build_chunks(strategy: str, docs: dict[str, list[str]]) -> list[Chunk]:
    kind, *params = strategy.split(":")
    chunks: list[Chunk] = []
    for slug, pages in docs.items():
        if kind == "naive":
            size = int(params[0]) if params else 800
            overlap = int(params[1]) if len(params) > 1 else 150
            chunks.extend(chunk_naive(slug, pages, size, overlap))
        elif kind == "section":
            chunks.extend(chunk_section(slug, pages, int(params[0]) if params else 1200))
        else:
            raise ValueError(f"unknown chunking strategy {strategy!r} (use naive:800:150 or section:1200)")
    return chunks


def strategy_slug(strategy: str) -> str:
    """A file-system-safe name for a strategy (':' is not allowed in Windows file names)."""
    return strategy.replace(":", "_")
