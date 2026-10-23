import numpy as np
import pytest

from app.bm25 import BM25, tokenize
from app.chunking import build_chunks, chunk_naive, chunk_section
from app.index import VectorIndex
from app.retrieval import Retriever
from tests.conftest import MINI_DOCS, fake_embed


def test_naive_windows_overlap_and_cover_text():
    text = " ".join(f"word{i}" for i in range(300))
    chunks = chunk_naive("d", [text], size=200, overlap=50)
    assert all(len(c.text) <= 200 for c in chunks)
    assert chunks[0].text[-50:] == chunks[1].text[:50]  # overlap really overlaps
    assert [c.idx for c in chunks] == list(range(len(chunks)))


def test_naive_tracks_the_page_each_chunk_starts_on():
    chunks = chunk_naive("d", ["a" * 500, "b" * 500, "c" * 500], size=400, overlap=0)
    assert chunks[0].page == 1 and chunks[-1].page == 3


def test_naive_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_naive("d", ["x"], size=100, overlap=100)


def test_section_never_exceeds_max_and_keeps_sentences_whole():
    page = "First sentence is here. Second sentence follows it. " * 30
    chunks = chunk_section("d", [page], max_chars=300)
    assert all(len(c.text) <= 300 for c in chunks)
    assert all(c.text.rstrip().endswith(".") for c in chunks)  # no mid-sentence cuts


def test_section_handles_one_giant_line():
    chunks = chunk_section("d", ["x" * 5000], max_chars=1000)
    assert all(len(c.text) <= 1000 for c in chunks) and sum(len(c.text) for c in chunks) == 5000


def test_build_chunks_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        build_chunks("semantic", MINI_DOCS)


def test_tokenize_keeps_cve_ids_and_splits_compounds():
    toks = tokenize("Patch-management for CVE-2023-4966 and SP 800-61r3")
    assert "cve-2023-4966" in toks and "patch" in toks and "management" in toks and "800-61r3" in toks
    assert tokenize("the") == []


def test_bm25_ranks_the_exact_term_first():
    docs = [tokenize(t) for t in ["general security guidance", "ransomware recovery steps", "log retention policy"]]
    scores = BM25(docs).scores(tokenize("ransomware"))
    assert int(np.argmax(scores)) == 1 and scores[0] == 0


def test_dense_floor_returns_nothing_when_nothing_is_similar(runtime):
    r = runtime.retriever
    assert r.search("incident response phases", mode="dense", k=3, min_score=0.0)
    assert r.search("zzzz qqqq xxxx", mode="dense", k=3, min_score=0.95) == []


def test_bm25_mode_needs_no_index():
    chunks = build_chunks("section:300", MINI_DOCS)
    hits = Retriever(chunks).search("emergency patching exploited", mode="bm25", k=2)
    assert hits and hits[0].chunk.doc == "patch"


def test_dense_mode_without_index_fails_loudly():
    with pytest.raises(RuntimeError, match="vector index"):
        Retriever(build_chunks("section:300", MINI_DOCS)).search("x", mode="dense")


def test_hybrid_fuses_both_rankers_and_gates_on_cosine(runtime):
    r = runtime.retriever
    hits = r.search("emergency patching of exploited vulnerabilities", mode="hybrid", k=3)
    assert hits and hits[0].chunk.doc == "patch" and all(h.dense is not None for h in hits)
    assert r.search("emergency patching", mode="hybrid", k=3, min_score=0.999) == []


def test_index_round_trip(tmp_path):
    chunks = build_chunks("section:300", MINI_DOCS)
    idx = VectorIndex.build(chunks, fake_embed)
    idx.save(tmp_path / "t")
    again = VectorIndex.load(tmp_path / "t")
    assert again.chunks == chunks and np.allclose(again.matrix, idx.matrix)
