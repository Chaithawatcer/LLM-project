"""The deployed image ships one precomputed index. If it drifts from the committed documents the app
refuses to use it (a stale index), so catch that here, before a deploy, rather than at the first visit."""
import pytest

from app.chunking import build_chunks, load_docs
from app.config import DOCS_DIR, INDEX_DIR, get_settings
from app.index import VectorIndex

STEM = INDEX_DIR / "naive_800_150"


@pytest.mark.skipif(not VectorIndex.exists(STEM), reason="index not built on this machine (python -m app.ingest)")
def test_shipped_index_matches_the_committed_documents():
    index = VectorIndex.load(STEM)
    assert index.chunks == build_chunks("naive:800:150", load_docs(DOCS_DIR))


def test_the_defaults_the_image_relies_on():
    s = get_settings()
    assert (s.chunking, s.retrieval) == ("naive:800:150", "hybrid")   # the Dockerfile ships exactly this index
