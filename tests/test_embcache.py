import numpy as np

from app import embcache
from app.embcache import EmbeddingCache, embed_cached
from tests.conftest import fake_embed


def test_only_unseen_texts_reach_the_proxy_and_results_keep_order(tmp_path, monkeypatch):
    sent: list[list[str]] = []

    def fake_llm_embed(texts, *, batch=32, model=None):
        sent.append(list(texts))
        return fake_embed(texts)

    monkeypatch.setattr(embcache.llm, "embed", fake_llm_embed)
    cache = EmbeddingCache(tmp_path / "c.db")

    first = embed_cached(["alpha", "beta", "alpha"], batch=2, cache=cache, model="m", progress=False)
    assert sum(len(b) for b in sent) == 2                      # "alpha" duplicated -> embedded once
    assert np.allclose(first[0], first[2])

    sent.clear()
    again = embed_cached(["beta", "gamma", "alpha"], batch=2, cache=cache, model="m", progress=False)
    assert sent == [["gamma"]]                                  # only the new text is sent
    assert np.allclose(again[0], first[1]) and np.allclose(again[2], first[0])   # order follows the input


def test_cache_is_per_model(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(embcache.llm, "embed", lambda t, **kw: calls.append(1) or fake_embed(t))
    cache = EmbeddingCache(tmp_path / "c.db")
    embed_cached(["x"], cache=cache, model="a", progress=False)
    embed_cached(["x"], cache=cache, model="b", progress=False)
    assert len(calls) == 2                                      # a different model must not reuse vectors
