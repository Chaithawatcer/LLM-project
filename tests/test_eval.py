"""Tests for the evaluation code itself: a wrong metric makes every number in the report wrong."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from eval import run_eval
from eval.metrics import (answer_has_all, balanced_accuracy, has_evidence, retrieval_scores, rows_match)


def test_evidence_matching_ignores_case_whitespace_and_typography():
    chunk = "The enterprise\nmonitors   the’ integrity of ALL owned assets"
    assert has_evidence(chunk, ["monitors the' integrity of all owned assets"])
    assert not has_evidence(chunk, ["monitors the integrity of unowned assets"])


def test_retrieval_scores_hit_mrr_and_doc_recall():
    ranked = [("a", "nothing"), ("b", "still nothing"), ("c", "the answer is here")]
    s = retrieval_scores(ranked, ["the answer is here"], ["c", "d"], k=3)
    assert s == {"hit": 1.0, "mrr": pytest.approx(1 / 3), "doc_recall": 0.5}
    assert retrieval_scores(ranked, ["the answer is here"], ["c"], k=2)["hit"] == 0.0  # outside the top-k


def test_doc_recall_cannot_tell_chunkings_apart_but_chunk_hit_can():
    """Why the metric is chunk-level: a window that cut the answer in half still 'retrieves the right PDF'."""
    cut = [("sp800-40r4", "...requires federal agencies to remediate new vulner"), ("sp800-40r4", "abilities added to the catalog within two weeks")]
    whole = [("sp800-40r4", "CISA requires federal agencies to remediate new vulnerabilities added to the catalog within two weeks.")]
    ev = ["requires federal agencies to remediate new vulnerabilities added to the catalog within two weeks"]
    assert retrieval_scores(cut, ev, ["sp800-40r4"], 5)["doc_recall"] == retrieval_scores(whole, ev, ["sp800-40r4"], 5)["doc_recall"] == 1.0
    assert retrieval_scores(cut, ev, ["sp800-40r4"], 5)["hit"] == 0.0
    assert retrieval_scores(whole, ev, ["sp800-40r4"], 5)["hit"] == 1.0


def test_rows_match_is_about_values_not_names_or_order():
    assert rows_match([(390,)], [(390,)])
    assert rows_match([("Cisco", 101), ("Microsoft", 390)], [("Microsoft", 390), ("Cisco", 101)])           # order ignored
    assert not rows_match([("Cisco", 101), ("Microsoft", 390)], [("Microsoft", 390), ("Cisco", 101)], ordered=True)
    assert rows_match([(20.6000001,)], [(20.6,)])                                                           # float noise
    assert not rows_match([(220,)], [(112,)])
    assert not rows_match([(1, 2)], [(1,)])                                                                 # shape matters


def test_answer_has_all_handles_thousands_separators_and_case():
    assert answer_has_all("There are 1,754 entries in total.", ["1754"])
    assert answer_has_all("MICROSOFT leads with 117.", ["Microsoft", "117"])
    assert not answer_has_all("About 390.", ["391"])


def test_balanced_accuracy_is_robust_to_group_size():
    assert balanced_accuracy([True] * 9 + [False], [True]) == pytest.approx((0.9 + 1.0) / 2)


# ---- score_item: the per-question verdict ------------------------------------------------

def _final(**kw):
    base = dict(answer="", tools_used=[], refused=False, steps=1, stopped="answered", grounded=True,
                invalid_citations=[], sql=[], citations=[])
    return {**base, **kw}


def _ctx(sources=None, sql_log=None):
    return SimpleNamespace(sources=sources or {}, sql_log=sql_log or [])


SQL_ITEM = {"id": "s", "type": "sql", "question": "q", "gold_sql": "", "answer_contains": ["390"], "expect_tools": ["query_kev"]}
DOC_ITEM = {"id": "d", "type": "doc", "question": "q", "evidence": ["emergency patching"], "expect_tools": ["search_docs"]}


def test_sql_item_needs_both_the_right_result_and_the_right_answer():
    ctx = _ctx(sql_log=[{"ok": True, "rows": [(390,)]}])
    good = run_eval.score_item(SQL_ITEM, _final(answer="390", tools_used=["query_kev"]), ctx, [(390,)])
    assert good["correct"] and good["exec_match"] and good["routing_ok"]
    wrong_answer = run_eval.score_item(SQL_ITEM, _final(answer="about 400", tools_used=["query_kev"]), ctx, [(390,)])
    assert not wrong_answer["correct"] and wrong_answer["exec_match"]            # right query, wrong words
    lucky_guess = run_eval.score_item(SQL_ITEM, _final(answer="390", tools_used=[]), _ctx(), [(390,)])
    assert not lucky_guess["correct"]                                            # right number, but never ran SQL


def test_doc_item_requires_evidence_to_have_reached_the_model():
    hit = _ctx(sources={"k": {"n": 1, "text": "Plan for EMERGENCY  patching quickly"}})
    assert run_eval.score_item(DOC_ITEM, _final(answer="x", tools_used=["search_docs"]), hit, None)["correct"]
    miss = _ctx(sources={"k": {"n": 1, "text": "unrelated text"}})
    assert not run_eval.score_item(DOC_ITEM, _final(answer="x", tools_used=["search_docs"]), miss, None)["correct"]
    refused = run_eval.score_item(DOC_ITEM, _final(answer="no", refused=True, tools_used=["search_docs"]), hit, None)
    assert not refused["correct"]                                                # refusing an answerable question is a miss


def test_out_of_scope_item_is_correct_only_when_refused():
    item = {"id": "o", "type": "oos", "question": "q", "expect_tools": []}
    assert run_eval.score_item(item, _final(refused=True), _ctx(), None)["correct"]
    assert not run_eval.score_item(item, _final(answer="Paris"), _ctx(), None)["correct"]


# ---- the committed dataset itself --------------------------------------------------------

def test_committed_dataset_is_sound(capsys):
    """Gold SQL runs and every evidence phrase exists in the document it names."""
    assert run_eval.cmd_check(SimpleNamespace()) == 0, capsys.readouterr().out
