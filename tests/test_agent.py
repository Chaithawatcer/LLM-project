import json
from dataclasses import replace

import pytest

from app.agent import REFUSAL, is_refusal, run_agent
from app.tools import dispatch
from tests.conftest import ScriptedChat, assistant, tool_call


def events(runtime, chat, question="q?", mode="agent"):
    return list(run_agent(question, runtime.new_context(), mode=mode, snapshot="2026-10-20", chat_fn=chat))


# ---- dispatch: untrusted input from the model ---------------------------------------

def test_dispatch_never_raises_and_explains(runtime):
    ctx = runtime.new_context()
    assert dispatch(ctx, "rm_rf", "{}").startswith("error: unknown tool")
    assert "not valid JSON" in dispatch(ctx, "query_kev", "{sql: broken")
    assert "must be a JSON object" in dispatch(ctx, "query_kev", "[1,2]")
    assert "invalid arguments" in dispatch(ctx, "query_kev", json.dumps({"sql": "x"}))
    assert "invalid arguments" in dispatch(ctx, "search_docs", json.dumps({"q": "typo"}))
    assert dispatch(ctx, "query_kev", json.dumps({"sql": "DROP TABLE kev"})).startswith("error:")


def test_a_disabled_tool_is_unknown(runtime):
    ctx = runtime.new_context()
    ctx.enabled = frozenset({"search_docs"})
    assert "unknown tool" in dispatch(ctx, "query_kev", json.dumps({"sql": "SELECT 1 FROM kev"}))


def test_search_results_are_fenced_numbered_and_recorded(runtime):
    ctx = runtime.new_context()
    out = dispatch(ctx, "search_docs", json.dumps({"query": "emergency patching"}))
    assert out.startswith('<tool_result tool="search_docs" trust="untrusted-data">') and "[1]" in out
    assert ctx.source_by_n(1)["doc"] == "patch"
    n_before = len(ctx.sources)
    dispatch(ctx, "search_docs", json.dumps({"query": "emergency patching"}))
    assert len(ctx.sources) == n_before  # the same chunk keeps the same [n]


def test_floor_gives_a_refusable_observation(runtime):
    runtime.settings = replace(runtime.settings, min_score=0.999)
    ctx = runtime.new_context()
    assert "NO_RELEVANT_PASSAGES" in dispatch(ctx, "search_docs", json.dumps({"query": "emergency patching"}))


# ---- the loop ---------------------------------------------------------------------------

def test_tool_then_answer_with_a_verified_citation(runtime):
    chat = ScriptedChat(
        assistant("", [tool_call("search_docs", json.dumps({"query": "emergency patching"}))]),
        assistant("Patch exploited flaws urgently [1]."),
    )
    ev = events(runtime, chat)
    assert [e["type"] for e in ev] == ["act", "obs", "final"]
    final = ev[-1]
    assert final["grounded"] and final["tools_used"] == ["search_docs"]
    assert final["citations"][0]["doc"] == "patch" and final["invalid_citations"] == []
    # the tool result went back as a role:"tool" message tied to the call id
    assert chat.sent[1][-1]["role"] == "tool" and chat.sent[1][-1]["tool_call_id"] == "c1"


def test_a_citation_the_model_invents_is_flagged_not_trusted(runtime):
    final = events(runtime, ScriptedChat(assistant("According to [7], do X.")))[-1]
    assert final["invalid_citations"] == [7] and final["citations"] == [] and not final["grounded"]


def test_sql_tool_result_and_log(runtime):
    chat = ScriptedChat(
        assistant("", [tool_call("query_kev", json.dumps({"sql": "SELECT COUNT(*) FROM kev WHERE vendor='Microsoft'"}))]),
        assistant("There are 390."),
    )
    final = events(runtime, chat)[-1]
    assert final["sql"][0]["ok"] and final["sql"][0]["n_rows"] == 1 and final["tools_used"] == ["query_kev"]


def test_duplicate_call_is_not_run_twice(runtime):
    call = json.dumps({"sql": "SELECT COUNT(*) FROM kev"})
    chat = ScriptedChat(
        assistant("", [tool_call("query_kev", call, "a")]),
        assistant("", [tool_call("query_kev", call, "b")]),
        assistant("done"),
    )
    obs = [e["preview"] for e in events(runtime, chat) if e["type"] == "obs"]
    assert obs[1].startswith("error: you already made this exact call")


def test_parallel_calls_each_get_a_tool_message(runtime):
    chat = ScriptedChat(
        assistant("", [tool_call("search_docs", json.dumps({"query": "log retention"}), "p1"),
                       tool_call("query_kev", json.dumps({"sql": "SELECT COUNT(*) FROM kev"}), "p2")]),
        assistant("both done"),
    )
    final = events(runtime, chat)[-1]
    answered = {m.get("tool_call_id") for m in chat.sent[1] if m["role"] == "tool"}
    assert answered == {"p1", "p2"} and final["tools_used"] == ["search_docs", "query_kev"]


def test_step_cap_forces_a_final_answer(runtime):
    n = {"i": 0}

    def never_stops(messages, **kw):
        n["i"] += 1
        if n["i"] <= runtime.settings.max_steps:  # distinct args each time so the duplicate guard doesn't mask the cap
            return assistant("", [tool_call("search_docs", json.dumps({"query": f"patch {n['i']}"}), f"id{n['i']}")])
        return assistant("Best I can say: patch fast [1].")

    final = events(runtime, never_stops)[-1]
    assert final["stopped"] == "max_steps" and final["answer"].startswith("Best I can say")


def test_refusal_sentence_is_detected(runtime):
    final = events(runtime, ScriptedChat(assistant(REFUSAL + " It is not about security.")))[-1]
    assert final["refused"] and not final["grounded"]


@pytest.mark.parametrize("text", [
    "I can't answer that from the sources I have. It is not about security.",
    "I cannot answer that from the sources I have. I am restricted to the tools.",   # seen from the real model
    "I can’t answer this from the provided sources.",
])
def test_refusal_is_recognised_by_meaning_not_exact_wording(text):
    assert is_refusal(text)


@pytest.mark.parametrize("text", [
    "Microsoft has 390 entries.",
    "NIST says you can answer that question by checking sources [1].",
    "",
])
def test_ordinary_answers_are_not_refusals(text):
    assert not is_refusal(text)


def test_modes_restrict_the_tools_the_model_is_offered(runtime):
    seen = {}

    def spy(messages, *, tools=None, **kw):
        seen["tools"] = tools
        return assistant("ok")

    events(runtime, spy, mode="rag_only")
    assert [t["function"]["name"] for t in seen["tools"]] == ["search_docs"]
    events(runtime, spy, mode="no_tools")
    assert seen["tools"] is None
