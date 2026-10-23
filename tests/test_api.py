import json

from fastapi.testclient import TestClient

from app.main import app
from tests.conftest import ScriptedChat, assistant, tool_call


def frames(resp):
    return [json.loads(f[6:]) for f in resp.text.split("\n\n") if f.startswith("data: ")]


def test_ask_streams_act_obs_final(runtime):
    app.state.runtime = runtime
    app.state.chat_fn = ScriptedChat(
        assistant("", [tool_call("query_kev", json.dumps({"sql": "SELECT COUNT(*) FROM kev"}))]),
        assistant("1754 entries."),
    )
    r = TestClient(app).post("/api/ask", json={"question": "How many KEV entries are there?"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    ev = frames(r)
    assert [e["type"] for e in ev] == ["act", "obs", "final"] and ev[-1]["answer"] == "1754 entries."


def test_bad_bodies_are_rejected_before_our_code_runs():
    c = TestClient(app)
    assert c.post("/api/ask", json={"question": "hi"}).status_code == 422  # too short
    assert c.post("/api/ask", json={"question": "x" * 2000}).status_code == 422  # too long
    assert c.post("/api/ask", json={"question": "valid question", "mode": "root"}).status_code == 422


def test_missing_key_becomes_an_actionable_error_event(runtime, monkeypatch):
    from app import llm

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(llm, "_client", None)
    app.state.runtime = runtime
    app.state.chat_fn = None  # use the real llm.chat -> LLMNotConfigured
    ev = frames(TestClient(app).post("/api/ask", json={"question": "anything at all"}))
    assert ev[-1]["type"] == "error" and "OPENAI_API_KEY" in ev[-1]["message"]


def test_health_never_returns_the_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-value")
    body = TestClient(app).get("/api/health").json()
    assert body["llm_key_set"] is True and "sk-secret-value" not in json.dumps(body)


def test_index_page_is_served():
    assert "Cyber Advisor" in TestClient(app).get("/").text
