import datetime as dt
import json

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from app.main import app
from app.ratelimit import RateLimiter, client_ip
from tests.conftest import ScriptedChat, assistant


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_per_minute_limit_blocks_then_recovers_after_the_window():
    clock = Clock()
    rl = RateLimiter(per_minute=3, per_day=0, clock=clock)
    assert [rl.check("a") for _ in range(3)] == [None, None, None]
    assert "3 per minute" in rl.check("a")
    clock.now += 61
    assert rl.check("a") is None


def test_clients_are_limited_independently():
    rl = RateLimiter(per_minute=1, per_day=0, clock=Clock())
    assert rl.check("a") is None and rl.check("b") is None
    assert rl.check("a") is not None


def test_daily_cap_is_global_and_resets_the_next_day():
    day = {"d": dt.date(2026, 10, 23)}
    rl = RateLimiter(per_minute=0, per_day=2, clock=Clock(), today=lambda: day["d"])
    assert rl.check("a") is None and rl.check("b") is None
    assert "daily" in rl.check("c")
    day["d"] = dt.date(2026, 10, 24)
    assert rl.check("c") is None


def test_zero_disables_a_ceiling():
    rl = RateLimiter(per_minute=0, per_day=0, clock=Clock())
    assert all(rl.check("a") is None for _ in range(500))


def test_a_refused_request_does_not_use_up_the_daily_budget():
    rl = RateLimiter(per_minute=1, per_day=3, clock=Clock())
    assert rl.check("a") is None
    assert all(rl.check("a") is not None for _ in range(10))   # refused by the per-minute ceiling
    assert rl.check("b") is None and rl.check("c") is None     # the day still has room: 1 + 2 = 3 counted


def _request(headers: dict[str, str], peer: str = "10.0.0.9") -> Request:
    return Request({"type": "http", "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
                    "client": (peer, 1234)})


def test_forwarded_header_is_only_trusted_behind_a_proxy(monkeypatch):
    req = _request({"X-Forwarded-For": "203.0.113.7, 10.1.1.1"})
    monkeypatch.delenv("TRUST_PROXY", raising=False)
    assert client_ip(req) == "10.0.0.9"                       # exposed directly: a forged header must not matter
    monkeypatch.setenv("TRUST_PROXY", "1")
    assert client_ip(req) == "203.0.113.7"
    assert client_ip(_request({})) == "10.0.0.9"              # trusted but absent: fall back to the peer


def test_api_refuses_with_429_before_any_model_call(runtime):
    calls = []

    def chat(messages, **kw):
        calls.append(1)
        return assistant("ok")

    app.state.runtime, app.state.chat_fn = runtime, chat
    app.state.limiter = RateLimiter(per_minute=1, per_day=0)
    client = TestClient(app)
    first = client.post("/api/ask", json={"question": "How many entries?"})
    assert first.status_code == 200 and len(calls) == 1
    second = client.post("/api/ask", json={"question": "How many entries?"})
    assert second.status_code == 429 and "per minute" in second.json()["detail"]
    assert second.headers["retry-after"] == "60" and len(calls) == 1   # the refused one cost nothing
    app.state.limiter = None
