"""A small in-process rate limiter for the public demo.

Every question costs several model calls on the owner's course key, so an open URL needs two ceilings:
  * per client: at most N questions per minute (stops one visitor or script from hogging the proxy)
  * global: at most M questions per UTC day (bounds the total spend however many clients there are)

In-memory on purpose: a single small instance needs nothing else. With several instances the limits
would be per instance - use Redis or the platform's own limiter then.
"""
from __future__ import annotations

import datetime as dt
import os
import threading
import time
from collections import deque

from fastapi import Request

WINDOW_S = 60.0
MAX_TRACKED_CLIENTS = 5000


class RateLimiter:
    def __init__(self, per_minute: int = 10, per_day: int = 300, *, clock=time.monotonic, today=None) -> None:
        self.per_minute, self.per_day = per_minute, per_day
        self._clock = clock
        self._today = today or (lambda: dt.datetime.now(dt.timezone.utc).date())
        self._hits: dict[str, deque[float]] = {}
        self._day, self._day_count = self._today(), 0
        self._lock = threading.Lock()

    def check(self, client: str) -> str | None:
        """Record a request. Returns None if allowed, else a message that says what to do."""
        with self._lock:
            now = self._clock()
            today = self._today()
            if today != self._day:
                self._day, self._day_count = today, 0
            if self.per_day and self._day_count >= self.per_day:
                return "The demo has reached its daily question limit. Please try again tomorrow."
            if self.per_minute:
                hits = self._hits.setdefault(client, deque())
                while hits and now - hits[0] > WINDOW_S:
                    hits.popleft()
                if len(hits) >= self.per_minute:
                    return f"Too many questions: the limit is {self.per_minute} per minute. Wait a moment and try again."
                hits.append(now)
                if len(self._hits) > MAX_TRACKED_CLIENTS:  # bound memory: drop clients with no recent hits
                    self._hits = {c: h for c, h in self._hits.items() if h and now - h[-1] <= WINDOW_S}
            self._day_count += 1
            return None


def limiter_from_env() -> RateLimiter:
    """RATE_PER_MIN and DAILY_LIMIT; 0 switches that ceiling off."""
    return RateLimiter(int(os.environ.get("RATE_PER_MIN", "10")), int(os.environ.get("DAILY_LIMIT", "300")))


def client_ip(request: Request) -> str:
    """The caller's address. Behind a reverse proxy (TRUST_PROXY=1) the socket peer is the proxy itself,
    so use the first X-Forwarded-For entry. A client can forge that header, which is why the global daily
    ceiling exists as well; never trust it when the app is exposed directly."""
    if os.environ.get("TRUST_PROXY") == "1":
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded.strip():
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"
