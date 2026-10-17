"""Small in-memory rate limiter suitable for a single local-demo process."""
from __future__ import annotations

import time
from collections import defaultdict, deque
from threading import Lock

from fastapi import Request

from app.core.errors import ApiError


class InMemoryRateLimiter:
    """Limit calls per client IP in a sliding one-minute window.

    This is intentionally local-process only. A production multi-instance API
    would use a shared store such as Redis or an API gateway.
    """

    def __init__(self, limit_per_minute: int):
        self._limit = limit_per_minute
        self._calls: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def check(self, key: str) -> None:
        now = time.monotonic()
        cutoff = now - 60
        with self._lock:
            calls = self._calls[key]
            while calls and calls[0] <= cutoff:
                calls.popleft()
            if len(calls) >= self._limit:
                raise ApiError(429, "RATE_LIMITED", "Too many requests; try again later")
            calls.append(now)


def require_rate_limit(request: Request) -> None:
    client_host = request.client.host if request.client else "unknown"
    request.app.state.rate_limiter.check(client_host)
