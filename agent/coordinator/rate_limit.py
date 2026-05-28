"""Simple in-memory rate limiter for expensive endpoints.

Uses a sliding window per session_id. For production, replace with
Redis-backed rate limiting (e.g., via slowapi or a custom middleware).
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import Any

from fastapi import HTTPException, Request

# Configurable via environment
_CHAT_RATE_LIMIT = int(os.getenv("CHAT_RATE_LIMIT_PER_MINUTE", "20"))
_CHAT_WINDOW_SECONDS = 60


class RateLimiter:
    """Sliding window rate limiter keyed by session or IP."""

    def __init__(self, max_requests: int = _CHAT_RATE_LIMIT, window_seconds: int = _CHAT_WINDOW_SECONDS):
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)

    def check(self, key: str) -> None:
        """Check if the key is within rate limits. Raises HTTPException if exceeded."""
        now = time.time()
        window_start = now - self._window_seconds

        # Clean old entries
        timestamps = self._requests[key]
        self._requests[key] = [t for t in timestamps if t > window_start]

        if len(self._requests[key]) >= self._max_requests:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Max {self._max_requests} requests per {self._window_seconds}s.",
            )

        self._requests[key].append(now)


# Singleton for the chat endpoint
chat_rate_limiter = RateLimiter()
