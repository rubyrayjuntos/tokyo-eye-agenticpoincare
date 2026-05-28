"""Authentication and session management.

JWT token creation/verification, user extraction dependency,
and bounded session store with TTL eviction.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from collections import OrderedDict
from typing import Any

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JWT Configuration
# ---------------------------------------------------------------------------

ENVIRONMENT = os.getenv("ENVIRONMENT", "dev")

JWT_SECRET: str | None = os.getenv("JWT_SECRET")
if not JWT_SECRET:
    if ENVIRONMENT == "prod":
        raise RuntimeError(
            "JWT_SECRET environment variable must be set in production. "
            "Refusing to start with a default secret."
        )
    JWT_SECRET = "dev-secret-change-in-production"
    logger.warning("Using default JWT_SECRET — acceptable for local dev only")

JWT_ALGORITHM = "HS256"

security = HTTPBearer(auto_error=False)


# ---------------------------------------------------------------------------
# Token Operations
# ---------------------------------------------------------------------------


def create_token(user_id: str, session_id: str) -> str:
    """Create a JWT token for a user session."""
    payload = {
        "sub": user_id,
        "session_id": session_id,
        "iat": int(time.time()),
        "exp": int(time.time()) + 86400 * 7,  # 7 days
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_token(token: str) -> dict | None:
    """Verify and decode a JWT token."""
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except JWTError:
        return None


# ---------------------------------------------------------------------------
# User Dependency
# ---------------------------------------------------------------------------


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> dict:
    """FastAPI dependency: extract current user from JWT.

    Auth is required by default. Set REQUIRE_AUTH=false to allow
    anonymous access (local dev only).
    """
    if credentials is None:
        # Auth is required by default; opt-out only for local dev
        if os.getenv("REQUIRE_AUTH", "true").lower() == "false":
            return {"sub": "anonymous", "session_id": str(uuid.uuid4())}
        raise HTTPException(status_code=401, detail="Not authenticated")

    payload = verify_token(credentials.credentials)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid token")
    return payload


# ---------------------------------------------------------------------------
# Session Store (bounded LRU with TTL)
# ---------------------------------------------------------------------------

_SESSION_MAX_SIZE = int(os.getenv("SESSION_MAX_SIZE", "10000"))
_SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "86400"))  # 24h


class SessionStore:
    """Bounded in-memory session store with LRU eviction and TTL.

    - Max size prevents unbounded memory growth from anonymous sessions.
    - TTL ensures stale sessions are eventually cleaned up.
    - For production, replace with DynamoDB or Redis.
    """

    def __init__(self, max_size: int = _SESSION_MAX_SIZE, ttl_seconds: int = _SESSION_TTL_SECONDS):
        self._sessions: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._timestamps: dict[str, float] = {}
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds

    def get(self, session_id: str) -> dict[str, Any]:
        """Get or create a session. Moves to end (most recently used)."""
        self._evict_expired()
        if session_id in self._sessions:
            self._sessions.move_to_end(session_id)
            return self._sessions[session_id]
        # Create new session
        return self._create(session_id)

    def set(self, session_id: str, data: dict[str, Any]) -> None:
        self._evict_if_full()
        self._sessions[session_id] = data
        self._sessions.move_to_end(session_id)
        self._timestamps[session_id] = time.time()

    def update(self, session_id: str, key: str, value: Any) -> None:
        session = self.get(session_id)
        session[key] = value
        self._timestamps[session_id] = time.time()

    def _create(self, session_id: str) -> dict[str, Any]:
        self._evict_if_full()
        data: dict[str, Any] = {}
        self._sessions[session_id] = data
        self._timestamps[session_id] = time.time()
        return data

    def _evict_if_full(self) -> None:
        """Evict oldest sessions if at capacity."""
        while len(self._sessions) >= self._max_size:
            oldest_id, _ = self._sessions.popitem(last=False)
            self._timestamps.pop(oldest_id, None)

    def _evict_expired(self) -> None:
        """Remove sessions older than TTL (lazy, checks up to 10 at a time)."""
        now = time.time()
        to_remove: list[str] = []
        for sid in list(self._sessions.keys())[:10]:
            if now - self._timestamps.get(sid, 0) > self._ttl_seconds:
                to_remove.append(sid)
            else:
                break  # OrderedDict is ordered by insertion/access
        for sid in to_remove:
            del self._sessions[sid]
            self._timestamps.pop(sid, None)

    @property
    def size(self) -> int:
        return len(self._sessions)


session_store = SessionStore()
