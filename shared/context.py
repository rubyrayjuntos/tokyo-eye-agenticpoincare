"""Request context propagation via contextvars.

Provides a RequestContext that flows through the entire call chain:
HTTP request → agent → LLM → tools → normalizer → DB

Usage:
    from shared.context import get_context, set_context, RequestContext

    # At request entry point (middleware or endpoint):
    ctx = RequestContext(request_id="req_abc", session_id="sess_123")
    set_context(ctx)

    # Anywhere deeper in the call chain:
    ctx = get_context()
    logger.info("Processing", extra={"request_id": ctx.request_id})
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RequestContext:
    """Immutable context that propagates through a single request lifecycle.

    Fields:
        request_id: Unique ID for this HTTP request / WebSocket message
        session_id: User session (persists across requests)
        run_id: Pipeline run ID (set when a pipeline starts)
        user_id: Authenticated user identifier
        extra: Additional key-value pairs for domain-specific context
    """

    request_id: str = field(default_factory=lambda: f"req_{uuid.uuid4().hex[:12]}")
    session_id: str | None = None
    run_id: str | None = None
    user_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def with_run_id(self, run_id: str) -> RequestContext:
        """Return a new context with run_id set (immutable update)."""
        return RequestContext(
            request_id=self.request_id,
            session_id=self.session_id,
            run_id=run_id,
            user_id=self.user_id,
            extra={**self.extra},
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for structured logging."""
        d: dict[str, Any] = {"request_id": self.request_id}
        if self.session_id:
            d["session_id"] = self.session_id
        if self.run_id:
            d["run_id"] = self.run_id
        if self.user_id:
            d["user_id"] = self.user_id
        if self.extra:
            d.update(self.extra)
        return d


# ContextVar for async-safe propagation
_request_context: ContextVar[RequestContext] = ContextVar(
    "request_context", default=RequestContext()
)


def get_context() -> RequestContext:
    """Get the current request context."""
    return _request_context.get()


def set_context(ctx: RequestContext) -> None:
    """Set the request context for the current async task."""
    _request_context.set(ctx)


def reset_context() -> None:
    """Reset to a fresh default context."""
    _request_context.set(RequestContext())
