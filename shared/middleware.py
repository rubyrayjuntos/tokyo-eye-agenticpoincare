"""FastAPI middleware for request context propagation.

Automatically sets up a RequestContext for each incoming request,
extracting session_id from the JWT token and generating a unique
request_id for tracing.
"""

from __future__ import annotations

import time
import uuid

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from shared.context import RequestContext, reset_context, set_context
from shared.logging import get_logger

logger = get_logger(__name__)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Middleware that establishes request context for every HTTP request.

    Sets:
    - request_id: from X-Request-ID header or auto-generated
    - session_id: extracted from JWT if present
    - user_id: extracted from JWT if present

    Also logs request start/end with timing.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Extract or generate request_id
        request_id = request.headers.get("X-Request-ID", f"req_{uuid.uuid4().hex[:12]}")

        # Try to extract session/user from auth header (best-effort, non-blocking)
        session_id = None
        user_id = None
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            try:
                from agent.coordinator.auth import verify_token
                payload = verify_token(auth_header[7:])
                if payload:
                    session_id = payload.get("session_id")
                    user_id = payload.get("sub")
            except Exception:
                pass

        # Set context for this request
        ctx = RequestContext(
            request_id=request_id,
            session_id=session_id,
            user_id=user_id,
        )
        set_context(ctx)

        # Log request start
        start_time = time.monotonic()
        logger.info(
            "Request started",
            method=request.method,
            path=request.url.path,
        )

        try:
            response = await call_next(request)

            # Add request_id to response headers for client-side correlation
            response.headers["X-Request-ID"] = request_id

            duration_ms = int((time.monotonic() - start_time) * 1000)
            logger.info(
                "Request completed",
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                duration_ms=duration_ms,
            )

            return response

        except Exception as e:
            duration_ms = int((time.monotonic() - start_time) * 1000)
            logger.error(
                "Request failed",
                method=request.method,
                path=request.url.path,
                error=str(e),
                duration_ms=duration_ms,
            )
            raise
        finally:
            reset_context()
