"""Structured JSON logging with automatic correlation ID injection.

In production (ENVIRONMENT=prod), logs are emitted as JSON lines for
ingestion by CloudWatch, Datadog, ELK, etc.

In development, logs use a human-readable format with color.

Usage:
    from shared.logging import get_logger, setup_logging

    # Call once at app startup:
    setup_logging()

    # In any module:
    logger = get_logger(__name__)
    logger.info("Processing structure", structure_id="4obe", phase="gnn")
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any


ENVIRONMENT = os.getenv("ENVIRONMENT", "dev")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


class StructuredFormatter(logging.Formatter):
    """JSON formatter that includes correlation IDs from request context."""

    def format(self, record: logging.LogRecord) -> str:
        from shared.context import get_context

        ctx = get_context()

        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": ctx.request_id,
        }

        # Add context fields
        if ctx.session_id:
            log_entry["session_id"] = ctx.session_id
        if ctx.run_id:
            log_entry["run_id"] = ctx.run_id
        if ctx.user_id:
            log_entry["user_id"] = ctx.user_id

        # Add any extra fields passed via logger.info("msg", extra={...})
        # or via the StructuredLogger's **kwargs
        if hasattr(record, "_structured_data"):
            log_entry.update(record._structured_data)

        # Add exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add source location for errors
        if record.levelno >= logging.ERROR:
            log_entry["source"] = f"{record.pathname}:{record.lineno}"

        return json.dumps(log_entry, default=str)


class DevFormatter(logging.Formatter):
    """Human-readable formatter for local development."""

    def format(self, record: logging.LogRecord) -> str:
        from shared.context import get_context

        ctx = get_context()
        prefix = f"[{ctx.request_id[:12]}]" if ctx.request_id else ""

        # Include structured data if present
        extra = ""
        if hasattr(record, "_structured_data") and record._structured_data:
            pairs = " ".join(f"{k}={v}" for k, v in record._structured_data.items())
            extra = f" | {pairs}"

        return (
            f"{record.levelname:<7} {prefix} {record.name}: "
            f"{record.getMessage()}{extra}"
        )


class StructuredLogger(logging.Logger):
    """Logger subclass that accepts keyword arguments as structured data.

    Usage:
        logger.info("GNN complete", structure_id="4obe", nodes=165, duration_ms=340)
    """

    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False, stacklevel=1, **kwargs):
        # Merge kwargs into a _structured_data attribute on the record
        if extra is None:
            extra = {}
        if kwargs:
            extra["_structured_data"] = kwargs
        super()._log(level, msg, args, exc_info=exc_info, extra=extra, stack_info=stack_info, stacklevel=stacklevel + 1)


# Register our custom logger class
logging.setLoggerClass(StructuredLogger)


def setup_logging() -> None:
    """Configure logging for the application. Call once at startup."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    # Remove existing handlers
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)

    if ENVIRONMENT == "prod":
        handler.setFormatter(StructuredFormatter())
    else:
        handler.setFormatter(DevFormatter())

    root.addHandler(handler)

    # Quiet noisy libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger for a module.

    Args:
        name: Usually __name__ of the calling module.

    Returns:
        A StructuredLogger that accepts keyword arguments.
    """
    return logging.getLogger(name)  # type: ignore[return-value]
