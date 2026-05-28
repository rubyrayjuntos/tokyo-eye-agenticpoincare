"""Shared cross-cutting utilities for Tokyo Eye.

This package provides:
- Structured JSON logging with correlation IDs
- Request context propagation (request_id, session_id, run_id)
- Application constants and configuration
"""

from shared.context import RequestContext, get_context, set_context
from shared.logging import get_logger, setup_logging

__all__ = [
    "RequestContext",
    "get_context",
    "get_logger",
    "set_context",
    "setup_logging",
]
