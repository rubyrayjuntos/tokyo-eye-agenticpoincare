"""Shared ingest authorization helpers for coordinator routes."""

from __future__ import annotations

import os
from typing import Any

from fastapi import HTTPException

from agent.coordinator.auth import get_current_user


def _resolve_request_subject(current_user: dict[str, Any]) -> str:
    """Resolve the authenticated subject for ingest authorization/audit."""
    for key in ("sub", "user_id", "username", "session_id"):
        value = current_user.get(key)
        if value:
            return str(value)
    return "anonymous"


def _authorize_ingest_request(requester: str) -> None:
    """Optionally restrict ingest to a configured subject allowlist."""
    allowed_subjects_env = os.getenv("SCIENCE_INGEST_ALLOWED_SUBJECTS", "")
    allowed_subjects = {
        subject.strip()
        for subject in allowed_subjects_env.split(",")
        if subject.strip()
    }
    if allowed_subjects and requester not in allowed_subjects:
        raise HTTPException(
            status_code=403,
            detail=f"Ingest not permitted for subject '{requester}'",
        )
