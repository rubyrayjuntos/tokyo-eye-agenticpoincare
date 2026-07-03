"""Per-session orchestrator instances for agent tool gating."""

from __future__ import annotations

import asyncio
from typing import Any

from agent.orchestration.orchestrator import SessionOrchestrator

_lock = asyncio.Lock()
_orchestrators: dict[str, SessionOrchestrator] = {}


def apply_orchestration_context(
    orchestrator: SessionOrchestrator,
    context: dict[str, Any] | None,
) -> None:
    """Sync discovery phase from client context without resetting other state."""
    if not context:
        return

    phase: str | None = None
    orch_state = context.get("orchestration_state")
    if isinstance(orch_state, dict):
        raw = orch_state.get("discovery_phase")
        if isinstance(raw, str) and raw:
            phase = raw
    if phase is None:
        raw = context.get("discovery_phase")
        if isinstance(raw, str) and raw:
            phase = raw

    if phase:
        orchestrator.transition_discovery({"type": "USER_SET_PHASE", "phase": phase})


async def get_session_orchestrator(
    session_id: str,
    context: dict[str, Any] | None = None,
) -> SessionOrchestrator:
    """Return the orchestrator for a session, creating it on first use."""
    async with _lock:
        orchestrator = _orchestrators.get(session_id)
        if orchestrator is None:
            orchestrator = SessionOrchestrator(session_id)
            _orchestrators[session_id] = orchestrator

    apply_orchestration_context(orchestrator, context)
    return orchestrator


def clear_session_orchestrator(session_id: str) -> None:
    """Remove a session orchestrator (primarily for tests)."""
    _orchestrators.pop(session_id, None)
