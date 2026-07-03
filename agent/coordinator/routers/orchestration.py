"""Session orchestration REST API — phase and policy snapshots."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from agent.coordinator.auth import get_current_user
from agent.coordinator.routers.ingest import _resolve_request_subject
from agent.coordinator.viewport import viewport_manager
from shared.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/session", tags=["orchestration"])


class OrchestrationEventRequest(BaseModel):
    type: str = Field(..., description="ADVANCE | USER_SET_PHASE | TOOL_COMPLETION")
    phase: str | None = None
    tool: str | None = None


@router.get("/{session_id}/orchestration")
async def get_orchestration_snapshot(
    session_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """Return the current SessionOrchestrator snapshot for a session."""
    from agent.orchestration.session_store import get_session_orchestrator

    _resolve_request_subject(current_user)
    orchestrator = await get_session_orchestrator(session_id)
    return orchestrator.get_state_snapshot()


@router.post("/{session_id}/orchestration/event")
async def post_orchestration_event(
    session_id: str,
    request: OrchestrationEventRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """Apply a discovery phase transition and push snapshot to connected clients."""
    from agent.orchestration.session_store import get_session_orchestrator

    _resolve_request_subject(current_user)
    orchestrator = await get_session_orchestrator(session_id)

    event: dict[str, Any] = {"type": request.type}
    if request.phase is not None:
        event["phase"] = request.phase
    if request.tool is not None:
        event["tool"] = request.tool

    changed = orchestrator.transition_discovery(event)
    snapshot = orchestrator.get_state_snapshot()

    if changed:
        await viewport_manager.send_to_session(
            session_id,
            {
                "type": "phase_transition",
                "phase": snapshot["discovery_phase"],
                "source": "api",
                "snapshot": _public_snapshot(snapshot),
            },
        )

    return {
        "changed": changed,
        "snapshot": snapshot,
    }


async def push_orchestration_snapshot(session_id: str) -> None:
    """Push orchestrator snapshot to all viewport connections for a session."""
    from agent.orchestration.session_store import get_session_orchestrator

    orchestrator = await get_session_orchestrator(session_id)
    snapshot = orchestrator.get_state_snapshot()
    await viewport_manager.send_to_session(
        session_id,
        {
            "type": "state_snapshot",
            "snapshot": _public_snapshot(snapshot),
        },
    )


def _public_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Shape snapshot for frontend WebSocket consumers."""
    policy = snapshot.get("policy") or {}
    return {
        "session_id": snapshot.get("session_id"),
        "discovery_phase": snapshot.get("discovery_phase"),
        "hypothesis_lifecycle": snapshot.get("hypothesis_lifecycle"),
        "structure_scope": snapshot.get("structure_scope"),
        "selected_residue": snapshot.get("selected_residue"),
        "policy": policy,
        "discovery_phase_state": snapshot.get("discovery_phase_state"),
        "hypothesis_state": snapshot.get("hypothesis_state"),
    }
