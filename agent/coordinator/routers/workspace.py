"""Workspace layout persistence for the Dockview workbench."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from psycopg.types.json import Json
from pydantic import BaseModel, Field

from agent.coordinator.auth import get_current_user
from agent.coordinator.deps import get_db
from agent.coordinator.routers.ingest import _resolve_request_subject
from shared.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/session", tags=["workspace"])

VALID_PHASE_GROUPS = frozenset({"exploration", "analysis", "review"})


class WorkspaceLayoutPayload(BaseModel):
    workspace_id: str = Field(default="default")
    layout_json: dict[str, Any] = Field(default_factory=dict)
    active_phase_group: str = Field(default="exploration")


@router.get("/{session_id}/workspace-layout")
async def get_workspace_layout(
    session_id: str,
    db=Depends(get_db),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """Load persisted Dockview layout for a session."""
    user_id = _resolve_request_subject(current_user)
    row = await db.fetch_one(
        """
        SELECT workspace_id, layout_json, active_phase_group, updated_at
        FROM agent_workspace_layout
        WHERE session_id = :session_id AND user_id = :user_id
        """,
        {"session_id": session_id, "user_id": user_id},
    )
    if not row:
        return {
            "session_id": session_id,
            "workspace_id": "default",
            "layout_json": {},
            "active_phase_group": "exploration",
            "updated_at": None,
        }
    return {
        "session_id": session_id,
        "workspace_id": row["workspace_id"],
        "layout_json": row["layout_json"],
        "active_phase_group": row["active_phase_group"],
        "updated_at": row["updated_at"],
    }


@router.put("/{session_id}/workspace-layout")
async def put_workspace_layout(
    session_id: str,
    request: WorkspaceLayoutPayload,
    db=Depends(get_db),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """Upsert Dockview layout snapshot for a session."""
    user_id = _resolve_request_subject(current_user)
    if request.active_phase_group not in VALID_PHASE_GROUPS:
        return JSONResponse(
            status_code=400,
            content={
                "error": "invalid_phase_group",
                "message": f"active_phase_group must be one of {sorted(VALID_PHASE_GROUPS)}",
            },
        )

    await db.execute(
        """
        INSERT INTO agent_workspace_layout (
            session_id, user_id, workspace_id, layout_json, active_phase_group, updated_at
        ) VALUES (
            :session_id, :user_id, :workspace_id, :layout_json, :active_phase_group, NOW()
        )
        ON CONFLICT (session_id) DO UPDATE SET
            user_id = EXCLUDED.user_id,
            workspace_id = EXCLUDED.workspace_id,
            layout_json = EXCLUDED.layout_json,
            active_phase_group = EXCLUDED.active_phase_group,
            updated_at = NOW()
        """,
        {
            "session_id": session_id,
            "user_id": user_id,
            "workspace_id": request.workspace_id,
            "layout_json": Json(request.layout_json),
            "active_phase_group": request.active_phase_group,
        },
    )
    await db.commit()

    return {
        "session_id": session_id,
        "workspace_id": request.workspace_id,
        "active_phase_group": request.active_phase_group,
        "saved": True,
    }
