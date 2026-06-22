"""Chat endpoint — LLM agent interaction with viewport directives."""

from __future__ import annotations

import logging
import os
import re
import uuid
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from agent.coordinator.auth import get_current_user, session_store
from agent.coordinator.rate_limit import chat_rate_limiter
from agent.coordinator.viewport import viewport_manager
from agent.models.viewport import (
    DirectiveAction,
    HighlightGroup,
    HighlightStyle,
    ViewportDirective,
    ViewportState,
)
from agent.tools.dtie.tools import ToolResult

logger = logging.getLogger(__name__)

ENVIRONMENT = os.getenv("ENVIRONMENT", "dev")

router = APIRouter(prefix="/api", tags=["chat"])


# ---------------------------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    viewport_state: ViewportState | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    viewport_directives: list[dict] | None = None
    tool_results: list[dict] | None = None


# ---------------------------------------------------------------------------
# Chat Endpoint
# ---------------------------------------------------------------------------


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, user: dict = Depends(get_current_user)):
    """Process a natural language message and execute tools as needed.

    The chat handler:
    1. Applies rate limiting per session
    2. Builds context from viewport state
    3. Routes to the LLM agent (or tool dispatch for structured requests)
    4. Parses viewport directives from the response
    5. Auto-highlights referenced residues
    6. Pushes directives over WebSocket to connected viewers
    """
    session_id = request.session_id or user.get("session_id", str(uuid.uuid4()))

    # Rate limit per session
    chat_rate_limiter.check(session_id)

    # Build context-enriched message
    agent_message = _build_agent_message(request, session_id)

    # Execute via LLM agent
    response_text, tool_results, directives = await _execute_agent(
        agent_message, session_id, request
    )

    # Auto-highlight referenced residues
    referenced = _extract_residue_references(response_text)
    if referenced:
        directives.append(ViewportDirective(
            action=DirectiveAction.HIGHLIGHT,
            structure_id=request.viewport_state.structure_id if request.viewport_state else None,
            highlight_groups=[HighlightGroup(
                residue_ids=referenced,
                color="#4ecdc4",
                style=HighlightStyle.GLOW,
                label="Referenced",
            )],
        ))

    # Push directives to connected viewers
    if directives and session_id:
        for d in directives:
            directive_payload = d.model_dump(mode="json")
            await viewport_manager.send_to_session(
                session_id,
                {"type": "semantic_command", **directive_payload},
            )

    return ChatResponse(
        response=response_text,
        session_id=session_id,
        viewport_directives=[d.model_dump(mode="json") for d in directives] if directives else None,
        tool_results=[r for r in tool_results] if tool_results else None,
    )


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


async def _execute_agent(
    message: str, session_id: str, request: ChatRequest
) -> tuple[str, list[dict], list[ViewportDirective]]:
    """Execute the LLM agent — full tool-calling loop."""
    from agent.llm.agents import create_coordinator
    from agent.llm.providers import get_provider
    from data.db import DBAdapter, get_connection

    directives: list[ViewportDirective] = []
    tool_results: list[dict] = []

    # Build context for the agent
    context: dict[str, Any] = {"session_id": session_id}
    if request.viewport_state:
        context["viewport"] = {
            "structure_id": request.viewport_state.structure_id,
            "current_metric": request.viewport_state.current_metric,
            "curvature": request.viewport_state.curvature,
        }

    try:
        async with get_connection() as conn:
            db = DBAdapter(conn)
            llm = get_provider()
            coordinator = create_coordinator(llm=llm, db=db)
            response = await coordinator.run(message, context=context)

        # Convert viewport directives from agent response
        for d in response.viewport_directives:
            try:
                directives.append(ViewportDirective(**d))
            except Exception:
                pass

        tool_results = [{"tools_called": response.tool_calls_made}] if response.tool_calls_made else []

        return response.text, tool_results, directives

    except Exception as e:
        logger.exception("Agent execution failed for session %s", session_id)
        # Never leak internal error details to the client in production
        if ENVIRONMENT == "prod":
            return "An internal error occurred. Please try again.", [], []
        return f"Agent error: {type(e).__name__}: {e}", [], []


def _build_agent_message(request: ChatRequest, session_id: str) -> str:
    """Build context-enriched message including viewport state."""
    if request.viewport_state is None:
        return request.message

    vs = request.viewport_state
    context = (
        f"\n[Viewport: {vs.structure_id}, metric={vs.current_metric}, "
        f"curvature={vs.curvature}]\n"
    )
    return f"{request.message}{context}"


def _extract_residue_references(text: str) -> list[str]:
    """Extract residue_id references from agent response text."""
    pattern = r"\b[a-z0-9_]+:[A-Z]:\d+\b"
    return re.findall(pattern, text)
