# Backend Orchestration Layer — Tool Gating
"""Wraps tool invocation to enforce phase-aware policy.

The gated_tool_invoke() function is the single enforcement point:
1. Check orchestrator.can_invoke_tool() before execution
2. Return structured ToolResult error if blocked (includes phase + unlock hint)
3. On success, call orchestrator.handle_tool_completion() to fire state transitions
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Awaitable

from agent.llm.base import ToolCall, ToolResult
from agent.orchestration.orchestrator import SessionOrchestrator

logger = logging.getLogger(__name__)


async def gated_tool_invoke(
    orchestrator: SessionOrchestrator,
    call: ToolCall,
    tool_handler: Callable[..., Awaitable[Any]],
) -> ToolResult:
    """Invoke a tool only if the current orchestrator policy allows it.

    Args:
        orchestrator: The session orchestrator that enforces policy.
        call: The ToolCall containing name, arguments, and id.
        tool_handler: The async handler function for the tool.

    Returns:
        ToolResult — either the successful tool output, or a structured
        error explaining why the tool is blocked and how to unlock it.
    """
    allowed, reason = orchestrator.can_invoke_tool(call.name)

    if not allowed:
        logger.info(
            "Tool gated: tool=%s phase=%s session=%s",
            call.name,
            orchestrator.discovery_phase.value,
            orchestrator.session_id,
        )
        error_payload = {
            "error": f"Tool '{call.name}' is not available in the current "
                     f"discovery phase ('{orchestrator.discovery_phase.value}').",
            "phase": orchestrator.discovery_phase.value,
            "reason": reason or "Tool is blocked at this stage.",
            "hint": reason or "",
        }
        return ToolResult(
            tool_call_id=call.id,
            content=json.dumps(error_payload),
            is_error=True,
        )

    # Tool is allowed — execute the handler
    try:
        result = await tool_handler(**call.arguments)

        # Convert result to structured data and string content
        if hasattr(result, "model_dump"):
            structured = result.model_dump(mode="python")
        elif hasattr(result, "__dict__"):
            structured = result.__dict__
        else:
            structured = result if isinstance(result, dict) else {"value": result}

        content = json.dumps(structured, default=str)

        # Notify orchestrator of successful tool completion
        orchestrator.handle_tool_completion(call.name, result)

        return ToolResult(
            tool_call_id=call.id,
            content=content,
            structured_data=structured,
        )
    except Exception as e:
        logger.exception("Tool %s failed during gated execution", call.name)
        return ToolResult(
            tool_call_id=call.id,
            content=json.dumps({"error": str(e)}),
            is_error=True,
        )
