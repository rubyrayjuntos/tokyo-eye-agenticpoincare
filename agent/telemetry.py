"""Agent telemetry — captures tool calls, results, timing, and session metadata.

This module provides observability into the agent's behavior that is completely
outside the agent's control. The agent cannot modify, suppress, or influence
what gets recorded here.

Telemetry is emitted as structured log events and optionally stored in-memory
for the current session (returned to the frontend for debugging).

Usage:
    from agent.telemetry import AgentTelemetry

    telemetry = AgentTelemetry(session_id="abc123")
    telemetry.record_request(user_message="...", context={...})
    telemetry.record_tool_call(name="ingest_structure", args={...})
    telemetry.record_tool_result(name="ingest_structure", result={...}, duration_ms=340)
    telemetry.record_response(text="...", tool_calls_made=[...])
    trace = telemetry.finalize()
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from agent.llm.tool_schema import compact_tool_definitions
from shared.logging import get_logger

logger = get_logger(__name__)

_INPUT_COMPONENT_ORDER = (
    "system_prompt",
    "tools_schema",
    "context_block",
    "user_query",
    "assistant_text",
    "assistant_tool_calls",
    "tool_results",
)
_OUTPUT_COMPONENT_ORDER = (
    "assistant_text",
    "assistant_tool_calls",
)


@dataclass
class ToolCallRecord:
    """A single tool call and its result."""

    call_id: str
    tool_name: str
    arguments: dict[str, Any]
    result_summary: str | None = None
    result_keys: list[str] | None = None
    is_error: bool = False
    error_message: str | None = None
    started_at: float = 0.0
    duration_ms: float = 0.0


@dataclass
class AgentTrace:
    """Complete trace of a single agent interaction."""

    trace_id: str
    session_id: str
    user_message: str
    context_snapshot: dict[str, Any]
    tool_calls: list[ToolCallRecord]
    final_response: str
    total_duration_ms: float
    llm_calls: int
    input_tokens: int
    output_tokens: int
    # Key facts from tool results that the agent should have used
    ground_truth_facts: dict[str, Any]
    token_attribution: dict[str, Any]
    llm_call_breakdown: list[dict[str, Any]]


def _serialized_len(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    return len(json.dumps(value, default=str))


def _split_context_from_user_message(user_message: str) -> tuple[int, int]:
    if not user_message.startswith("<context>"):
        return 0, len(user_message)
    end_idx = user_message.find("</context>")
    if end_idx == -1:
        return 0, len(user_message)
    end_idx += len("</context>")
    context_chars = end_idx
    remaining = user_message[end_idx:].lstrip()
    return context_chars, len(remaining)


def _estimate_input_components(
    messages: list[Any],
    system_prompt: str | None,
    tools: list[Any] | None,
) -> dict[str, int]:
    components = {key: 0 for key in _INPUT_COMPONENT_ORDER}
    components["system_prompt"] = len(system_prompt or "")
    if tools:
        components["tools_schema"] = _serialized_len(compact_tool_definitions(tools))

    for idx, msg in enumerate(messages):
        if msg.role == "user":
            context_chars, query_chars = _split_context_from_user_message(msg.content or "")
            if idx == 0:
                components["context_block"] += context_chars
                components["user_query"] += query_chars
            else:
                components["user_query"] += len(msg.content or "")
        elif msg.role == "assistant":
            components["assistant_text"] += len(msg.content or "")
            if getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls or []:
                    components["assistant_tool_calls"] += _serialized_len(tc.arguments) + len(tc.name) + len(tc.id)
        elif msg.role == "tool_result" and getattr(msg, "tool_results", None):
            for tr in msg.tool_results or []:
                components["tool_results"] += len(tr.content or "")

    return components


def _estimate_output_components(response: Any) -> dict[str, int]:
    components = {key: 0 for key in _OUTPUT_COMPONENT_ORDER}
    components["assistant_text"] = len(response.content or "")
    if getattr(response, "tool_calls", None):
        for tc in response.tool_calls or []:
            components["assistant_tool_calls"] += _serialized_len(tc.arguments) + len(tc.name) + len(tc.id)
    return components


def _build_attribution(
    components: dict[str, float],
    total_tokens: int,
    total_chars: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, chars in components.items():
        if chars <= 0:
            continue
        fraction = (chars / total_chars) if total_chars > 0 else 0.0
        result[key] = {
            "chars": int(chars),
            "estimated_tokens": round(total_tokens * fraction, 1),
            "percent": round(fraction * 100, 1),
        }
    return result


def _summarize_result(result: Any, max_len: int = 200) -> str:
    """Create a short summary string of a tool result for logging."""
    if isinstance(result, dict):
        # Extract key scalar values
        summary_parts = []
        for key in ("pdb_id", "structure_id", "residue_count", "status",
                    "count", "error", "db_status", "chains"):
            if key in result:
                val = result[key]
                summary_parts.append(f"{key}={val}")
        if summary_parts:
            return ", ".join(summary_parts)
        return f"dict with keys: {list(result.keys())[:10]}"
    return str(result)[:max_len]


def _extract_ground_truth(tool_name: str, result: dict[str, Any]) -> dict[str, Any]:
    """Extract key factual claims from a tool result.

    These are the values the agent SHOULD report. If its response
    contradicts these, it's hallucinating.
    """
    facts: dict[str, Any] = {}

    if tool_name == "ingest_structure":
        for key in ("pdb_id", "structure_id", "residue_count", "chains",
                    "db_status", "ready_for_gnn"):
            if key in result:
                facts[key] = result[key]
        metadata = result.get("metadata", {})
        if isinstance(metadata, dict):
            for key in ("title", "resolution", "method"):
                if key in metadata:
                    facts[key] = metadata[key]

    elif tool_name == "run_full_pipeline":
        for key in ("status", "structure_id", "phases_completed"):
            if key in result:
                facts[key] = result[key]

    elif tool_name in ("get_source_leaks", "get_high_uncertainty_residues"):
        if "count" in result:
            facts[f"{tool_name}_count"] = result["count"]
        if "source_leaks" in result:
            facts["source_leak_count"] = len(result["source_leaks"])

    elif tool_name == "get_graph_metrics":
        if "metrics" in result and isinstance(result["metrics"], list):
            facts["metric_count"] = len(result["metrics"])

    elif tool_name == "get_hypotheses":
        if "hypotheses" in result:
            facts["hypothesis_count"] = len(result["hypotheses"])

    return facts


class AgentTelemetry:
    """Captures a complete trace of one agent chat interaction.

    Instantiate per-request. Records are outside the agent's control.
    """

    def __init__(self, session_id: str):
        self.trace_id = str(uuid.uuid4())
        self.session_id = session_id
        self._started_at = time.perf_counter()
        self._tool_calls: list[ToolCallRecord] = []
        self._current_tool_start: float = 0.0
        self._user_message: str = ""
        self._context_snapshot: dict[str, Any] = {}
        self._final_response: str = ""
        self._ground_truth: dict[str, Any] = {}
        self._llm_calls: int = 0
        self._input_tokens: int = 0
        self._output_tokens: int = 0
        self._input_component_chars: dict[str, float] = {key: 0.0 for key in _INPUT_COMPONENT_ORDER}
        self._output_component_chars: dict[str, float] = {key: 0.0 for key in _OUTPUT_COMPONENT_ORDER}
        self._llm_call_breakdown: list[dict[str, Any]] = []

    def record_request(self, user_message: str, context: dict[str, Any]) -> None:
        """Record the incoming user request and context."""
        self._user_message = user_message
        self._context_snapshot = context
        logger.info(
            "agent_request",
            trace_id=self.trace_id,
            session_id=self.session_id,
            message_length=len(user_message),
            context_keys=list(context.keys()) if context else [],
            structure_id=context.get("structure_id") or context.get("active_structure_id"),
        )

    def record_tool_call(self, call_id: str, tool_name: str, arguments: dict[str, Any]) -> None:
        """Record that a tool was invoked (before result is known)."""
        self._current_tool_start = time.perf_counter()
        record = ToolCallRecord(
            call_id=call_id,
            tool_name=tool_name,
            arguments=arguments,
            started_at=self._current_tool_start,
        )
        self._tool_calls.append(record)
        logger.info(
            "agent_tool_call",
            trace_id=self.trace_id,
            session_id=self.session_id,
            tool_name=tool_name,
            arguments=arguments,
        )

    def record_tool_result(
        self,
        call_id: str,
        tool_name: str,
        result: Any,
        is_error: bool = False,
    ) -> None:
        """Record the result of a tool call."""
        duration_ms = (time.perf_counter() - self._current_tool_start) * 1000

        # Find the matching record
        for record in reversed(self._tool_calls):
            if record.call_id == call_id:
                record.duration_ms = duration_ms
                record.is_error = is_error
                if is_error:
                    record.error_message = str(result)[:500] if result else None
                else:
                    record.result_summary = _summarize_result(result)
                    if isinstance(result, dict):
                        record.result_keys = list(result.keys())[:20]
                break

        # Extract ground truth facts from successful tool results
        if not is_error and isinstance(result, dict):
            facts = _extract_ground_truth(tool_name, result)
            if facts:
                self._ground_truth.update(facts)

        logger.info(
            "agent_tool_result",
            trace_id=self.trace_id,
            session_id=self.session_id,
            tool_name=tool_name,
            is_error=is_error,
            duration_ms=round(duration_ms, 1),
            result_summary=_summarize_result(result) if not is_error else None,
            error=str(result)[:200] if is_error else None,
        )

    def record_usage(self, input_tokens: int, output_tokens: int) -> None:
        """Record LLM token usage for a single call."""
        self._llm_calls += 1
        self._input_tokens += input_tokens
        self._output_tokens += output_tokens

    def record_llm_call(
        self,
        *,
        messages: list[Any],
        system_prompt: str | None,
        tools: list[Any] | None,
        response: Any,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Record approximate token attribution for one LLM call."""
        input_components = _estimate_input_components(messages, system_prompt, tools)
        output_components = _estimate_output_components(response)
        input_total_chars = float(sum(input_components.values()))
        output_total_chars = float(sum(output_components.values()))

        for key, value in input_components.items():
            self._input_component_chars[key] += value
        for key, value in output_components.items():
            self._output_component_chars[key] += value

        self._llm_call_breakdown.append(
            {
                "call_index": self._llm_calls + 1,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "input": _build_attribution(input_components, input_tokens, input_total_chars),
                "output": _build_attribution(output_components, output_tokens, output_total_chars),
            }
        )

    def record_response(self, text: str) -> None:
        """Record the agent's final text response."""
        self._final_response = text
        logger.info(
            "agent_response",
            trace_id=self.trace_id,
            session_id=self.session_id,
            response_length=len(text),
            tool_calls_count=len(self._tool_calls),
            llm_calls=self._llm_calls,
            total_input_tokens=self._input_tokens,
            total_output_tokens=self._output_tokens,
        )

    def finalize(self) -> AgentTrace:
        """Finalize the trace and return the complete record."""
        total_duration_ms = (time.perf_counter() - self._started_at) * 1000
        total_tokens = self._input_tokens + self._output_tokens
        input_chars_total = float(sum(self._input_component_chars.values()))
        output_chars_total = float(sum(self._output_component_chars.values()))
        input_attribution = _build_attribution(self._input_component_chars, self._input_tokens, input_chars_total)
        output_attribution = _build_attribution(self._output_component_chars, self._output_tokens, output_chars_total)

        total_attribution: dict[str, Any] = {}
        for key in sorted(set(input_attribution) | set(output_attribution)):
            estimated_tokens = (
                input_attribution.get(key, {}).get("estimated_tokens", 0.0)
                + output_attribution.get(key, {}).get("estimated_tokens", 0.0)
            )
            if estimated_tokens <= 0:
                continue
            total_attribution[key] = {
                "estimated_tokens": round(estimated_tokens, 1),
                "percent": round((estimated_tokens / total_tokens) * 100, 1) if total_tokens > 0 else 0.0,
            }

        token_attribution = {
            "input": input_attribution,
            "output": output_attribution,
            "total": total_attribution,
        }

        trace = AgentTrace(
            trace_id=self.trace_id,
            session_id=self.session_id,
            user_message=self._user_message,
            context_snapshot=self._context_snapshot,
            tool_calls=self._tool_calls,
            final_response=self._final_response,
            total_duration_ms=total_duration_ms,
            llm_calls=self._llm_calls,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            ground_truth_facts=self._ground_truth,
            token_attribution=token_attribution,
            llm_call_breakdown=self._llm_call_breakdown,
        )

        # Log the complete trace summary
        logger.info(
            "agent_trace_complete",
            trace_id=self.trace_id,
            session_id=self.session_id,
            total_duration_ms=round(total_duration_ms, 1),
            tool_calls=[
                {
                    "name": tc.tool_name,
                    "duration_ms": round(tc.duration_ms, 1),
                    "is_error": tc.is_error,
                }
                for tc in self._tool_calls
            ],
            ground_truth_facts=self._ground_truth,
            llm_calls=self._llm_calls,
            tokens=self._input_tokens + self._output_tokens,
            token_attribution=token_attribution.get("total", {}),
        )

        return trace
