"""Persistent hybrid memory for dashboard chat sessions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from agent.telemetry import AgentTrace
from shared.logging import get_logger

logger = get_logger(__name__)

_SUMMARY_MAX_LINES = 8
_SUMMARY_MAX_CHARS = 1600
_PROMPT_BLOCK_MAX_CHARS = 3600
_RECENT_EXCHANGE_LIMIT = 4
_RECALL_LIMIT = 5
_USER_PREVIEW_CHARS = 160
_ASSISTANT_PREVIEW_CHARS = 240
_RECALL_PREVIEW_CHARS = 600


class SupportsMemoryDB(Protocol):
    """Minimal DB surface needed by the memory layer."""

    async def execute(self, query: str, params: dict[str, Any] | None = None) -> None:
        ...

    async def execute_many(self, query: str, params_list: list[dict[str, Any]] | list[tuple]) -> None:
        ...

    async def fetch_one(self, query: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
        ...

    async def fetch_all(self, query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        ...


@dataclass(slots=True)
class MemoryPromptContext:
    """Prompt block and the persisted artifacts it was assembled from."""

    block: str
    summary: str | None
    recent_exchanges: list[dict[str, Any]]
    recalled_items: list[dict[str, Any]]


def merge_orchestration_context(
    context: dict[str, Any] | None,
    orchestration_state: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge persisted orchestration state only when the caller did not provide one."""

    merged = dict(context or {})
    if "orchestration_state" not in merged and isinstance(orchestration_state, dict):
        merged["orchestration_state"] = orchestration_state
    return merged


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return " ".join(text.split())


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[: max(limit - 3, 0)]}..."


def _extract_structure_id(context: dict[str, Any] | None) -> str | None:
    if not context:
        return None
    for key in ("structure_id", "active_structure_id"):
        value = context.get(key)
        if isinstance(value, str) and value:
            return value
    compare = context.get("compare")
    if isinstance(compare, dict):
        value = compare.get("primary_structure_id")
        if isinstance(value, str) and value:
            return value
    return None


def _extract_run_id(context: dict[str, Any] | None) -> str | None:
    if not context:
        return None
    for key in ("run_id", "active_run_id"):
        value = context.get(key)
        if isinstance(value, str) and value:
            return value
    pipeline = context.get("pipeline")
    if isinstance(pipeline, dict):
        value = pipeline.get("run_id")
        if isinstance(value, str) and value:
            return value
    return None


def _extract_user_id(context: dict[str, Any] | None) -> str | None:
    if not context:
        return None
    value = context.get("user_id")
    if isinstance(value, str) and value:
        return value
    return None


def _context_fact_pairs(context: dict[str, Any] | None) -> list[tuple[str, str]]:
    if not context:
        return []

    facts: list[tuple[str, str]] = []
    structure_id = _extract_structure_id(context)
    structure_title = context.get("structure_title")
    if isinstance(structure_id, str) and structure_id:
        facts.append(("structure_id", structure_id))
    if isinstance(structure_title, str) and structure_title:
        facts.append(("structure_title", structure_title))

    residue_count = context.get("residue_count")
    if isinstance(residue_count, int):
        facts.append(("residue_count", str(residue_count)))

    data_summary = context.get("data_summary")
    if isinstance(data_summary, dict):
        summary_residue_count = data_summary.get("residue_count")
        if isinstance(summary_residue_count, int):
            facts.append(("residue_count", str(summary_residue_count)))
        source_leak_count = data_summary.get("source_leak_count")
        if isinstance(source_leak_count, int):
            facts.append(("source_leak_count", str(source_leak_count)))
        hypothesis_count = data_summary.get("hypothesis_count")
        if isinstance(hypothesis_count, int):
            facts.append(("hypothesis_count", str(hypothesis_count)))

    deduped: dict[str, str] = {}
    for name, value in facts:
        deduped[name] = value
    return list(deduped.items())


def _format_fact_text(fact_name: str, fact_value_text: str, structure_id: str | None) -> str:
    label = fact_name.replace("_", " ")
    if structure_id and fact_name in {"residue_count", "source_leak_count", "hypothesis_count", "structure_title"}:
        return f"{structure_id} {label}: {fact_value_text}"
    return f"{label}: {fact_value_text}"


def _summarize_exchange(user_message: str, assistant_message: str, structure_id: str | None) -> str:
    user_preview = _truncate(_normalize_text(user_message), _USER_PREVIEW_CHARS)
    assistant_preview = _truncate(_normalize_text(assistant_message), _ASSISTANT_PREVIEW_CHARS)
    prefix = f"[{structure_id}] " if structure_id else ""
    return f"{prefix}User asked: {user_preview} Assistant replied: {assistant_preview}"


def _merge_summary(
    existing_summary: str | None,
    user_message: str,
    assistant_message: str,
    structure_id: str | None,
) -> str:
    lines = [line.strip() for line in (existing_summary or "").splitlines() if line.strip()]
    lines.append(f"- {_summarize_exchange(user_message, assistant_message, structure_id)}")
    merged = "\n".join(lines[-_SUMMARY_MAX_LINES:])
    return _truncate(merged, _SUMMARY_MAX_CHARS)


def _build_trace_summary(trace: AgentTrace, structure_id: str | None, run_id: str | None) -> str:
    tool_names = ", ".join(sorted({tc.tool_name for tc in trace.tool_calls if tc.tool_name}))
    parts = [
        _summarize_exchange(trace.user_message, trace.final_response, structure_id),
    ]
    if run_id:
        parts.append(f"run_id={run_id}")
    if tool_names:
        parts.append(f"tools={tool_names}")
    if trace.ground_truth_facts:
        keys = ", ".join(sorted(trace.ground_truth_facts.keys()))
        parts.append(f"ground_truth={keys}")
    return _truncate(" | ".join(parts), _SUMMARY_MAX_CHARS)


def _format_recent_exchanges(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""

    lines = ["Recent durable exchanges:"]
    for row in reversed(rows):
        user_preview = _truncate(_normalize_text(row.get("user_message")), _USER_PREVIEW_CHARS)
        assistant_preview = _truncate(_normalize_text(row.get("assistant_message")), _ASSISTANT_PREVIEW_CHARS)
        lines.append(f"- User: {user_preview}")
        lines.append(f"  Assistant: {assistant_preview}")
    return "\n".join(lines)


def _format_recalled_items(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""

    lines = ["Relevant prior memory:"]
    for row in rows:
        memory_kind = _normalize_text(row.get("memory_kind")) or "memory"
        body_text = _truncate(_normalize_text(row.get("body_text")), _RECALL_PREVIEW_CHARS)
        lines.append(f"- [{memory_kind}] {body_text}")
    return "\n".join(lines)


async def build_memory_prompt_context(
    *,
    db: SupportsMemoryDB,
    session_id: str,
    user_message: str,
    context: dict[str, Any] | None,
    user_id: str | None = None,
) -> MemoryPromptContext:
    """Load persistent summary + recall and render them into a prompt block."""

    structure_id = _extract_structure_id(context)
    run_id = _extract_run_id(context)
    recall_user_filter = (
        "(:user_id IS NOT NULL AND user_id = :user_id)"
        if user_id
        else "FALSE"
    )

    try:
        summary_row = await db.fetch_one(
            """
            SELECT rolling_summary
            FROM agent_session_summary
            WHERE session_id = :session_id
            """,
            {"session_id": session_id},
        )
        recent_exchanges = await db.fetch_all(
            """
            SELECT user_message, assistant_message, created_at
            FROM agent_chat_exchange
            WHERE session_id = :session_id
            ORDER BY created_at DESC
            LIMIT :limit
            """,
            {"session_id": session_id, "limit": _RECENT_EXCHANGE_LIMIT},
        )
        recalled_items = await db.fetch_all(
            f"""
            WITH recall_candidates AS (
                SELECT
                    'fact' AS memory_kind,
                    memory_key AS memory_id,
                    fact_text AS body_text,
                    updated_at AS memory_ts,
                    ts_rank_cd(search_document, websearch_to_tsquery('english', :query_text)) +
                        CASE WHEN :structure_id IS NOT NULL AND structure_id = :structure_id THEN 0.25 ELSE 0 END +
                        CASE WHEN :run_id IS NOT NULL AND run_id = :run_id THEN 0.25 ELSE 0 END AS score
                FROM agent_memory_fact
                WHERE {recall_user_filter}
                  AND (
                        search_document @@ websearch_to_tsquery('english', :query_text)
                        OR (:structure_id IS NOT NULL AND structure_id = :structure_id)
                        OR (:run_id IS NOT NULL AND run_id = :run_id)
                  )

                UNION ALL

                SELECT
                    'note' AS memory_kind,
                    note_id AS memory_id,
                    note_text AS body_text,
                    updated_at AS memory_ts,
                    ts_rank_cd(search_document, websearch_to_tsquery('english', :query_text)) +
                        CASE WHEN :structure_id IS NOT NULL AND structure_id = :structure_id THEN 0.25 ELSE 0 END +
                        CASE WHEN :run_id IS NOT NULL AND run_id = :run_id THEN 0.25 ELSE 0 END AS score
                FROM agent_memory_note
                WHERE {recall_user_filter}
                  AND (
                        search_document @@ websearch_to_tsquery('english', :query_text)
                        OR (:structure_id IS NOT NULL AND structure_id = :structure_id)
                        OR (:run_id IS NOT NULL AND run_id = :run_id)
                  )

                UNION ALL

                SELECT
                    'trace' AS memory_kind,
                    trace_id AS memory_id,
                    response_summary AS body_text,
                    created_at AS memory_ts,
                    ts_rank_cd(search_document, websearch_to_tsquery('english', :query_text)) +
                        CASE WHEN :structure_id IS NOT NULL AND structure_id = :structure_id THEN 0.25 ELSE 0 END +
                        CASE WHEN :run_id IS NOT NULL AND run_id = :run_id THEN 0.25 ELSE 0 END AS score
                FROM agent_trace_summary
                WHERE {recall_user_filter}
                  AND (
                        search_document @@ websearch_to_tsquery('english', :query_text)
                        OR (:structure_id IS NOT NULL AND structure_id = :structure_id)
                        OR (:run_id IS NOT NULL AND run_id = :run_id)
                  )

                UNION ALL

                SELECT
                    'exchange' AS memory_kind,
                    exchange_id AS memory_id,
                    exchange_summary AS body_text,
                    created_at AS memory_ts,
                    ts_rank_cd(search_document, websearch_to_tsquery('english', :query_text)) +
                        CASE WHEN :structure_id IS NOT NULL AND structure_id = :structure_id THEN 0.25 ELSE 0 END +
                        CASE WHEN :run_id IS NOT NULL AND run_id = :run_id THEN 0.25 ELSE 0 END AS score
                FROM agent_chat_exchange
                WHERE session_id != :session_id
                  AND {recall_user_filter}
                  AND (
                        search_document @@ websearch_to_tsquery('english', :query_text)
                        OR (:structure_id IS NOT NULL AND structure_id = :structure_id)
                        OR (:run_id IS NOT NULL AND run_id = :run_id)
                  )

                UNION ALL

                SELECT
                    'doc' AS memory_kind,
                    chunk_id AS memory_id,
                    chunk_text AS body_text,
                    updated_at AS memory_ts,
                    ts_rank_cd(search_document, websearch_to_tsquery('english', :query_text)) +
                        CASE WHEN :structure_id IS NOT NULL AND structure_id = :structure_id THEN 0.25 ELSE 0 END +
                        CASE WHEN :run_id IS NOT NULL AND run_id = :run_id THEN 0.25 ELSE 0 END AS score
                FROM agent_doc_chunk
                WHERE (
                        search_document @@ websearch_to_tsquery('english', :query_text)
                        OR (:structure_id IS NOT NULL AND structure_id = :structure_id)
                        OR (:run_id IS NOT NULL AND run_id = :run_id)
                  )
            )
            SELECT memory_kind, memory_id, body_text, memory_ts, score
            FROM recall_candidates
            WHERE score > 0
            ORDER BY score DESC, memory_ts DESC
            LIMIT :limit
            """,
            {
                "query_text": user_message,
                "structure_id": structure_id,
                "run_id": run_id,
                "session_id": session_id,
                "user_id": user_id,
                "limit": _RECALL_LIMIT,
            },
        )
    except Exception as exc:
        logger.warning("Persistent memory lookup unavailable for session %s: %s", session_id, exc)
        return MemoryPromptContext(block="", summary=None, recent_exchanges=[], recalled_items=[])

    summary = _normalize_text((summary_row or {}).get("rolling_summary")) or None
    sections = []
    if summary:
        sections.append(f"Persistent session summary:\n{summary}")

    recalled_block = _format_recalled_items(recalled_items)
    if recalled_block:
        sections.append(recalled_block)

    recent_block = _format_recent_exchanges(recent_exchanges)
    if recent_block:
        sections.append(recent_block)

    if not sections:
        return MemoryPromptContext(block="", summary=summary, recent_exchanges=recent_exchanges, recalled_items=recalled_items)

    block = "\n\nPersistent Memory:\n" + "\n\n".join(sections)
    return MemoryPromptContext(
        block=_truncate(block, _PROMPT_BLOCK_MAX_CHARS),
        summary=summary,
        recent_exchanges=recent_exchanges,
        recalled_items=recalled_items,
    )


async def load_persisted_orchestration_state(
    *,
    db: SupportsMemoryDB,
    session_id: str,
    context: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Load the most recent persisted orchestration state for the session/run."""

    run_id = _extract_run_id(context)
    structure_id = _extract_structure_id(context)

    try:
        row = await db.fetch_one(
            """
            SELECT note_text
            FROM agent_memory_note
            WHERE session_id = :session_id
              AND note_kind = 'orchestration_state'
              AND (
                    (:run_id IS NOT NULL AND run_id = :run_id)
                    OR (:run_id IS NULL AND :structure_id IS NOT NULL AND structure_id = :structure_id)
                    OR (:run_id IS NULL AND :structure_id IS NULL)
              )
            ORDER BY
                CASE WHEN :run_id IS NOT NULL AND run_id = :run_id THEN 0 ELSE 1 END,
                CASE WHEN :structure_id IS NOT NULL AND structure_id = :structure_id THEN 0 ELSE 1 END,
                updated_at DESC
            LIMIT 1
            """,
            {
                "session_id": session_id,
                "run_id": run_id,
                "structure_id": structure_id,
            },
        )
    except Exception as exc:
        logger.warning("Persistent orchestration-state lookup unavailable for session %s: %s", session_id, exc)
        return None

    note_text = (row or {}).get("note_text")
    if not isinstance(note_text, str) or not note_text.strip():
        return None

    try:
        payload = json.loads(note_text)
    except json.JSONDecodeError as exc:
        logger.warning("Invalid orchestration-state note for session %s: %s", session_id, exc)
        return None

    return payload if isinstance(payload, dict) else None


async def persist_memory_interaction(
    *,
    db: SupportsMemoryDB,
    session_id: str,
    user_message: str,
    assistant_message: str,
    context: dict[str, Any] | None,
    trace: AgentTrace,
    retrieved_memory_block: str,
    next_orchestration_state: dict[str, Any] | None = None,
    user_id: str | None = None,
) -> None:
    """Persist durable chat memory without failing the user request."""

    structure_id = _extract_structure_id(context)
    run_id = _extract_run_id(context)
    if user_id is None:
        user_id = _extract_user_id(context)

    try:
        summary_row = await db.fetch_one(
            """
            SELECT rolling_summary, message_count
            FROM agent_session_summary
            WHERE session_id = :session_id
            """,
            {"session_id": session_id},
        )
        merged_summary = _merge_summary(
            (summary_row or {}).get("rolling_summary"),
            user_message,
            assistant_message,
            structure_id,
        )
        next_message_count = int((summary_row or {}).get("message_count") or 0) + 1
        exchange_summary = _summarize_exchange(user_message, assistant_message, structure_id)

        await db.execute(
            """
            INSERT INTO agent_session_summary (
                session_id,
                user_id,
                latest_structure_id,
                latest_run_id,
                rolling_summary,
                last_exchange_summary,
                message_count,
                last_interaction_at,
                updated_at
            ) VALUES (
                :session_id,
                :user_id,
                :latest_structure_id,
                :latest_run_id,
                :rolling_summary,
                :last_exchange_summary,
                :message_count,
                NOW(),
                NOW()
            )
            ON CONFLICT (session_id) DO UPDATE SET
                user_id = EXCLUDED.user_id,
                latest_structure_id = EXCLUDED.latest_structure_id,
                latest_run_id = EXCLUDED.latest_run_id,
                rolling_summary = EXCLUDED.rolling_summary,
                last_exchange_summary = EXCLUDED.last_exchange_summary,
                message_count = EXCLUDED.message_count,
                last_interaction_at = NOW(),
                updated_at = NOW()
            """,
            {
                "session_id": session_id,
                "user_id": user_id,
                "latest_structure_id": structure_id,
                "latest_run_id": run_id,
                "rolling_summary": merged_summary,
                "last_exchange_summary": exchange_summary,
                "message_count": next_message_count,
            },
        )

        await db.execute(
            """
            INSERT INTO agent_chat_exchange (
                session_id,
                user_id,
                structure_id,
                run_id,
                user_message,
                assistant_message,
                exchange_summary,
                retrieved_memory_block
            ) VALUES (
                :session_id,
                :user_id,
                :structure_id,
                :run_id,
                :user_message,
                :assistant_message,
                :exchange_summary,
                :retrieved_memory_block
            )
            """,
            {
                "session_id": session_id,
                "user_id": user_id,
                "structure_id": structure_id,
                "run_id": run_id,
                "user_message": user_message,
                "assistant_message": assistant_message,
                "exchange_summary": exchange_summary,
                "retrieved_memory_block": retrieved_memory_block,
            },
        )

        trace_summary = _build_trace_summary(trace, structure_id, run_id)
        tool_names = ",".join(sorted({tc.tool_name for tc in trace.tool_calls if tc.tool_name}))
        fact_names = ",".join(sorted(trace.ground_truth_facts.keys()))

        await db.execute(
            """
            INSERT INTO agent_trace_summary (
                trace_id,
                session_id,
                structure_id,
                run_id,
                user_message,
                response_summary,
                tool_names_text,
                ground_truth_fact_names,
                input_tokens,
                output_tokens,
                total_duration_ms
            ) VALUES (
                :trace_id,
                :session_id,
                :structure_id,
                :run_id,
                :user_message,
                :response_summary,
                :tool_names_text,
                :ground_truth_fact_names,
                :input_tokens,
                :output_tokens,
                :total_duration_ms
            )
            ON CONFLICT (trace_id) DO UPDATE SET
                response_summary = EXCLUDED.response_summary,
                tool_names_text = EXCLUDED.tool_names_text,
                ground_truth_fact_names = EXCLUDED.ground_truth_fact_names,
                input_tokens = EXCLUDED.input_tokens,
                output_tokens = EXCLUDED.output_tokens,
                total_duration_ms = EXCLUDED.total_duration_ms
            """,
            {
                "trace_id": trace.trace_id,
                "session_id": session_id,
                "structure_id": structure_id,
                "run_id": run_id,
                "user_message": trace.user_message,
                "response_summary": trace_summary,
                "tool_names_text": tool_names,
                "ground_truth_fact_names": fact_names,
                "input_tokens": trace.input_tokens,
                "output_tokens": trace.output_tokens,
                "total_duration_ms": trace.total_duration_ms,
            },
        )

        fact_params: list[dict[str, Any]] = []
        for fact_name, fact_value_text in _context_fact_pairs(context):
            fact_params.append(
                {
                    "memory_key": f"context:{session_id}:{structure_id or '-'}:{run_id or '-'}:{fact_name}",
                    "session_id": session_id,
                    "user_id": user_id,
                    "structure_id": structure_id,
                    "run_id": run_id,
                    "fact_type": "dashboard_context",
                    "fact_name": fact_name,
                    "fact_value_text": fact_value_text,
                    "fact_text": _format_fact_text(fact_name, fact_value_text, structure_id),
                    "source_kind": "dashboard_context",
                    "source_id": session_id,
                }
            )

        for fact_name, fact_value in trace.ground_truth_facts.items():
            fact_value_text = _normalize_text(fact_value)
            if not fact_value_text:
                continue
            fact_params.append(
                {
                    "memory_key": f"trace:{trace.trace_id}:{fact_name}",
                    "session_id": session_id,
                    "user_id": user_id,
                    "structure_id": structure_id,
                    "run_id": run_id,
                    "fact_type": "ground_truth",
                    "fact_name": fact_name,
                    "fact_value_text": fact_value_text,
                    "fact_text": _format_fact_text(fact_name, fact_value_text, structure_id),
                    "source_kind": "trace",
                    "source_id": trace.trace_id,
                }
            )

        if fact_params:
            await db.execute_many(
                """
                INSERT INTO agent_memory_fact (
                    memory_key,
                    session_id,
                    user_id,
                    structure_id,
                    run_id,
                    fact_type,
                    fact_name,
                    fact_value_text,
                    fact_text,
                    source_kind,
                    source_id,
                    confidence,
                    last_observed_at,
                    updated_at
                ) VALUES (
                    :memory_key,
                    :session_id,
                    :user_id,
                    :structure_id,
                    :run_id,
                    :fact_type,
                    :fact_name,
                    :fact_value_text,
                    :fact_text,
                    :source_kind,
                    :source_id,
                    1.0,
                    NOW(),
                    NOW()
                )
                ON CONFLICT (memory_key) DO UPDATE SET
                    fact_value_text = EXCLUDED.fact_value_text,
                    fact_text = EXCLUDED.fact_text,
                    source_kind = EXCLUDED.source_kind,
                    source_id = EXCLUDED.source_id,
                    confidence = EXCLUDED.confidence,
                    last_observed_at = NOW(),
                    updated_at = NOW()
                """,
                fact_params,
            )

        if isinstance(next_orchestration_state, dict):
            await db.execute(
                """
                INSERT INTO agent_memory_note (
                    session_id,
                    user_id,
                    structure_id,
                    run_id,
                    note_kind,
                    title,
                    note_text,
                    source,
                    updated_at
                ) VALUES (
                    :session_id,
                    :user_id,
                    :structure_id,
                    :run_id,
                    'orchestration_state',
                    :title,
                    :note_text,
                    'agent',
                    NOW()
                )
                """,
                {
                    "session_id": session_id,
                    "user_id": user_id,
                    "structure_id": structure_id,
                    "run_id": run_id,
                    "title": "Latest orchestration continuation state",
                    "note_text": json.dumps(next_orchestration_state, sort_keys=True),
                },
            )
    except Exception as exc:
        logger.warning("Persistent memory write unavailable for session %s: %s", session_id, exc)
