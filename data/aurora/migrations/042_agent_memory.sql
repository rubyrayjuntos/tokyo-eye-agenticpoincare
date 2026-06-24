-- ============================================================================
-- Migration 042: Persistent agent memory and hybrid recall
-- Date: 2026-06-09
-- Purpose: Add low-cost, retrieval-friendly memory tables for dashboard chat.
--          Uses Postgres full-text search first, with pgvector-ready schema left
--          to a later phase when local embeddings are wired in.
-- ============================================================================

-- Session-level rolling memory used to replace purely in-memory chat history.
CREATE TABLE IF NOT EXISTS agent_session_summary (
    session_id              TEXT PRIMARY KEY,
    user_id                 TEXT,
    latest_structure_id     TEXT,
    latest_run_id           TEXT,
    rolling_summary         TEXT NOT NULL DEFAULT '',
    last_exchange_summary   TEXT,
    message_count           INTEGER NOT NULL DEFAULT 0,
    created_at              TIMESTAMPTZ DEFAULT NOW(),
    updated_at              TIMESTAMPTZ DEFAULT NOW(),
    last_interaction_at     TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_session_summary_structure
    ON agent_session_summary (latest_structure_id);

CREATE INDEX IF NOT EXISTS idx_agent_session_summary_last_interaction
    ON agent_session_summary (last_interaction_at DESC);

COMMENT ON TABLE agent_session_summary IS
    'Persistent rolling summaries for agent chat sessions keyed by session_id.';

-- Durable exchange log. Stores raw turns plus a compact retrieval summary.
CREATE TABLE IF NOT EXISTS agent_chat_exchange (
    exchange_id              TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    session_id               TEXT NOT NULL,
    user_id                  TEXT,
    structure_id             TEXT,
    run_id                   TEXT,
    user_message             TEXT NOT NULL,
    assistant_message        TEXT NOT NULL,
    exchange_summary         TEXT NOT NULL,
    retrieved_memory_block   TEXT,
    created_at               TIMESTAMPTZ DEFAULT NOW(),
    search_document          TSVECTOR GENERATED ALWAYS AS (
        setweight(to_tsvector('english', COALESCE(exchange_summary, '')), 'A') ||
        setweight(to_tsvector('english', COALESCE(user_message, '')), 'B') ||
        setweight(to_tsvector('english', COALESCE(assistant_message, '')), 'C')
    ) STORED
);

CREATE INDEX IF NOT EXISTS idx_agent_chat_exchange_session
    ON agent_chat_exchange (session_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_agent_chat_exchange_structure
    ON agent_chat_exchange (structure_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_agent_chat_exchange_run
    ON agent_chat_exchange (run_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_agent_chat_exchange_search
    ON agent_chat_exchange USING GIN (search_document);

COMMENT ON TABLE agent_chat_exchange IS
    'Durable agent/user exchanges used for session replay, summary refresh, and cross-session recall.';

-- Grounded fact store. Populated from dashboard context and tool-backed ground truth.
CREATE TABLE IF NOT EXISTS agent_memory_fact (
    memory_key              TEXT PRIMARY KEY,
    session_id              TEXT,
    user_id                 TEXT,
    structure_id            TEXT,
    run_id                  TEXT,
    fact_type               TEXT NOT NULL,
    fact_name               TEXT NOT NULL,
    fact_value_text         TEXT NOT NULL,
    fact_text               TEXT NOT NULL,
    source_kind             TEXT NOT NULL,
    source_id               TEXT,
    confidence              DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    created_at              TIMESTAMPTZ DEFAULT NOW(),
    updated_at              TIMESTAMPTZ DEFAULT NOW(),
    last_observed_at        TIMESTAMPTZ DEFAULT NOW(),
    search_document         TSVECTOR GENERATED ALWAYS AS (
        setweight(to_tsvector('english', COALESCE(fact_name, '')), 'A') ||
        setweight(to_tsvector('english', COALESCE(fact_text, '')), 'B') ||
        setweight(to_tsvector('english', COALESCE(fact_value_text, '')), 'C')
    ) STORED
);

CREATE INDEX IF NOT EXISTS idx_agent_memory_fact_session
    ON agent_memory_fact (session_id, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_agent_memory_fact_structure
    ON agent_memory_fact (structure_id, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_agent_memory_fact_run
    ON agent_memory_fact (run_id, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_agent_memory_fact_search
    ON agent_memory_fact USING GIN (search_document);

COMMENT ON TABLE agent_memory_fact IS
    'Grounded durable facts available for exact-match and full-text memory recall.';

-- Free-form notes for later operator or agent-authored memory capture.
CREATE TABLE IF NOT EXISTS agent_memory_note (
    note_id                  TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    session_id               TEXT,
    user_id                  TEXT,
    structure_id             TEXT,
    run_id                   TEXT,
    note_kind                TEXT NOT NULL DEFAULT 'operator',
    title                    TEXT,
    note_text                TEXT NOT NULL,
    source                   TEXT NOT NULL DEFAULT 'agent',
    created_at               TIMESTAMPTZ DEFAULT NOW(),
    updated_at               TIMESTAMPTZ DEFAULT NOW(),
    search_document          TSVECTOR GENERATED ALWAYS AS (
        setweight(to_tsvector('english', COALESCE(title, '')), 'A') ||
        setweight(to_tsvector('english', COALESCE(note_text, '')), 'B')
    ) STORED
);

CREATE INDEX IF NOT EXISTS idx_agent_memory_note_structure
    ON agent_memory_note (structure_id, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_agent_memory_note_run
    ON agent_memory_note (run_id, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_agent_memory_note_search
    ON agent_memory_note USING GIN (search_document);

COMMENT ON TABLE agent_memory_note IS
    'Free-text operator or agent notes available for later hybrid memory recall.';

-- Compact trace summaries optimized for retrieval, not full raw telemetry.
CREATE TABLE IF NOT EXISTS agent_trace_summary (
    trace_id                  TEXT PRIMARY KEY,
    session_id                TEXT NOT NULL,
    structure_id              TEXT,
    run_id                    TEXT,
    user_message              TEXT NOT NULL,
    response_summary          TEXT NOT NULL,
    tool_names_text           TEXT,
    ground_truth_fact_names   TEXT,
    input_tokens              INTEGER NOT NULL DEFAULT 0,
    output_tokens             INTEGER NOT NULL DEFAULT 0,
    total_duration_ms         DOUBLE PRECISION NOT NULL DEFAULT 0,
    created_at                TIMESTAMPTZ DEFAULT NOW(),
    search_document           TSVECTOR GENERATED ALWAYS AS (
        setweight(to_tsvector('english', COALESCE(response_summary, '')), 'A') ||
        setweight(to_tsvector('english', COALESCE(user_message, '')), 'B') ||
        setweight(to_tsvector('english', COALESCE(tool_names_text, '')), 'C') ||
        setweight(to_tsvector('english', COALESCE(ground_truth_fact_names, '')), 'C')
    ) STORED
);

CREATE INDEX IF NOT EXISTS idx_agent_trace_summary_session
    ON agent_trace_summary (session_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_agent_trace_summary_structure
    ON agent_trace_summary (structure_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_agent_trace_summary_run
    ON agent_trace_summary (run_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_agent_trace_summary_search
    ON agent_trace_summary USING GIN (search_document);

COMMENT ON TABLE agent_trace_summary IS
    'Retrieval-friendly summaries of agent traces, tool usage, and grounded facts.';

-- Optional repo/doc chunks for future FTS-first or embedding-backed recall.
CREATE TABLE IF NOT EXISTS agent_doc_chunk (
    chunk_id                 TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    source_path              TEXT NOT NULL,
    source_kind              TEXT NOT NULL DEFAULT 'repo_doc',
    title                    TEXT,
    structure_id             TEXT,
    run_id                   TEXT,
    chunk_order              INTEGER NOT NULL DEFAULT 0,
    chunk_text               TEXT NOT NULL,
    created_at               TIMESTAMPTZ DEFAULT NOW(),
    updated_at               TIMESTAMPTZ DEFAULT NOW(),
    search_document          TSVECTOR GENERATED ALWAYS AS (
        setweight(to_tsvector('english', COALESCE(title, '')), 'A') ||
        setweight(to_tsvector('english', COALESCE(chunk_text, '')), 'B')
    ) STORED,
    UNIQUE (source_path, chunk_order)
);

CREATE INDEX IF NOT EXISTS idx_agent_doc_chunk_structure
    ON agent_doc_chunk (structure_id, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_agent_doc_chunk_run
    ON agent_doc_chunk (run_id, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_agent_doc_chunk_search
    ON agent_doc_chunk USING GIN (search_document);

COMMENT ON TABLE agent_doc_chunk IS
    'Chunked documentation or findings corpora for low-cost full-text agent recall.';
