-- Migration 005: Viewport State Persistence and Ingestion Status
-- Adds viewport snapshot persistence and pipeline ingestion status tracking.
-- Idempotent: safe to run multiple times (IF NOT EXISTS / DO blocks).
--
-- Requirements: 7.1, 7.2, 7.3, 7.4, 11.4, 2.8

-- ============================================================================
-- VIEWPORT STATE PERSISTENCE
-- ============================================================================

CREATE TABLE IF NOT EXISTS fact_viewport_snapshot (
    snapshot_id     TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL,
    name            TEXT NOT NULL,
    viewport_states JSONB NOT NULL,       -- All viewport states at capture time
    structure_ids   JSONB NOT NULL,       -- Referenced structures (not raw coords)
    camera_states   JSONB,                -- Per-viewport camera positions
    selections      JSONB,                -- Active selections
    annotations     JSONB,                -- Active annotations
    sharing_token   TEXT,                 -- NULL if not shared
    token_expiry    TIMESTAMPTZ,          -- NULL if not shared
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_snapshot_session ON fact_viewport_snapshot(session_id);
CREATE INDEX IF NOT EXISTS idx_snapshot_token ON fact_viewport_snapshot(sharing_token) WHERE sharing_token IS NOT NULL;

-- ============================================================================
-- INGESTION STATUS TRACKING
-- ============================================================================

CREATE TABLE IF NOT EXISTS dim_ingestion_status (
    structure_id    TEXT NOT NULL,
    phase_name      TEXT NOT NULL,        -- "gnn", "phase2", "phase4", "phase4b", "phase5", "normalization"
    status          TEXT NOT NULL,        -- "complete", "failed", "pending", "not_started"
    completed_at    TIMESTAMPTZ,
    error_message   TEXT,
    PRIMARY KEY (structure_id, phase_name)
);

CREATE INDEX IF NOT EXISTS idx_ingestion_structure ON dim_ingestion_status(structure_id);
