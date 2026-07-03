-- ============================================================================
-- Migration 049: Pipeline Audit Events
-- Date: 2026-06-26
-- Purpose: Persistent structured audit trail for compute pipeline instrumentation
--          (geometric validation, curvature passthrough, enforcement, preconditions).
--          Complements normalization_audit (governed write path) with runtime
--          pipeline observability queryable over time.
-- ============================================================================

CREATE TABLE IF NOT EXISTS audit_pipeline_events (
    event_id            TEXT PRIMARY KEY,
    timestamp           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    event_type          TEXT NOT NULL,
    severity            TEXT NOT NULL CHECK (severity IN ('info', 'warning', 'error')),
    structure_id        TEXT,
    pipeline_job_id     TEXT,
    job_name            TEXT,
    contract_version    TEXT,
    details             JSONB NOT NULL DEFAULT '{}'::jsonb,
    correlation_id      TEXT,
    enforcement_level   TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audit_pipeline_structure_ts
    ON audit_pipeline_events(structure_id, timestamp DESC)
    WHERE structure_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_audit_pipeline_job_ts
    ON audit_pipeline_events(pipeline_job_id, timestamp DESC)
    WHERE pipeline_job_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_audit_pipeline_severity_ts
    ON audit_pipeline_events(severity, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_audit_pipeline_event_type
    ON audit_pipeline_events(event_type, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_audit_pipeline_correlation
    ON audit_pipeline_events(correlation_id)
    WHERE correlation_id IS NOT NULL;
