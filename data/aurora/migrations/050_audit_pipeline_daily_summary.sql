-- ============================================================================
-- Migration 050: Pipeline audit daily aggregates (retention companion)
-- Date: 2026-06-26
-- Purpose: Roll up detailed audit_pipeline_events older than retention window
--          while preserving queryable trends.
-- ============================================================================

CREATE TABLE IF NOT EXISTS audit_pipeline_daily_summary (
    summary_date        DATE NOT NULL,
    event_type          TEXT NOT NULL,
    severity            TEXT NOT NULL,
    job_name            TEXT NOT NULL DEFAULT '',
    structure_id        TEXT NOT NULL DEFAULT '',
    event_count         INTEGER NOT NULL DEFAULT 0,
    sample_details      JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (summary_date, event_type, severity, job_name, structure_id)
);

CREATE INDEX IF NOT EXISTS idx_audit_daily_summary_date
    ON audit_pipeline_daily_summary(summary_date DESC);
