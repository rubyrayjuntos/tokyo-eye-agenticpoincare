-- ============================================================================
-- Migration 028: Normalization Audit Trail
-- Date: 2026-05-27
-- Purpose: Record every normalization attempt (success or failure) for
--          operational monitoring, debugging, and governance compliance.
--          This is the "audit log" for the governed write path.
-- ============================================================================

CREATE TABLE IF NOT EXISTS normalization_audit (
    audit_id            TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    run_id              TEXT,                             -- may be NULL if validation fails before run_id extraction
    structure_id        TEXT,
    payload_type        TEXT NOT NULL,                    -- 'gnn_output', 'phase3_persistence', etc.
    status              TEXT NOT NULL,                    -- 'success', 'validation_error', 'write_error', 'rejected'
    assets_created      INTEGER DEFAULT 0,
    error_message       TEXT,
    error_details       JSONB,
    payload_summary     JSONB,                            -- non-sensitive summary (node count, space type, etc.)
    duration_ms         INTEGER,                          -- wall-clock time for the normalization
    caller_identity     TEXT,                             -- who/what called the normalizer
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_norm_audit_run ON normalization_audit(run_id);
CREATE INDEX IF NOT EXISTS idx_norm_audit_status ON normalization_audit(status);
CREATE INDEX IF NOT EXISTS idx_norm_audit_type ON normalization_audit(payload_type);
CREATE INDEX IF NOT EXISTS idx_norm_audit_created ON normalization_audit(created_at);

-- Partial index for fast failure lookups (operational monitoring)
CREATE INDEX IF NOT EXISTS idx_norm_audit_failures
    ON normalization_audit(created_at DESC)
    WHERE status != 'success';

-- ============================================================================
-- NOTES
-- ============================================================================
-- - Every call to the Normalizer (success or failure) produces an audit record.
-- - payload_summary stores non-sensitive metadata (e.g., node count, space type)
--   NOT the full payload (which may be large).
-- - This table is append-only and should never be modified after creation.
-- - Useful for: debugging failed writes, monitoring throughput, detecting
--   anomalies, and compliance reporting.
-- - The partial index on failures enables fast "show me recent problems" queries.
