-- ============================================================================
-- Migration 046: MD Validation Results
-- Date: 2026-06-22
-- Purpose: Create fact_md_validation table for storing molecular dynamics
--          validation results linked to cryptic sites and provenance.
-- Requirements: 6.2, 6.3, 6.4
-- ============================================================================

CREATE TABLE IF NOT EXISTS fact_md_validation (
    validation_id       TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    structure_id        TEXT NOT NULL,
    site_id             TEXT NOT NULL,
    run_id              TEXT NOT NULL,
    pocket_open_fraction DOUBLE PRECISION,
    confidence_delta    DOUBLE PRECISION,
    duration_ns         DOUBLE PRECISION NOT NULL,
    temperature_k       DOUBLE PRECISION NOT NULL,
    force_field         TEXT NOT NULL,
    dry_run             BOOLEAN NOT NULL DEFAULT FALSE,
    duration_ms         DOUBLE PRECISION,
    status              TEXT NOT NULL DEFAULT 'complete',
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (run_id, site_id)
);

CREATE INDEX IF NOT EXISTS idx_fact_md_validation_structure
    ON fact_md_validation (structure_id);

CREATE INDEX IF NOT EXISTS idx_fact_md_validation_site
    ON fact_md_validation (site_id);

CREATE INDEX IF NOT EXISTS idx_fact_md_validation_run
    ON fact_md_validation (run_id);
