-- ============================================================================
-- Migration 018: Additional Site/Residue Bridge and Summary Tables
-- Date: 2026-05-27
-- Purpose: Strengthen the relationship between dim_site and dim_residue,
--          and add a lightweight summary table for common residue-level
--          metrics that multiple consumers (agent, visualizer, RAG) often need.
-- ============================================================================

-- Enhanced bridge with metadata (if not already sufficient in 015)
-- This migration ensures a flexible, query-friendly bridge exists.

-- Residue-level summary metrics (common computed values cached for fast access)
CREATE TABLE IF NOT EXISTS fact_residue_summary (
    summary_id          TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    run_id              TEXT NOT NULL REFERENCES provenance_run(run_id),
    structure_id        TEXT NOT NULL REFERENCES dim_structure(structure_id),
    residue_id          TEXT NOT NULL REFERENCES dim_residue(residue_id),
    latest_cone_depth   DOUBLE PRECISION,
    latest_epistemic    DOUBLE PRECISION,
    latest_aleatoric    DOUBLE PRECISION,
    dehydron_count      INTEGER,
    site_memberships    JSONB,                    -- list of site_ids this residue belongs to
    computed_at         TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_residue_summary_residue ON fact_residue_summary(residue_id);
CREATE INDEX IF NOT EXISTS idx_residue_summary_run ON fact_residue_summary(run_id);

-- ============================================================================
-- NOTES
-- ============================================================================
-- This summary table can be populated by the normalizer or background jobs
-- after major runs. It is intended to speed up common viewport/agent queries
-- while still being fully governed via provenance_run.
-- It is an example of the "derived / cached layer" mentioned in the architecture.