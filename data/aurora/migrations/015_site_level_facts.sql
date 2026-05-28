-- ============================================================================
-- Migration 015: Site-Level Facts (Higher-Order Constructs)
-- Date: 2026-05-27
-- Purpose: Dedicated tables for site-level (allosteric, source-leak, etc.)
--          scientific outputs, following the lightweight dim_site pattern.
--          These complement the residue-level facts.
-- ============================================================================

CREATE TABLE IF NOT EXISTS fact_site_output (
    site_output_id      TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    run_id              TEXT NOT NULL REFERENCES provenance_run(run_id),
    structure_id        TEXT NOT NULL REFERENCES dim_structure(structure_id),
    site_id             TEXT NOT NULL REFERENCES dim_site(site_id),
    output_type         TEXT NOT NULL,                    -- e.g. 'allosteric_score', 'source_leak_potential'
    score               DOUBLE PRECISION,
    confidence          DOUBLE PRECISION,
    contributing_residues JSONB,                          -- list of residue_ids
    metadata            JSONB,
    source_type         TEXT NOT NULL,
    computed_at         TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_site_output_site ON fact_site_output(site_id);
CREATE INDEX IF NOT EXISTS idx_site_output_run ON fact_site_output(run_id);

-- Bridge / detailed per-residue contribution to a site (for explainability)
CREATE TABLE IF NOT EXISTS fact_site_residue_contribution (
    contribution_id     TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    site_output_id      TEXT NOT NULL REFERENCES fact_site_output(site_output_id) ON DELETE CASCADE,
    residue_id          TEXT NOT NULL REFERENCES dim_residue(residue_id),
    contribution_score  DOUBLE PRECISION,
    contribution_type   TEXT,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_site_contrib_residue ON fact_site_residue_contribution(residue_id);

-- ============================================================================
-- NOTES
-- ============================================================================
-- These tables support the `dim_site` concept from migration 001 and allow
-- higher-order biological constructs to have their own governed outputs
-- while remaining traceable to contributing residues and runs.