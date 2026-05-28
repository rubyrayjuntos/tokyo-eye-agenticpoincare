-- ============================================================================
-- Migration 014: Flexible Computed Properties (Extensibility)
-- Date: 2026-05-27
-- Purpose: A flexible table for future computed properties and derived
--          metrics that can be attached at residue, site, or structure level.
--          This provides an escape hatch for extensibility without constant
--          schema changes during early development.
-- ============================================================================

CREATE TABLE IF NOT EXISTS fact_computed_property (
    property_id         TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    run_id              TEXT NOT NULL REFERENCES provenance_run(run_id),
    structure_id        TEXT NOT NULL REFERENCES dim_structure(structure_id),
    residue_id          TEXT REFERENCES dim_residue(residue_id),
    site_id             TEXT REFERENCES dim_site(site_id),
    property_name       TEXT NOT NULL,                    -- e.g. 'custom_score_v2', 'allosteric_potential'
    property_value      JSONB NOT NULL,
    property_type       TEXT,                             -- numeric, boolean, json, text, etc.
    source_type         TEXT NOT NULL,
    model_version       TEXT,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_computed_prop_residue ON fact_computed_property(residue_id);
CREATE INDEX IF NOT EXISTS idx_computed_prop_site ON fact_computed_property(site_id);
CREATE INDEX IF NOT EXISTS idx_computed_prop_name ON fact_computed_property(property_name);
CREATE INDEX IF NOT EXISTS idx_computed_prop_run ON fact_computed_property(run_id);

-- ============================================================================
-- NOTES
-- ============================================================================
-- - This table allows rapid addition of new derived metrics during research
--   phases without requiring new table migrations for every experiment.
-- - Over time, frequently used properties can be promoted to dedicated columns
--   or fact tables as the model stabilizes.
-- - Must still be governed via provenance_run (run_id is required).