-- ============================================================================
-- Migration 012: Annotations and CDD Domain Data
-- Date: 2026-05-27
-- Purpose: External annotation data (CDD domains, etc.) linked to the
--          dimensional model, primarily at residue/segment level.
-- ============================================================================

CREATE TABLE IF NOT EXISTS fact_cdd_annotation (
    annotation_id       TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    run_id              TEXT NOT NULL REFERENCES provenance_run(run_id),
    structure_id        TEXT NOT NULL REFERENCES dim_structure(structure_id),
    chain_label         TEXT,
    domain_id           TEXT,
    domain_name         TEXT,
    start_residue       INTEGER,
    end_residue         INTEGER,
    e_value             DOUBLE PRECISION,
    bit_score           DOUBLE PRECISION,
    is_synthetic        BOOLEAN DEFAULT FALSE,            -- for stub/fallback annotations
    source_type         TEXT NOT NULL,                    -- 'external' for NCBI CDD
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cdd_structure ON fact_cdd_annotation(structure_id);
CREATE INDEX IF NOT EXISTS idx_cdd_run ON fact_cdd_annotation(run_id);

-- Generic annotation table for future extensibility (e.g., custom annotations, PTMs, etc.)
CREATE TABLE IF NOT EXISTS fact_generic_annotation (
    annotation_id       TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    run_id              TEXT NOT NULL REFERENCES provenance_run(run_id),
    structure_id        TEXT NOT NULL REFERENCES dim_structure(structure_id),
    residue_id          TEXT REFERENCES dim_residue(residue_id),
    annotation_type     TEXT NOT NULL,                    -- e.g. 'ptm', 'mutation', 'custom'
    annotation_key      TEXT,
    annotation_value    JSONB,
    source_type         TEXT NOT NULL,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_generic_annotation_residue ON fact_generic_annotation(residue_id);
CREATE INDEX IF NOT EXISTS idx_generic_annotation_run ON fact_generic_annotation(run_id);

-- ============================================================================
-- NOTES
-- ============================================================================
-- These tables allow external and custom annotations to be governed under
-- the same provenance and dimensional model as computed scientific data.
-- The generic annotation table provides a flexible escape hatch for future needs.