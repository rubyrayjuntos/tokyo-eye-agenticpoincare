-- ============================================================================
-- Migration 008: Immunogenicity and Redzone Facts
-- Date: 2026-05-27
-- Purpose: Fact tables for immunogenicity screening and metabolic liability
--          (redzone) outputs. Residue-anchored where appropriate, full
--          provenance linkage.
-- ============================================================================

-- Immunogenicity epitopes / runs (per structure, with residue links)
CREATE TABLE IF NOT EXISTS fact_immunogenicity_run (
    immunogenicity_run_id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    run_id                TEXT NOT NULL REFERENCES provenance_run(run_id),
    structure_id          TEXT NOT NULL REFERENCES dim_structure(structure_id),
    method                TEXT,                    -- e.g. NetMHCIIpan
    source_type           TEXT NOT NULL,
    computed_at           TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS fact_immunogenic_epitope (
    epitope_id            TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    immunogenicity_run_id TEXT NOT NULL REFERENCES fact_immunogenicity_run(immunogenicity_run_id),
    structure_id          TEXT NOT NULL REFERENCES dim_structure(structure_id),
    residue_id            TEXT REFERENCES dim_residue(residue_id),  -- primary anchor when applicable
    start_residue         INTEGER,
    end_residue           INTEGER,
    peptide_sequence      TEXT,
    score                 DOUBLE PRECISION,
    source_type           TEXT NOT NULL,
    created_at            TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_immuno_epitope_residue ON fact_immunogenic_epitope(residue_id);
CREATE INDEX IF NOT EXISTS idx_immuno_epitope_structure ON fact_immunogenic_epitope(structure_id);

-- Metabolic liability (CYP, etc.)
CREATE TABLE IF NOT EXISTS fact_metabolism_site (
    metabolism_site_id    TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    run_id                TEXT NOT NULL REFERENCES provenance_run(run_id),
    structure_id          TEXT NOT NULL REFERENCES dim_structure(structure_id),
    residue_id            TEXT REFERENCES dim_residue(residue_id),
    smart_pattern         TEXT,
    enzyme                TEXT,                    -- CYP450 isoform
    risk_score            DOUBLE PRECISION,
    source_type           TEXT NOT NULL,
    created_at            TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_metabolism_residue ON fact_metabolism_site(residue_id);

-- ============================================================================
-- NOTES
-- ============================================================================
-- These tables follow the same residue + provenance patterns as previous
-- migrations. They can later be extended with site-level bridging if needed.