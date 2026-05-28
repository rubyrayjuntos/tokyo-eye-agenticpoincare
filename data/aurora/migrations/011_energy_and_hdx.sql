-- ============================================================================
-- Migration 011: Energy Calculations and HDX Correlation
-- Date: 2026-05-27
-- Purpose: Fact tables for energy/provenance steps and HDX-MS validation
--          correlation data. Follows residue-centric + provenance model.
-- ============================================================================

-- Energy / physics kernel step provenance (per-residue or per-structure)
CREATE TABLE IF NOT EXISTS fact_energy_provenance_step (
    energy_step_id      TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    run_id              TEXT NOT NULL REFERENCES provenance_run(run_id),
    structure_id        TEXT NOT NULL REFERENCES dim_structure(structure_id),
    residue_id          TEXT REFERENCES dim_residue(residue_id),  -- when per-residue
    step_name           TEXT NOT NULL,                    -- e.g. 'sasa', 'energy', 'packing'
    value               DOUBLE PRECISION,
    unit                TEXT,
    source_type         TEXT NOT NULL,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_energy_residue ON fact_energy_provenance_step(residue_id);
CREATE INDEX IF NOT EXISTS idx_energy_run ON fact_energy_provenance_step(run_id);

-- HDX-MS correlation data (often per-residue or per-peptide)
CREATE TABLE IF NOT EXISTS fact_hdx_correlation (
    hdx_id              TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    run_id              TEXT NOT NULL REFERENCES provenance_run(run_id),
    structure_id        TEXT NOT NULL REFERENCES dim_structure(structure_id),
    residue_id          TEXT REFERENCES dim_residue(residue_id),
    peptide_sequence    TEXT,
    protection_factor   DOUBLE PRECISION,
    correlation_score   DOUBLE PRECISION,
    source_type         TEXT NOT NULL,                    -- often 'external' (user-uploaded DynamX CSV)
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_hdx_residue ON fact_hdx_correlation(residue_id);
CREATE INDEX IF NOT EXISTS idx_hdx_run ON fact_hdx_correlation(run_id);

-- ============================================================================
-- NOTES
-- ============================================================================
-- These tables support both automated physics kernel steps and external
-- experimental validation data (HDX), while maintaining the same governance
-- and provenance standards as the rest of the model.