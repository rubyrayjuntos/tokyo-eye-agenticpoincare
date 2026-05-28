-- ============================================================================
-- Migration 005: DTIE Phase Outputs (Example)
-- Date: 2026-05-27
-- Purpose: Representative fact tables for DTIE pipeline phases, consistently
--          anchored at residue level where appropriate.
--          Builds on the patterns established in 001–004.
-- ============================================================================

-- Phase 1: Witness Embedding (per structure, but with residue mapping)
CREATE TABLE IF NOT EXISTS fact_phase1_witness_embedding (
    phase1_id               TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    run_id                  TEXT NOT NULL REFERENCES provenance_run(run_id),
    structure_id            TEXT NOT NULL REFERENCES dim_structure(structure_id),
    n_witnesses             INTEGER,
    n_landmarks             INTEGER,
    curvature_c             DOUBLE PRECISION,
    witness_data            JSONB,                    -- coordinates, etc.
    landmark_to_residue     JSONB,
    source_type             TEXT NOT NULL,
    computed_at             TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_phase1_run ON fact_phase1_witness_embedding(run_id);

-- Phase 3: Persistence / Topological Output (can be residue-linked)
CREATE TABLE IF NOT EXISTS fact_phase3_persistence (
    phase3_id               TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    run_id                  TEXT NOT NULL REFERENCES provenance_run(run_id),
    structure_id            TEXT NOT NULL REFERENCES dim_structure(structure_id),
    residue_id              TEXT REFERENCES dim_residue(residue_id),  -- optional per-residue view
    barcode_length          INTEGER,
    max_alpha               DOUBLE PRECISION,
    persistence_data        JSONB,
    source_type             TEXT NOT NULL,
    computed_at             TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_phase3_residue ON fact_phase3_persistence(residue_id);
CREATE INDEX IF NOT EXISTS idx_phase3_run ON fact_phase3_persistence(run_id);

-- Dehydron facts (strongly residue-linked)
CREATE TABLE IF NOT EXISTS fact_dehydron (
    dehydron_id             TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    run_id                  TEXT NOT NULL REFERENCES provenance_run(run_id),
    structure_id            TEXT NOT NULL REFERENCES dim_structure(structure_id),
    donor_residue_id        TEXT NOT NULL REFERENCES dim_residue(residue_id),
    acceptor_residue_id     TEXT NOT NULL REFERENCES dim_residue(residue_id),
    wrapping_count          DOUBLE PRECISION,
    is_dehydron             BOOLEAN,
    midpoint_x              DOUBLE PRECISION,
    midpoint_y              DOUBLE PRECISION,
    midpoint_z              DOUBLE PRECISION,
    source_type             TEXT NOT NULL,
    computed_at             TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_dehydron_donor ON fact_dehydron(donor_residue_id);
CREATE INDEX IF NOT EXISTS idx_dehydron_acceptor ON fact_dehydron(acceptor_residue_id);
CREATE INDEX IF NOT EXISTS idx_dehydron_run ON fact_dehydron(run_id);

-- ============================================================================
-- NOTES
-- ============================================================================
-- - These are representative. Real production tables will be refined in Phase 1,
--   incorporating the strongest patterns from the ADK source (especially
--   006_dtie_persistence.sql) while enforcing the new residue-first + provenance
--   requirements.
-- - All tables link back to provenance_run for full traceability.