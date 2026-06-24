-- ============================================================================
-- Migration 033: Phase Persistence Tier 1 Fact Tables
-- Date: 2026-05-28
-- Purpose: Dedicated fact tables for Tier 1 phase outputs that require
--          full provenance and rich queryable columns: source-leak detection
--          and allosteric site identification.
-- Requirements: 6.1, 6.2, 6.3, 6.4, 6.5
-- ============================================================================

-- ============================================================================
-- FACT TABLE: Source-Leak Candidates (per-residue)
-- Stores residues flagged as source leaks with epistemic uncertainty metrics.
-- Natural key: (run_id, residue_id) — one record per residue per run.
-- Requirements: 6.1, 6.4, 6.5
-- ============================================================================

CREATE TABLE IF NOT EXISTS fact_source_leak (
    leak_id                 TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    run_id                  TEXT NOT NULL REFERENCES provenance_run(run_id),
    structure_id            TEXT NOT NULL REFERENCES dim_structure(structure_id),
    residue_id              TEXT NOT NULL REFERENCES dim_residue(residue_id),
    epistemic_uncertainty   DOUBLE PRECISION NOT NULL,
    cone_depth              DOUBLE PRECISION NOT NULL,
    leak_score              DOUBLE PRECISION NOT NULL,
    is_confirmed            BOOLEAN DEFAULT FALSE,
    computed_at             TIMESTAMPTZ DEFAULT NOW()
);

-- Idempotent upsert key: one source-leak record per residue per run
CREATE UNIQUE INDEX IF NOT EXISTS uq_source_leak_natural
    ON fact_source_leak(run_id, residue_id);

-- Query indexes for common access patterns
CREATE INDEX IF NOT EXISTS idx_source_leak_structure
    ON fact_source_leak(structure_id);
CREATE INDEX IF NOT EXISTS idx_source_leak_run
    ON fact_source_leak(run_id);
CREATE INDEX IF NOT EXISTS idx_source_leak_residue
    ON fact_source_leak(residue_id);

-- ============================================================================
-- FACT TABLE: Allosteric Site Predictions
-- Stores predicted allosteric pockets with centroid and confidence.
-- Natural key: (run_id, site_id) — one record per site per run.
-- Requirements: 6.2, 6.4, 6.5
-- ============================================================================

CREATE TABLE IF NOT EXISTS fact_allosteric_site (
    allosteric_id           TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    site_id                 TEXT NOT NULL,
    run_id                  TEXT NOT NULL REFERENCES provenance_run(run_id),
    structure_id            TEXT NOT NULL REFERENCES dim_structure(structure_id),
    centroid_x              DOUBLE PRECISION,
    centroid_y              DOUBLE PRECISION,
    centroid_z              DOUBLE PRECISION,
    confidence_score        DOUBLE PRECISION,
    cluster_method          TEXT,
    n_residues              INTEGER,
    computed_at             TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (run_id, site_id)
);

-- Query indexes for common access patterns
CREATE INDEX IF NOT EXISTS idx_allosteric_site_structure
    ON fact_allosteric_site(structure_id);
CREATE INDEX IF NOT EXISTS idx_allosteric_site_run
    ON fact_allosteric_site(run_id);
CREATE INDEX IF NOT EXISTS idx_allosteric_site_site_id
    ON fact_allosteric_site(site_id);

-- ============================================================================
-- JUNCTION TABLE: Allosteric Site ↔ Residue Membership
-- Links each allosteric site to its constituent residues (many-to-many).
-- References the natural key (run_id, site_id) to tie membership to a
-- specific detection run.
-- Requirements: 6.3, 6.4
-- ============================================================================

CREATE TABLE IF NOT EXISTS fact_allosteric_site_residue (
    id                      TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    run_id                  TEXT NOT NULL,
    site_id                 TEXT NOT NULL,
    residue_id              TEXT NOT NULL REFERENCES dim_residue(residue_id),
    contribution_score      DOUBLE PRECISION,
    FOREIGN KEY (run_id, site_id)
        REFERENCES fact_allosteric_site(run_id, site_id)
        ON DELETE CASCADE
);

-- Unique constraint: one membership record per site-residue pair per run
CREATE UNIQUE INDEX IF NOT EXISTS uq_site_residue
    ON fact_allosteric_site_residue(run_id, site_id, residue_id);

-- Query indexes
CREATE INDEX IF NOT EXISTS idx_site_residue_site
    ON fact_allosteric_site_residue(site_id);
CREATE INDEX IF NOT EXISTS idx_site_residue_residue
    ON fact_allosteric_site_residue(residue_id);
CREATE INDEX IF NOT EXISTS idx_site_residue_run
    ON fact_allosteric_site_residue(run_id);

-- ============================================================================
-- NOTES
-- ============================================================================
-- - fact_source_leak uses ON CONFLICT (run_id, residue_id) DO NOTHING/UPDATE
--   at the application layer for idempotent upserts (Requirement 6.5).
-- - fact_allosteric_site uses ON CONFLICT (run_id, site_id) DO NOTHING/UPDATE
--   at the application layer for idempotent upserts (Requirement 6.5).
-- - All tables reference provenance_run(run_id) for full traceability.
-- - These dedicated tables replace the generic fact_site_output approach for
--   Tier 1 phases, providing richer queryable columns per the design spec.
