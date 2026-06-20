-- ============================================================================
-- Migration 044: Binding Site Scan at Ingestion
-- Date: 2026-06-19
-- Purpose: Create fact_cryptic_site table for storing pre-computed binding site
--          candidates (cryptic and surface), and fact_binding_site_scan for scan
--          run metadata. Supports the exhaustive scan phase that runs after GNN
--          inference and graph topology computation.
-- Requirements: 3.4, 4.4
-- ============================================================================

-- ============================================================================
-- 1. fact_cryptic_site: Stores all candidate binding sites discovered during
--    the full-structure scan phase (GNN-derived, geometry-derived, or hybrid).
-- ============================================================================

CREATE TABLE IF NOT EXISTS fact_cryptic_site (
    site_id                 TEXT PRIMARY KEY,
    structure_id            TEXT NOT NULL,
    run_id                  TEXT NOT NULL,
    residue_ids             JSONB NOT NULL,
    centroid_x              DOUBLE PRECISION NOT NULL,
    centroid_y              DOUBLE PRECISION NOT NULL,
    centroid_z              DOUBLE PRECISION NOT NULL,
    site_type               TEXT NOT NULL,
    discovery_method        TEXT NOT NULL DEFAULT 'gnn_strain',
    druggability_score      DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    site_rank               INTEGER,
    composite_gnn_score     DOUBLE PRECISION,
    fpocket_druggability    DOUBLE PRECISION,
    volume_angstrom3        DOUBLE PRECISION,
    provenance_gate         TEXT,
    heuristic_version       TEXT,
    md_validation_status    TEXT NOT NULL DEFAULT 'pending',
    scan_run_id             TEXT,
    created_at              TIMESTAMPTZ DEFAULT NOW()
);

-- Primary query pattern: retrieve all sites for a structure ordered by rank
CREATE INDEX IF NOT EXISTS idx_fact_cryptic_site_structure_rank
    ON fact_cryptic_site (structure_id, site_rank);

-- Support filtering by site_type
CREATE INDEX IF NOT EXISTS idx_fact_cryptic_site_structure_type
    ON fact_cryptic_site (structure_id, site_type);

-- Support filtering by md_validation_status
CREATE INDEX IF NOT EXISTS idx_fact_cryptic_site_md_status
    ON fact_cryptic_site (structure_id, md_validation_status);

-- Support cleanup of scan results during re-scan
CREATE INDEX IF NOT EXISTS idx_fact_cryptic_site_scan_run
    ON fact_cryptic_site (structure_id, scan_run_id);

-- ============================================================================
-- 2. fact_binding_site_scan: Stores metadata about each full-structure scan run
--    for provenance and reproducibility tracking.
-- ============================================================================

CREATE TABLE IF NOT EXISTS fact_binding_site_scan (
    scan_id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    structure_id            TEXT NOT NULL,
    run_id                  TEXT NOT NULL,
    heuristic_version       TEXT NOT NULL,
    model_version           TEXT NOT NULL,
    scan_parameters         JSONB NOT NULL,
    sites_found             INTEGER NOT NULL DEFAULT 0,
    duration_ms             INTEGER,
    status                  TEXT NOT NULL DEFAULT 'complete',
    created_at              TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (structure_id, run_id)
);

-- Lookup scans by structure for history
CREATE INDEX IF NOT EXISTS idx_binding_site_scan_structure
    ON fact_binding_site_scan (structure_id);

-- ============================================================================
-- 3. Constraint: md_validation_status values
-- ============================================================================

ALTER TABLE fact_cryptic_site
    ADD CONSTRAINT chk_md_validation_status
    CHECK (md_validation_status IN ('pending', 'running', 'passed', 'failed', 'timeout'));

-- ============================================================================
-- 4. Constraint: discovery_method values
-- ============================================================================

ALTER TABLE fact_cryptic_site
    ADD CONSTRAINT chk_discovery_method
    CHECK (discovery_method IN ('gnn_strain', 'geometry', 'hybrid'));

-- ============================================================================
-- NOTES
-- ============================================================================
-- * fact_cryptic_site is the central table for all pre-computed binding site
--   candidates. The scan_run_id links back to fact_binding_site_scan for
--   provenance. Sites created via on-demand discovery (non-scan) will have
--   scan_run_id = NULL.
-- * The (structure_id, site_rank) index is the primary access pattern for the
--   query interface (Requirement 5.1).
-- * md_validation_status follows the state machine:
--   pending → running → passed | failed | timeout
--   passed | failed → pending (re-validation)
-- * druggability_score is [0, 1] — enforced at application level.
-- * site_rank is 1-indexed, with rank 1 = most druggable.
