-- ============================================================================
-- Migration 045: Hyperbolic Motif Persistence
-- Date: 2026-06-22
-- Purpose: Create fact_hyperbolic_motif table for storing recurring geometric
--          patterns discovered via HDBSCAN clustering in hyperbolic embedding
--          space. Supports cross-structure comparison and dashboard hydration.
-- Requirements: 8.1, 8.2, 8.3
-- ============================================================================

-- ============================================================================
-- 1. fact_hyperbolic_motif: Stores motif clusters identified by Poincaré
--    distance-based HDBSCAN clustering of hyperbolic embeddings.
-- ============================================================================

CREATE TABLE IF NOT EXISTS fact_hyperbolic_motif (
    motif_id            TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    structure_id        TEXT NOT NULL,
    run_id              TEXT NOT NULL,
    cluster_id          INTEGER NOT NULL,
    residue_ids         TEXT[] NOT NULL,
    medoid_residue_id   TEXT NOT NULL,
    centroid_angle_deg  DOUBLE PRECISION NOT NULL,
    centroid_radius     DOUBLE PRECISION NOT NULL,
    motif_size          INTEGER NOT NULL,
    angular_sector      TEXT NOT NULL,
    classification      TEXT,
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (run_id, cluster_id)
);

-- Primary query pattern: retrieve all motifs for a structure (dashboard hydration)
CREATE INDEX IF NOT EXISTS idx_fact_hyperbolic_motif_structure
    ON fact_hyperbolic_motif (structure_id);

-- Support provenance queries: find all motifs from a specific run
CREATE INDEX IF NOT EXISTS idx_fact_hyperbolic_motif_run
    ON fact_hyperbolic_motif (run_id);

-- ============================================================================
-- NOTES
-- ============================================================================
-- * The UNIQUE constraint on (run_id, cluster_id) supports idempotent upserts:
--   re-running motif analysis with the same run_id replaces previous results.
-- * residue_ids is a TEXT[] array of canonical residue keys (e.g. "4obe_A_12").
-- * angular_sector encodes the Poincaré disc region (e.g. "NW", "SE", "core").
-- * classification is optional and may contain labels like "beta_sheet_cluster",
--   "loop_motif", "helix_bundle", etc., assigned by heuristic or ML.
-- * centroid_angle_deg and centroid_radius describe the cluster center in polar
--   coordinates on the Poincaré disc.
