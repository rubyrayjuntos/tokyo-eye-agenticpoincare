-- ============================================================================
-- Migration 009: Folding Paths and Validation Mining Outputs
-- Date: 2026-05-27
-- Purpose: Tables for LERP folding trajectories and validation mining
--          results (outlier correlations, glue sites, wrapper suggestions).
--          Follows residue-centric + provenance-first model.
-- ============================================================================

-- LERP folding paths (large artifacts referenced from GCS, metadata in DB)
CREATE TABLE IF NOT EXISTS fact_folding_path (
    folding_path_id     TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    run_id              TEXT NOT NULL REFERENCES provenance_run(run_id),
    structure_id        TEXT NOT NULL REFERENCES dim_structure(structure_id),
    method              TEXT NOT NULL DEFAULT 'lerp',
    num_frames          INTEGER,
    num_atoms           INTEGER,
    gcs_uri             TEXT NOT NULL,             -- pointer to large binary
    checksum            TEXT,
    source_type         TEXT NOT NULL,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_folding_structure ON fact_folding_path(structure_id);

-- Validation mining outputs
CREATE TABLE IF NOT EXISTS fact_validation_mining_run (
    validation_run_id   TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    run_id              TEXT NOT NULL REFERENCES provenance_run(run_id),
    structure_id        TEXT NOT NULL REFERENCES dim_structure(structure_id),
    status              TEXT,
    duration_sec        DOUBLE PRECISION,
    started_at          TIMESTAMPTZ,
    completed_at        TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS fact_outlier_correlation (
    correlation_id      TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    validation_run_id   TEXT NOT NULL REFERENCES fact_validation_mining_run(validation_run_id),
    outlier_id          TEXT,
    dehydron_id         TEXT REFERENCES fact_dehydron(dehydron_id),
    distance            DOUBLE PRECISION,
    correlation_score   DOUBLE PRECISION,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS fact_glue_site (
    glue_site_id        TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    validation_run_id   TEXT NOT NULL REFERENCES fact_validation_mining_run(validation_run_id),
    structure_id        TEXT NOT NULL REFERENCES dim_structure(structure_id),
    avg_rho             DOUBLE PRECISION,
    void_volume_est     DOUBLE PRECISION,
    score               DOUBLE PRECISION,
    representative_label TEXT,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS fact_wrapper_suggestion (
    wrapper_suggestion_id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    glue_site_id          TEXT NOT NULL REFERENCES fact_glue_site(glue_site_id),
    wrapper_type          TEXT,
    wrapping_gain         DOUBLE PRECISION,
    predicted_ddg         DOUBLE PRECISION,
    created_at            TIMESTAMPTZ DEFAULT NOW()
);

-- Bridge for glue sites to dehydrons
CREATE TABLE IF NOT EXISTS bridge_glue_site_dehydron (
    glue_site_id        TEXT NOT NULL REFERENCES fact_glue_site(glue_site_id),
    dehydron_id         TEXT NOT NULL REFERENCES fact_dehydron(dehydron_id),
    PRIMARY KEY (glue_site_id, dehydron_id)
);

-- ============================================================================
-- NOTES
-- ============================================================================
-- These tables demonstrate how higher-order validation outputs (glue sites,
-- wrapper suggestions) can be modeled while still linking back to residue-level
-- facts (dehydrons) and the central provenance spine.
-- Large trajectory data stays in object storage (referenced by gcs_uri).