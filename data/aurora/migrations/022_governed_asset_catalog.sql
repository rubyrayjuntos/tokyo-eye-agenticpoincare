-- ============================================================================
-- Migration 022: Governed Asset Catalog (Foundational)
-- Date: 2026-05-27
-- Purpose: Core table for tracking all governed assets in the system.
--          This is the central catalog for provenance, discovery, and
--          governance queries. Supports the "Governed Asset" concept
--          from the architecture.
-- ============================================================================

CREATE TABLE IF NOT EXISTS governed_asset (
    asset_id            TEXT PRIMARY KEY,
    asset_type          TEXT NOT NULL,                    -- e.g. gnn_embedding, dtie_phase_result, folding_trajectory, annotation, etc.
    structure_id        TEXT REFERENCES dim_structure(structure_id),
    residue_id          TEXT REFERENCES dim_residue(residue_id),
    site_id             TEXT REFERENCES dim_site(site_id),
    run_id              TEXT NOT NULL REFERENCES provenance_run(run_id),
    storage_uri         TEXT,
    checksum            TEXT,
    access_level        TEXT NOT NULL DEFAULT 'internal',
    schema_name         TEXT,
    schema_version      TEXT,
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    metadata            JSONB
);

CREATE INDEX IF NOT EXISTS idx_governed_asset_type ON governed_asset(asset_type);
CREATE INDEX IF NOT EXISTS idx_governed_asset_residue ON governed_asset(residue_id);
CREATE INDEX IF NOT EXISTS idx_governed_asset_run ON governed_asset(run_id);
CREATE INDEX IF NOT EXISTS idx_governed_asset_structure ON governed_asset(structure_id);

-- ============================================================================
-- NOTES
-- ============================================================================
-- This table is intended to become the single source of truth for "what
-- governed data exists?" It will be populated by the normalizer / write
-- path. All future governed outputs should be registered here.