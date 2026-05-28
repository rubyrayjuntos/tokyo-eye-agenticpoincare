-- ============================================================================
-- Migration 001: Core Dimensional Model
-- Date: 2026-05-27
-- Aligned with: ADR-001 (Residue as Primary Granular Anchor)
-- Purpose: Establish the foundational dimensional structure with residue
--          as the primary analytical grain for scientific data.
-- ============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================================
-- DIMENSIONS
-- ============================================================================

-- Top-level structure
CREATE TABLE IF NOT EXISTS dim_structure (
    structure_id    TEXT PRIMARY KEY,
    pdb_id          TEXT,
    method          TEXT,
    resolution      DOUBLE PRECISION,
    source          TEXT,                    -- RCSB, AlphaFold, user, etc.
    deposited_date  DATE,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_dim_structure_pdb ON dim_structure(pdb_id);

-- Chain level
CREATE TABLE IF NOT EXISTS dim_chain (
    chain_id        TEXT PRIMARY KEY,
    structure_id    TEXT NOT NULL REFERENCES dim_structure(structure_id) ON DELETE CASCADE,
    chain_label     TEXT,
    entity_type     TEXT,                    -- protein, nucleic_acid, etc.
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_dim_chain_structure ON dim_chain(structure_id);

-- Residue level (PRIMARY ANALYTICAL GRAIN per ADR-001)
CREATE TABLE IF NOT EXISTS dim_residue (
    residue_id      TEXT PRIMARY KEY,
    chain_id        TEXT NOT NULL REFERENCES dim_chain(chain_id) ON DELETE CASCADE,
    residue_index   INTEGER NOT NULL,
    residue_name    TEXT,
    residue_name_3  TEXT,
    sse_code        TEXT,                    -- secondary structure
    sasa            DOUBLE PRECISION,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_dim_residue_chain ON dim_residue(chain_id);
CREATE INDEX IF NOT EXISTS idx_dim_residue_structure ON dim_residue(chain_id);  -- via join

-- Atom level (secondary, linked to residue)
CREATE TABLE IF NOT EXISTS dim_atom (
    atom_id         TEXT PRIMARY KEY,
    residue_id      TEXT NOT NULL REFERENCES dim_residue(residue_id) ON DELETE CASCADE,
    atom_name       TEXT,
    element         TEXT,
    x               DOUBLE PRECISION,
    y               DOUBLE PRECISION,
    z               DOUBLE PRECISION,
    occupancy       DOUBLE PRECISION,
    b_factor        DOUBLE PRECISION,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_dim_atom_residue ON dim_atom(residue_id);

-- Lightweight / derived "site" dimension for higher-order concepts
-- (allosteric sites, source leaks, high-uncertainty regions, etc.)
CREATE TABLE IF NOT EXISTS dim_site (
    site_id         TEXT PRIMARY KEY,
    structure_id    TEXT NOT NULL REFERENCES dim_structure(structure_id) ON DELETE CASCADE,
    site_type       TEXT,                    -- allosteric, source_leak, glue_site, etc.
    description     TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_dim_site_structure ON dim_site(structure_id);

-- Bridge: sites can span multiple residues
CREATE TABLE IF NOT EXISTS bridge_site_residue (
    site_id         TEXT NOT NULL REFERENCES dim_site(site_id) ON DELETE CASCADE,
    residue_id      TEXT NOT NULL REFERENCES dim_residue(residue_id) ON DELETE CASCADE,
    role            TEXT,                    -- e.g., "core", "peripheral"
    PRIMARY KEY (site_id, residue_id)
);

CREATE INDEX IF NOT EXISTS idx_bridge_site_residue ON bridge_site_residue(residue_id);

-- ============================================================================
-- EMBEDDING SPACE REGISTRY (Foundation for multi-space support)
-- ============================================================================

CREATE TABLE IF NOT EXISTS embedding_space (
    space_id        TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    space_type      TEXT NOT NULL,           -- euclidean | hyperbolic
    dimensionality  INTEGER NOT NULL,
    curvature       DOUBLE PRECISION,        -- only for hyperbolic
    model_name      TEXT,
    description     TEXT,
    is_active       BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_embedding_space_type ON embedding_space(space_type);

-- ============================================================================
-- NOTES
-- ============================================================================
-- - residue_id is the preferred join key for most scientific fact tables.
-- - dim_site is intentionally lightweight and derived.
-- - This migration focuses on dimensions only. Fact tables will follow in
--   subsequent migrations (e.g., 002_fact_tables.sql).
-- - Aligns with locked decisions from 2026-05-27.