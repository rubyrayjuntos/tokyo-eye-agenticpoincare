-- ============================================================================
-- Migration 002: Example Fact Tables (Residue-Centric)
-- Date: 2026-05-27
-- Purpose: Demonstrate the preferred pattern for scientific outputs.
--          All examples join primarily on residue_id (per ADR-001).
--          These are illustrative — real fact tables will be refined in Phase 1.
-- ============================================================================

-- ============================================================================
-- Fact: GNN Node Output (Core scientific output at residue level)
-- ============================================================================
CREATE TABLE IF NOT EXISTS fact_gnn_node_output (
    node_output_id          TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    run_id                  TEXT NOT NULL,                    -- provenance link
    structure_id            TEXT NOT NULL REFERENCES dim_structure(structure_id),
    residue_id              TEXT NOT NULL REFERENCES dim_residue(residue_id),  -- primary grain
    model_version           TEXT NOT NULL,
    space_id                TEXT REFERENCES embedding_space(space_id),         -- which embedding space
    input_features          JSONB,
    projections             JSONB,                             -- Euclidean (backward compat)
    hyp_projections         JSONB,                             -- Native hyperbolic (v4+)
    embedding               VECTOR,                            -- primary vector for this space
    cone_depth              DOUBLE PRECISION,
    cone_width              DOUBLE PRECISION,
    epistemic_uncertainty   DOUBLE PRECISION,
    aleatoric_uncertainty   DOUBLE PRECISION,
    total_uncertainty       DOUBLE PRECISION,
    expert_weights          JSONB,
    source_type             TEXT NOT NULL,                     -- deterministic | probabilistic | derived
    computed_at             TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_gnn_node_residue ON fact_gnn_node_output(residue_id);
CREATE INDEX IF NOT EXISTS idx_gnn_node_run ON fact_gnn_node_output(run_id);
CREATE INDEX IF NOT EXISTS idx_gnn_node_structure ON fact_gnn_node_output(structure_id);
CREATE INDEX IF NOT EXISTS idx_gnn_node_space ON fact_gnn_node_output(space_id);

-- Vector index for similarity search (example)
-- CREATE INDEX IF NOT EXISTS idx_gnn_embedding ON fact_gnn_node_output USING ivfflat (embedding vector_cosine_ops);

-- ============================================================================
-- Fact: Phase Output (Generic pattern for DTIE phases)
-- ============================================================================
CREATE TABLE IF NOT EXISTS fact_phase_output (
    phase_output_id     TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    run_id              TEXT NOT NULL,
    structure_id        TEXT NOT NULL REFERENCES dim_structure(structure_id),
    residue_id          TEXT REFERENCES dim_residue(residue_id),  -- optional — some phases are structure-level
    phase               TEXT NOT NULL,                            -- e.g., "1", "2", "3", "3.5", "4b"
    phase_name          TEXT,
    output_data         JSONB NOT NULL,                           -- flexible payload
    source_type         TEXT NOT NULL,
    model_version       TEXT,
    computed_at         TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_phase_output_residue ON fact_phase_output(residue_id);
CREATE INDEX IF NOT EXISTS idx_phase_output_run ON fact_phase_output(run_id);
CREATE INDEX IF NOT EXISTS idx_phase_output_structure ON fact_phase_output(structure_id);
CREATE INDEX IF NOT EXISTS idx_phase_output_phase ON fact_phase_output(phase);

-- ============================================================================
-- Fact: Site Embedding (Example of higher-order / site-level data)
-- ============================================================================
CREATE TABLE IF NOT EXISTS fact_site_embedding (
    site_embedding_id   TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    run_id              TEXT NOT NULL,
    structure_id        TEXT NOT NULL REFERENCES dim_structure(structure_id),
    site_id             TEXT REFERENCES dim_site(site_id),
    space_id            TEXT REFERENCES embedding_space(space_id),
    embedding           VECTOR,
    cone_depth          DOUBLE PRECISION,
    source_type         TEXT NOT NULL,
    model_version       TEXT,
    computed_at         TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_site_embedding_site ON fact_site_embedding(site_id);
CREATE INDEX IF NOT EXISTS idx_site_embedding_run ON fact_site_embedding(run_id);

-- ============================================================================
-- NOTES
-- ============================================================================
-- - All fact tables include run_id for strong provenance.
-- - residue_id is the preferred grain (see ADR-001).
-- - embedding + space_id pattern supports multiple representation spaces.
-- - These are starting points. Real production tables will be refined
--   during Phase 1 with input from existing strong migrations
--   (especially 003_gnn_vector_embeddings.sql and 006_dtie_persistence.sql
--   from the ADK source).