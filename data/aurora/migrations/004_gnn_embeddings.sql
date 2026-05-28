-- ============================================================================
-- Migration 004: GNN Embeddings & Vector Facts
-- Date: 2026-05-27
-- Purpose: Production-grade tables for GNN node embeddings and related vector
--          outputs, anchored at the residue level (per ADR-001).
--          Supports multiple embedding spaces (Euclidean + Hyperbolic).
-- ============================================================================

-- Core GNN node output / embedding table (refined from earlier example)
CREATE TABLE IF NOT EXISTS fact_gnn_node_embedding (
    embedding_id            TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    run_id                  TEXT NOT NULL REFERENCES provenance_run(run_id),
    structure_id            TEXT NOT NULL REFERENCES dim_structure(structure_id),
    residue_id              TEXT NOT NULL REFERENCES dim_residue(residue_id),  -- primary grain
    space_id                TEXT NOT NULL REFERENCES embedding_space(space_id),

    -- Input features (for auditability)
    input_rho               DOUBLE PRECISION,
    input_tau_flag          DOUBLE PRECISION,
    input_ss_type           DOUBLE PRECISION,
    input_sasa              DOUBLE PRECISION,

    -- Outputs
    embedding               VECTOR NOT NULL,                 -- the actual vector
    hyp_projections         JSONB,                           -- for 2D disc when applicable
    cone_depth              DOUBLE PRECISION,
    cone_width              DOUBLE PRECISION,
    epistemic_uncertainty   DOUBLE PRECISION,
    aleatoric_uncertainty   DOUBLE PRECISION,
    total_uncertainty       DOUBLE PRECISION,
    expert_weights          JSONB,

    source_type             TEXT NOT NULL,
    model_version           TEXT NOT NULL,
    computed_at             TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_gnn_emb_residue ON fact_gnn_node_embedding(residue_id);
CREATE INDEX IF NOT EXISTS idx_gnn_emb_run ON fact_gnn_node_embedding(run_id);
CREATE INDEX IF NOT EXISTS idx_gnn_emb_space ON fact_gnn_node_embedding(space_id);
CREATE INDEX IF NOT EXISTS idx_gnn_emb_structure ON fact_gnn_node_embedding(structure_id);

-- Optional: Clustering / graph features per residue (often computed alongside GNN)
CREATE TABLE IF NOT EXISTS fact_residue_graph_features (
    feature_id              TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    run_id                  TEXT NOT NULL REFERENCES provenance_run(run_id),
    structure_id            TEXT NOT NULL REFERENCES dim_structure(structure_id),
    residue_id              TEXT NOT NULL REFERENCES dim_residue(residue_id),
    clustering_coefficient  DOUBLE PRECISION,
    degree                  INTEGER,
    betweenness             DOUBLE PRECISION,
    source_type             TEXT NOT NULL,
    computed_at             TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_res_graph_feat_residue ON fact_residue_graph_features(residue_id);

-- ============================================================================
-- NOTES
-- ============================================================================
-- - This builds on 001 (dimensions) and 003 (provenance).
-- - embedding_space reference allows clean separation of Euclidean vs Hyperbolic outputs.
-- - Designed to be compatible with (and improved from) the vector embedding tables
--   in the ADK source while following the new residue-centric + provenance-first model.