-- ============================================================================
-- Migration 031: Graph Topology Tables (Governed)
-- Date: 2026-05-27
-- Purpose: Create governed graph topology tables for persisting molecular
--          contact graph edges (with edge type classification) and per-node
--          graph-theoretic metrics. Replaces the earlier untyped fact_graph_edge
--          from migration 006 with a richer schema supporting edge_type,
--          hyperbolic distances, and natural-key idempotent upserts.
-- Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
-- ============================================================================

-- Drop the old untyped graph edge table (from migration 006) and its indexes.
-- The new schema is incompatible (adds edge_type to natural key, adds FK refs).
DROP TABLE IF EXISTS fact_graph_edge CASCADE;
DROP TABLE IF EXISTS fact_residue_graph_features CASCADE;

-- ============================================================================
-- FACT TABLE: Graph Edges (Governed)
-- Stores classified molecular contact graph edges with full provenance.
-- Natural key: (run_id, source_residue_id, target_residue_id, edge_type)
-- Requirements: 8.1, 8.3, 8.5
-- ============================================================================

CREATE TABLE IF NOT EXISTS fact_graph_edge (
    edge_id             TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    run_id              TEXT NOT NULL REFERENCES provenance_run(run_id),
    structure_id        TEXT NOT NULL REFERENCES dim_structure(structure_id),
    source_residue_id   TEXT NOT NULL REFERENCES dim_residue(residue_id),
    target_residue_id   TEXT NOT NULL REFERENCES dim_residue(residue_id),
    edge_type           TEXT NOT NULL,  -- 'h_bond', 'contact', 'covalent', 'disulfide', 'salt_bridge'
    distance_angstrom   DOUBLE PRECISION,
    hyperbolic_distance DOUBLE PRECISION,
    weight              DOUBLE PRECISION DEFAULT 1.0,
    metadata            JSONB,
    computed_at         TIMESTAMPTZ DEFAULT NOW()
);

-- Natural key unique index for idempotent upserts
CREATE UNIQUE INDEX uq_graph_edge_natural
    ON fact_graph_edge (run_id, source_residue_id, target_residue_id, edge_type);

-- Lookup indexes
CREATE INDEX idx_graph_edge_structure
    ON fact_graph_edge (structure_id);
CREATE INDEX idx_graph_edge_run
    ON fact_graph_edge (run_id);
CREATE INDEX idx_graph_edge_type
    ON fact_graph_edge (structure_id, edge_type);
CREATE INDEX idx_graph_edge_source_residue
    ON fact_graph_edge (source_residue_id);
CREATE INDEX idx_graph_edge_target_residue
    ON fact_graph_edge (target_residue_id);

COMMENT ON TABLE fact_graph_edge IS
    'Governed molecular contact graph edges with type classification. Writes via Normalizer only.';

-- ============================================================================
-- FACT TABLE: Graph Node Metrics
-- Stores per-residue graph-theoretic metrics computed from the contact graph.
-- Natural key: (run_id, residue_id)
-- Requirements: 8.2, 8.4, 8.5
-- ============================================================================

CREATE TABLE IF NOT EXISTS fact_graph_node_metrics (
    metric_id               TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    run_id                  TEXT NOT NULL REFERENCES provenance_run(run_id),
    structure_id            TEXT NOT NULL REFERENCES dim_structure(structure_id),
    residue_id              TEXT NOT NULL REFERENCES dim_residue(residue_id),
    degree                  INTEGER,
    betweenness             DOUBLE PRECISION,
    clustering_coefficient  DOUBLE PRECISION,
    closeness               DOUBLE PRECISION,
    eigenvector_centrality  DOUBLE PRECISION,
    is_bridge               BOOLEAN DEFAULT FALSE,
    conductance             DOUBLE PRECISION,
    computed_at             TIMESTAMPTZ DEFAULT NOW()
);

-- Natural key unique index for idempotent upserts
CREATE UNIQUE INDEX uq_graph_metrics_natural
    ON fact_graph_node_metrics (run_id, residue_id);

-- Lookup indexes
CREATE INDEX idx_graph_metrics_structure
    ON fact_graph_node_metrics (structure_id);
CREATE INDEX idx_graph_metrics_run
    ON fact_graph_node_metrics (run_id);
CREATE INDEX idx_graph_metrics_residue
    ON fact_graph_node_metrics (residue_id);
CREATE INDEX idx_graph_metrics_bridges
    ON fact_graph_node_metrics (structure_id, is_bridge)
    WHERE is_bridge = TRUE;

COMMENT ON TABLE fact_graph_node_metrics IS
    'Per-residue graph-theoretic metrics (degree, betweenness, bridges, etc.). Writes via Normalizer only.';
