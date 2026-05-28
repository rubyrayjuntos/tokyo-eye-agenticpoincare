-- ============================================================================
-- Migration 008: GNN Node Clustering Coefficient
--
-- Adds clustering_coefficient to fact_gnn_node_output so that the per-node
-- graph topology metric computed by precompute_clustering() in Gnnv3.py is
-- persisted alongside the other GNN node outputs.
--
-- clustering_coefficient is a Cα-graph (8Å cutoff) metric: the fraction of a
-- residue's neighbours that are also neighbours of each other.  High values
-- identify tightly-packed structural cores; low values identify flexible
-- surface loops.  Useful for:
--   - Filtering allosteric pathway candidates (prefer low-clustering bridges)
--   - Weighting betweenness centrality scores
--   - Cross-referencing against fact_atom_feature sasa_percent values
-- ============================================================================

ALTER TABLE fact_gnn_node_output
    ADD COLUMN IF NOT EXISTS clustering_coefficient DOUBLE PRECISION;

CREATE INDEX IF NOT EXISTS idx_gnn_node_clustering
    ON fact_gnn_node_output(clustering_coefficient)
    WHERE clustering_coefficient IS NOT NULL;

COMMENT ON COLUMN fact_gnn_node_output.clustering_coefficient IS
    'Local clustering coefficient on the Cα 8Å radius graph. '
    'NULL for rows written before migration 008.';
