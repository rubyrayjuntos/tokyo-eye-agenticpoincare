-- ============================================================================
-- Migration 040: Add V6 expert routing metadata to GNN embeddings
-- Date: 2026-05-31
-- Purpose: Persist per-run expert_load and routing_entropy alongside GNN
--          node embeddings. These are v6 MoE routing statistics that enable
--          specialization validation and monitoring.
--
-- expert_load: JSONB array of 4 floats — mean routing probability per expert
-- routing_entropy: scalar — Shannon entropy of the routing distribution
--
-- Both are per-run aggregates stored on each node row for query simplicity
-- (avoids an extra join for the most common access pattern).
-- ============================================================================

ALTER TABLE fact_gnn_node_embedding
    ADD COLUMN IF NOT EXISTS expert_load_per_run JSONB,
    ADD COLUMN IF NOT EXISTS routing_entropy DOUBLE PRECISION;

COMMENT ON COLUMN fact_gnn_node_embedding.expert_load_per_run IS
    'V6: Per-expert mean routing probability for this run [4 floats]. NULL for v5 runs.';

COMMENT ON COLUMN fact_gnn_node_embedding.routing_entropy IS
    'V6: Shannon entropy of the expert routing distribution for this run. NULL for v5 runs.';

-- Index for monitoring queries that filter by routing entropy
CREATE INDEX IF NOT EXISTS idx_gnn_emb_routing_entropy
    ON fact_gnn_node_embedding (routing_entropy)
    WHERE routing_entropy IS NOT NULL;
