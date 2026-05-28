-- ============================================================================
-- Migration 030: Add unique constraint on natural key for GNN embeddings
-- Date: 2026-05-27
-- Purpose: Enable idempotent upserts on (run_id, residue_id, space_id) —
--          the natural key for "this residue's embedding from this run in
--          this space". This replaces the previous ON CONFLICT (embedding_id)
--          pattern which could never trigger (embedding_id is always a fresh UUID).
-- ============================================================================

-- Add unique constraint on the natural key
CREATE UNIQUE INDEX IF NOT EXISTS uq_gnn_emb_run_residue_space
    ON fact_gnn_node_embedding (run_id, residue_id, space_id);

COMMENT ON INDEX uq_gnn_emb_run_residue_space IS
    'Natural key for idempotent upserts: one embedding per (run, residue, space).';
