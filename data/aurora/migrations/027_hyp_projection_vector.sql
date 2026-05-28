-- ============================================================================
-- Migration 027: Native Hyperbolic Projection Vector Column
-- Date: 2026-05-27
-- Purpose: Add a VECTOR(2) column for native 2D Poincaré disc projections
--          from v4 GNN. This enables vector similarity search directly on
--          disc coordinates without JSONB parsing.
-- Related: V4_OUTPUT_SHAPE_VALIDATION.md, Note 1
-- ============================================================================

-- Add native 2D disc projection as a proper vector column
ALTER TABLE fact_gnn_node_embedding
    ADD COLUMN IF NOT EXISTS hyp_projection_2d VECTOR(2);

-- Index for similarity search on disc coordinates
CREATE INDEX IF NOT EXISTS idx_gnn_emb_hyp_2d
    ON fact_gnn_node_embedding
    USING ivfflat (hyp_projection_2d vector_l2_ops)
    WHERE hyp_projection_2d IS NOT NULL;

-- ============================================================================
-- NOTES
-- ============================================================================
-- - This column stores the native 2D Poincaré disc projection from v4 GNN.
-- - The existing `hyp_projections` JSONB column is retained for backward
--   compatibility and for storing additional metadata alongside the projection.
-- - For v3 outputs, this column will be NULL (v3 does not produce native
--   disc projections).
-- - The IVFFlat index enables efficient nearest-neighbor search on the disc.
--   Note: For hyperbolic spaces, L2 distance on disc coordinates is an
--   approximation. True Poincaré distance requires the custom function
--   from the ADK source. This index is useful for fast approximate filtering.
