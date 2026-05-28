-- ============================================================================
-- Migration 020: Materialized View Candidates (Performance Layer)
-- Date: 2026-05-27
-- Purpose: Example materialized views for the most common high-frequency
--          queries (latest residue embeddings, site summaries, etc.).
--          These are candidates for Phase 1 or early Phase 2.
-- ============================================================================

-- Candidate: Latest hyperbolic embedding per residue (for viewport + agent)
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_residue_latest_hyperbolic AS
SELECT DISTINCT ON (residue_id)
    residue_id,
    embedding,
    cone_depth,
    epistemic_uncertainty,
    aleatoric_uncertainty,
    computed_at
FROM fact_gnn_node_embedding
WHERE space_id IN (
    SELECT space_id FROM embedding_space 
    WHERE space_type = 'hyperbolic' AND is_active = true
)
ORDER BY residue_id, computed_at DESC;

CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_residue_latest_hyperbolic 
    ON mv_residue_latest_hyperbolic(residue_id);

-- Candidate: Active site summary
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_active_site_summary AS
SELECT 
    s.site_id,
    s.structure_id,
    s.site_type,
    COUNT(bsr.residue_id) AS residue_count,
    AVG(e.epistemic_uncertainty) AS avg_epistemic
FROM dim_site s
LEFT JOIN bridge_site_residue bsr ON bsr.site_id = s.site_id
LEFT JOIN fact_gnn_node_embedding e ON e.residue_id = bsr.residue_id
GROUP BY s.site_id, s.structure_id, s.site_type;

-- ============================================================================
-- NOTES
-- ============================================================================
-- These are starting candidates only. Refresh strategy, indexing, and
-- actual need will be validated during Phase 1 based on real query patterns
-- from the agent and visualizer.