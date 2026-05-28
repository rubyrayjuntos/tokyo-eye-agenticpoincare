-- ============================================================================
-- Migration 021: Residue Summary Materialized View (Performance)
-- Date: 2026-05-27
-- Purpose: High-frequency query support for common residue-level data
--          needed by the agent, visualizer, and future RAG systems.
-- ============================================================================

CREATE MATERIALIZED VIEW IF NOT EXISTS mv_residue_current_state AS
SELECT
    r.residue_id,
    r.residue_index,
    r.residue_name,
    s.structure_id,
    s.pdb_id,
    -- Latest embedding (any space, prefer hyperbolic if available)
    e.space_id,
    e.embedding,
    e.cone_depth,
    e.epistemic_uncertainty,
    e.aleatoric_uncertainty,
    -- Dehydron status (simplified)
    EXISTS (
        SELECT 1 FROM fact_dehydron d
        WHERE d.donor_residue_id = r.residue_id OR d.acceptor_residue_id = r.residue_id
    ) AS has_dehydron,
    -- Site membership
    ARRAY_AGG(DISTINCT site.site_type) FILTER (WHERE site.site_id IS NOT NULL) AS active_site_types
FROM dim_residue r
JOIN dim_chain c ON c.chain_id = r.chain_id
JOIN dim_structure s ON s.structure_id = c.structure_id
LEFT JOIN LATERAL (
    SELECT * FROM fact_gnn_node_embedding ee
    ORDER BY ee.computed_at DESC
    LIMIT 1
) e ON e.residue_id = r.residue_id
LEFT JOIN bridge_site_residue bsr ON bsr.residue_id = r.residue_id
LEFT JOIN dim_site site ON site.site_id = bsr.site_id
GROUP BY r.residue_id, r.residue_index, r.residue_name, s.structure_id, s.pdb_id,
         e.space_id, e.embedding, e.cone_depth, e.epistemic_uncertainty, e.aleatoric_uncertainty;

CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_residue_current_state_residue 
    ON mv_residue_current_state(residue_id);

-- Refresh strategy note: This should be refreshed after major pipeline runs
-- or on a schedule. For now it is a candidate for Phase 1/2.