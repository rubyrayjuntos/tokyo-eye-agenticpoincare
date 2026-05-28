-- ============================================================================
-- Migration 019: Common Residue-Level Views
-- Date: 2026-05-27
-- Purpose: Production-oriented views that combine residue dimensions with
--          the most frequently accessed metrics (latest embeddings, dehydron
--          status, site membership, uncertainty, etc.).
-- ============================================================================

CREATE OR REPLACE VIEW v_residue_current_state AS
SELECT
    r.residue_id,
    r.residue_index,
    r.residue_name,
    s.structure_id,
    s.pdb_id,
    e.space_id,
    e.embedding,
    e.cone_depth,
    e.epistemic_uncertainty,
    e.aleatoric_uncertainty,
    d.is_dehydron,
    d.wrapping_count,
    ARRAY_AGG(DISTINCT site.site_type) FILTER (WHERE site.site_id IS NOT NULL) AS site_types
FROM dim_residue r
JOIN dim_chain c ON c.chain_id = r.chain_id
JOIN dim_structure s ON s.structure_id = c.structure_id
LEFT JOIN LATERAL (
    SELECT * FROM fact_gnn_node_embedding ee
    WHERE ee.residue_id = r.residue_id
    ORDER BY ee.computed_at DESC
    LIMIT 1
) e ON true
LEFT JOIN fact_dehydron d 
    ON d.donor_residue_id = r.residue_id OR d.acceptor_residue_id = r.residue_id
LEFT JOIN bridge_site_residue bsr ON bsr.residue_id = r.residue_id
LEFT JOIN dim_site site ON site.site_id = bsr.site_id
GROUP BY r.residue_id, r.residue_index, r.residue_name, s.structure_id, s.pdb_id, 
         e.space_id, e.embedding, e.cone_depth, e.epistemic_uncertainty, 
         e.aleatoric_uncertainty, d.is_dehydron, d.wrapping_count;

-- ============================================================================
-- NOTES
-- ============================================================================
-- This view is intentionally broad. It will likely be split into more focused
-- views (e.g., v_residue_latest_hyperbolic, v_residue_dehydron_status) during Phase 1.