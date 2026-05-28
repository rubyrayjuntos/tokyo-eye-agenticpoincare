-- ============================================================================
-- Migration 017: Common Helper Views (Residue-Centric Access)
-- Date: 2026-05-27
-- Purpose: Starter set of views that make common residue-level queries
--          simpler and more consistent across the platform.
--          These are lightweight and can be materialized later if needed.
-- ============================================================================

-- Latest embedding per residue (across all spaces)
CREATE OR REPLACE VIEW v_residue_latest_embeddings AS
WITH ranked AS (
    SELECT
        e.residue_id,
        e.space_id,
        e.embedding,
        e.cone_depth,
        e.epistemic_uncertainty,
        e.aleatoric_uncertainty,
        r.model_version,
        r.completed_at,
        ROW_NUMBER() OVER (PARTITION BY e.residue_id, e.space_id ORDER BY r.completed_at DESC) AS rn
    FROM fact_gnn_node_embedding e
    JOIN provenance_run r ON r.run_id = e.run_id
)
SELECT *
FROM ranked
WHERE rn = 1;

-- Residue + site membership summary
CREATE OR REPLACE VIEW v_residue_site_membership AS
SELECT
    r.residue_id,
    r.residue_index,
    r.residue_name,
    s.site_id,
    s.site_type,
    s.description AS site_description
FROM dim_residue r
LEFT JOIN bridge_site_residue bsr ON bsr.residue_id = r.residue_id
LEFT JOIN dim_site s ON s.site_id = bsr.site_id;

-- Basic provenance summary for a residue's latest GNN output
CREATE OR REPLACE VIEW v_residue_latest_gnn_provenance AS
SELECT
    e.residue_id,
    r.run_id,
    r.model_version,
    r.source_type,
    r.completed_at,
    e.space_id,
    e.epistemic_uncertainty,
    e.aleatoric_uncertainty
FROM fact_gnn_node_embedding e
JOIN provenance_run r ON r.run_id = e.run_id
WHERE (e.residue_id, e.space_id, r.completed_at) IN (
    SELECT residue_id, space_id, MAX(completed_at)
    FROM fact_gnn_node_embedding ee
    JOIN provenance_run rr ON rr.run_id = ee.run_id
    GROUP BY residue_id, space_id
);

-- ============================================================================
-- NOTES
-- ============================================================================
-- These views are starting points. They will be refined and expanded during
-- Phase 1 based on actual query patterns from the agent, visualizer, and
-- future RAG workloads. Some may later be converted to materialized views.