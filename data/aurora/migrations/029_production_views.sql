-- ============================================================================
-- Migration 029: Production Access Views (Agent + Visualizer)
-- Date: 2026-05-27
-- Purpose: Purpose-built views for the two primary consumers:
--          1. Agent coordinator (needs residue state for decision-making)
--          2. Visualizer (needs embeddings + uncertainty for rendering)
--          These replace the earlier "candidate" views with focused,
--          production-grade implementations.
-- ============================================================================

-- ============================================================================
-- AGENT VIEWS (decision-making, workflow orchestration)
-- ============================================================================

-- Agent needs: "For this structure, what is the current scientific state
-- of each residue?" (latest embeddings, uncertainty, site membership, dehydron)
CREATE OR REPLACE VIEW v_agent_residue_state AS
SELECT
    r.residue_id,
    r.residue_index,
    r.residue_name,
    c.chain_label,
    s.structure_id,
    s.pdb_id,
    -- Latest GNN output (prefer hyperbolic if available)
    latest_emb.space_id,
    latest_emb.cone_depth,
    latest_emb.cone_width,
    latest_emb.epistemic_uncertainty,
    latest_emb.aleatoric_uncertainty,
    latest_emb.total_uncertainty,
    latest_emb.model_version,
    latest_emb.computed_at AS embedding_computed_at,
    -- Dehydron status
    COALESCE(dehydron_agg.dehydron_count, 0) AS dehydron_count,
    dehydron_agg.min_wrapping AS min_wrapping_count,
    -- Site membership
    site_agg.site_types,
    site_agg.site_count
FROM dim_residue r
JOIN dim_chain c ON c.chain_id = r.chain_id
JOIN dim_structure s ON s.structure_id = c.structure_id
LEFT JOIN LATERAL (
    SELECT e.space_id, e.cone_depth, e.cone_width,
           e.epistemic_uncertainty, e.aleatoric_uncertainty,
           e.total_uncertainty, e.model_version, e.computed_at
    FROM fact_gnn_node_embedding e
    JOIN embedding_space es ON es.space_id = e.space_id
    WHERE e.residue_id = r.residue_id
    ORDER BY
        CASE WHEN es.space_type = 'hyperbolic' THEN 0 ELSE 1 END,
        e.computed_at DESC
    LIMIT 1
) latest_emb ON true
LEFT JOIN LATERAL (
    SELECT
        COUNT(*) AS dehydron_count,
        MIN(wrapping_count) AS min_wrapping
    FROM fact_dehydron d
    WHERE (d.donor_residue_id = r.residue_id OR d.acceptor_residue_id = r.residue_id)
      AND d.is_dehydron = true
) dehydron_agg ON true
LEFT JOIN LATERAL (
    SELECT
        ARRAY_AGG(DISTINCT site.site_type) AS site_types,
        COUNT(DISTINCT site.site_id) AS site_count
    FROM bridge_site_residue bsr
    JOIN dim_site site ON site.site_id = bsr.site_id
    WHERE bsr.residue_id = r.residue_id
) site_agg ON true;

-- Agent needs: "What are the high-uncertainty residues for this structure?"
-- (for directing investigation / source-leak detection)
CREATE OR REPLACE VIEW v_agent_high_uncertainty_residues AS
SELECT
    r.residue_id,
    r.residue_index,
    r.residue_name,
    c.chain_label,
    s.structure_id,
    s.pdb_id,
    e.epistemic_uncertainty,
    e.aleatoric_uncertainty,
    e.total_uncertainty,
    e.cone_depth,
    e.model_version,
    prov.run_id,
    prov.pipeline_name
FROM fact_gnn_node_embedding e
JOIN dim_residue r ON r.residue_id = e.residue_id
JOIN dim_chain c ON c.chain_id = r.chain_id
JOIN dim_structure s ON s.structure_id = c.structure_id
JOIN provenance_run prov ON prov.run_id = e.run_id
WHERE e.epistemic_uncertainty IS NOT NULL
  AND prov.run_type = 'inference'
ORDER BY e.epistemic_uncertainty DESC;

-- Agent needs: "What runs have been executed for this structure?"
CREATE OR REPLACE VIEW v_agent_structure_runs AS
SELECT
    pr.structure_id,
    pr.run_id,
    pr.model_version,
    pr.pipeline_name,
    pr.run_type,
    pr.source_type,
    pr.started_at,
    pr.completed_at,
    pr.parent_run_id,
    COUNT(ga.asset_id) AS assets_produced
FROM provenance_run pr
LEFT JOIN governed_asset ga ON ga.run_id = pr.run_id
GROUP BY pr.run_id, pr.structure_id, pr.model_version, pr.pipeline_name,
         pr.run_type, pr.source_type, pr.started_at, pr.completed_at, pr.parent_run_id
ORDER BY pr.started_at DESC;

-- ============================================================================
-- VISUALIZER VIEWS (rendering, viewport directives)
-- ============================================================================

-- Visualizer needs: "Give me all hyperbolic embeddings for this structure
-- so I can render the Poincaré disc."
CREATE OR REPLACE VIEW v_viz_poincare_disc_data AS
SELECT
    r.residue_id,
    r.residue_index,
    r.residue_name,
    c.chain_label,
    e.hyp_projections,
    e.hyp_projection_2d,
    e.cone_depth,
    e.cone_width,
    e.epistemic_uncertainty,
    e.model_version,
    e.computed_at,
    es.curvature,
    es.name AS space_name
FROM fact_gnn_node_embedding e
JOIN dim_residue r ON r.residue_id = e.residue_id
JOIN dim_chain c ON c.chain_id = r.chain_id
JOIN embedding_space es ON es.space_id = e.space_id
WHERE es.space_type = 'hyperbolic'
  AND es.is_active = true
ORDER BY r.residue_index;

-- Visualizer needs: "What sites exist for this structure and which residues
-- are in each?" (for overlay rendering)
CREATE OR REPLACE VIEW v_viz_site_overlay AS
SELECT
    s.site_id,
    s.site_type,
    s.description AS site_description,
    s.structure_id,
    r.residue_id,
    r.residue_index,
    r.residue_name,
    c.chain_label,
    bsr.role AS residue_role
FROM dim_site s
JOIN bridge_site_residue bsr ON bsr.site_id = s.site_id
JOIN dim_residue r ON r.residue_id = bsr.residue_id
JOIN dim_chain c ON c.chain_id = r.chain_id
ORDER BY s.site_type, s.site_id, r.residue_index;

-- Visualizer needs: "Give me dehydron pairs for rendering hydrogen bond overlays."
CREATE OR REPLACE VIEW v_viz_dehydron_pairs AS
SELECT
    d.dehydron_id,
    d.structure_id,
    d.donor_residue_id,
    d.acceptor_residue_id,
    d.wrapping_count,
    d.is_dehydron,
    d.midpoint_x,
    d.midpoint_y,
    d.midpoint_z,
    dr.residue_index AS donor_index,
    dr.residue_name AS donor_name,
    ar.residue_index AS acceptor_index,
    ar.residue_name AS acceptor_name
FROM fact_dehydron d
JOIN dim_residue dr ON dr.residue_id = d.donor_residue_id
JOIN dim_residue ar ON ar.residue_id = d.acceptor_residue_id
WHERE d.is_dehydron = true;

-- ============================================================================
-- PROVENANCE VIEWS (audit, lineage, governance)
-- ============================================================================

-- "Show me the provenance chain for a specific run (flattened)."
CREATE OR REPLACE VIEW v_provenance_chain AS
WITH RECURSIVE chain AS (
    SELECT run_id, model_version, pipeline_name, run_type,
           parent_run_id, started_at, 0 AS depth
    FROM provenance_run

    UNION ALL

    SELECT p.run_id, p.model_version, p.pipeline_name, p.run_type,
           p.parent_run_id, p.started_at, c.depth + 1
    FROM provenance_run p
    JOIN chain c ON p.run_id = c.parent_run_id
    WHERE c.depth < 10
)
SELECT * FROM chain;

-- "Show me recent normalization failures for monitoring."
CREATE OR REPLACE VIEW v_normalization_failures AS
SELECT
    audit_id,
    run_id,
    structure_id,
    payload_type,
    status,
    error_message,
    duration_ms,
    caller_identity,
    created_at
FROM normalization_audit
WHERE status != 'success'
ORDER BY created_at DESC;

-- "Asset governance dashboard: what do we have?"
CREATE OR REPLACE VIEW v_governance_dashboard AS
SELECT
    asset_type,
    COUNT(*) AS total_assets,
    COUNT(DISTINCT structure_id) AS structures_covered,
    COUNT(DISTINCT run_id) AS producing_runs,
    MIN(created_at) AS earliest_asset,
    MAX(created_at) AS latest_asset
FROM governed_asset
GROUP BY asset_type
ORDER BY total_assets DESC;

-- ============================================================================
-- NOTES
-- ============================================================================
-- These views are designed for the two primary consumers (agent + visualizer)
-- and for operational governance. They should be:
-- - Tested against realistic data volumes in Phase 2
-- - Candidates for materialization if query latency becomes an issue
-- - Extended as new access patterns emerge from actual usage
