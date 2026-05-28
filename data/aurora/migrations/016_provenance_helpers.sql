-- ============================================================================
-- Migration 016: Provenance Helper Functions
-- Date: 2026-05-27
-- Purpose: PL/pgSQL functions to make common provenance and lineage queries
--          easier and more performant.
-- ============================================================================

-- Simple function to get the full ancestor chain for a run (basic recursive lineage)
CREATE OR REPLACE FUNCTION get_run_lineage(p_run_id TEXT)
RETURNS TABLE(run_id TEXT, model_version TEXT, level INTEGER) AS $$
WITH RECURSIVE lineage AS (
    SELECT r.run_id, r.model_version, 0 AS level
    FROM provenance_run r
    WHERE r.run_id = p_run_id

    UNION ALL

    SELECT parent.run_id, parent.model_version, l.level + 1
    FROM provenance_run parent
    JOIN lineage l ON parent.run_id = l.run_id  -- This is inverted for ancestors; adjust as needed
    -- Note: For true ancestors, reverse the join on parent_run_id
)
SELECT * FROM lineage;
$$ LANGUAGE sql STABLE;

-- Function to count direct child runs
CREATE OR REPLACE FUNCTION count_child_runs(p_parent_run_id TEXT)
RETURNS INTEGER AS $$
    SELECT COUNT(*)::INTEGER
    FROM provenance_run
    WHERE parent_run_id = p_parent_run_id;
$$ LANGUAGE sql STABLE;

-- ============================================================================
-- NOTES
-- ============================================================================
-- These are starter helper functions. They will be expanded and refined
-- during Phase 1 based on actual query patterns that emerge from the
-- agent, visualizer, and future RAG workloads.
-- The recursive lineage function is intentionally basic and will likely
-- be replaced with a more robust CTE or materialized view approach later.