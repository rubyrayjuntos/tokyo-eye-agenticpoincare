-- =============================================================================
-- Migration 034: RCSB Ingest Support
-- Date: 2026-05-28
-- Purpose: Add file_path and metadata columns to dim_structure for RCSB ingest,
--          and create a lightweight structures view for the agent.
-- =============================================================================

-- Add columns for tracking downloaded structure files
ALTER TABLE dim_structure ADD COLUMN IF NOT EXISTS file_path TEXT;
ALTER TABLE dim_structure ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}';
ALTER TABLE dim_structure ADD COLUMN IF NOT EXISTS ingested_at TIMESTAMPTZ;

-- Index for quick lookup by source
CREATE INDEX IF NOT EXISTS idx_dim_structure_source ON dim_structure(source);

-- View for the agent to list available structures
CREATE OR REPLACE VIEW v_agent_structures AS
SELECT
    structure_id,
    pdb_id,
    method,
    resolution,
    source,
    file_path,
    metadata,
    ingested_at,
    created_at
FROM dim_structure
ORDER BY ingested_at DESC NULLS LAST, created_at DESC;
