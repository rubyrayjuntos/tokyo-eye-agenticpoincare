-- ============================================================================
-- Migration 010: Folding Paths (Production Grade)
-- Date: 2026-05-27
-- Purpose: Proper table for LERP and other folding trajectory outputs.
--          Large binary data lives in object storage; this table holds
--          governed metadata + provenance.
-- ============================================================================

CREATE TABLE IF NOT EXISTS fact_folding_path (
    folding_path_id     TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    run_id              TEXT NOT NULL REFERENCES provenance_run(run_id),
    structure_id        TEXT NOT NULL REFERENCES dim_structure(structure_id),
    method              TEXT NOT NULL,                    -- 'lerp', 'md', etc.
    num_frames          INTEGER NOT NULL,
    num_atoms           INTEGER NOT NULL,
    gcs_uri             TEXT NOT NULL,                    -- pointer to the actual trajectory file
    checksum            TEXT,
    source_type         TEXT NOT NULL,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_folding_path_structure ON fact_folding_path(structure_id);
CREATE INDEX IF NOT EXISTS idx_folding_path_run ON fact_folding_path(run_id);

-- Optional: Per-frame metadata if we want to support partial queries later
CREATE TABLE IF NOT EXISTS fact_folding_frame (
    frame_id            TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    folding_path_id     TEXT NOT NULL REFERENCES fact_folding_path(folding_path_id) ON DELETE CASCADE,
    frame_index         INTEGER NOT NULL,
    rmsd_to_start       DOUBLE PRECISION,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_folding_frame_path ON fact_folding_frame(folding_path_id);