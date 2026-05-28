-- ============================================================================
-- Migration 003: Provenance Spine
-- Date: 2026-05-27
-- Purpose: Establish the core provenance model as a first-class, queryable
--          structure. Every governed asset should be traceable via run_id.
-- Aligned with: Core Data Principles (Provenance is Mandatory)
-- ============================================================================

-- Central provenance / run record
CREATE TABLE IF NOT EXISTS provenance_run (
    run_id              TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    structure_id        TEXT REFERENCES dim_structure(structure_id),
    model_version       TEXT NOT NULL,                    -- e.g. "GOSPConeMapper-v4", "DTIE-v3-full"
    checkpoint_uri      TEXT,
    checkpoint_sha256   TEXT,
    code_version        TEXT,                             -- git commit or equivalent
    pipeline_name       TEXT,
    orchestrator        TEXT,                             -- e.g. "local", "vertex-pipelines"
    run_type            TEXT NOT NULL DEFAULT 'inference',-- inference | training | analysis
    source_type         TEXT NOT NULL,                    -- deterministic | probabilistic | external | derived
    parameters          JSONB,                            -- e.g. n_landmarks, curvature, effector_sites
    warnings            JSONB,
    parent_run_id       TEXT REFERENCES provenance_run(run_id),
    started_at          TIMESTAMPTZ DEFAULT NOW(),
    completed_at        TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_provenance_run_structure ON provenance_run(structure_id);
CREATE INDEX IF NOT EXISTS idx_provenance_run_model ON provenance_run(model_version);
CREATE INDEX IF NOT EXISTS idx_provenance_run_type ON provenance_run(run_type);

-- Optional: Fine-grained provenance events within a run
-- (can be used for detailed step-by-step lineage if needed)
CREATE TABLE IF NOT EXISTS provenance_event (
    event_id            TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    run_id              TEXT NOT NULL REFERENCES provenance_run(run_id) ON DELETE CASCADE,
    event_type          TEXT NOT NULL,                    -- e.g. "ingestion", "phase_1", "gnn_inference"
    description         TEXT,
    input_refs          JSONB,                            -- references to upstream assets
    output_refs         JSONB,
    metadata            JSONB,
    event_time          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_provenance_event_run ON provenance_event(run_id);

-- ============================================================================
-- NOTES
-- ============================================================================
-- - All fact tables should carry a run_id referencing this table.
-- - run_type allows clear separation of training vs inference (per ADR-003).
-- - parent_run_id supports hierarchical provenance (e.g. sub-runs or ensemble runs).
-- - This is the foundation for the "provenance spine" described in the architecture.