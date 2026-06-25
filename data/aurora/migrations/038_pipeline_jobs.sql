-- Migration 038: Persist pipeline jobs to survive server restarts
-- Reconciles older local schemas with the current dashboard contract.

CREATE TABLE IF NOT EXISTS pipeline_job (
    job_id          TEXT PRIMARY KEY,
    structure_id    TEXT NOT NULL REFERENCES dim_structure(structure_id),
    status          TEXT NOT NULL DEFAULT 'queued'
                    CHECK (status IN ('queued', 'running', 'complete', 'failed')),
    current_step    TEXT NOT NULL DEFAULT 'ingestion',
    progress        DOUBLE PRECISION NOT NULL DEFAULT 0 CHECK (progress >= 0 AND progress <= 100),
    modules         JSONB NOT NULL DEFAULT '[]'::jsonb,
    error           TEXT,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at    TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE pipeline_job
    ALTER COLUMN job_id TYPE TEXT USING job_id::text,
    ALTER COLUMN structure_id TYPE TEXT USING structure_id::text,
    ALTER COLUMN progress TYPE DOUBLE PRECISION USING progress::double precision;

ALTER TABLE pipeline_job
    ADD COLUMN IF NOT EXISTS modules JSONB NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS error TEXT,
    ADD COLUMN IF NOT EXISTS started_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS completed_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT now();

UPDATE pipeline_job
SET started_at = COALESCE(started_at, created_at, now()),
    updated_at = COALESCE(updated_at, created_at, now())
WHERE started_at IS NULL OR updated_at IS NULL;

ALTER TABLE pipeline_job
    ALTER COLUMN started_at SET NOT NULL;

CREATE INDEX IF NOT EXISTS idx_pipeline_job_status
    ON pipeline_job(status)
    WHERE status IN ('queued', 'running');

CREATE INDEX IF NOT EXISTS idx_pipeline_job_structure
    ON pipeline_job(structure_id);
