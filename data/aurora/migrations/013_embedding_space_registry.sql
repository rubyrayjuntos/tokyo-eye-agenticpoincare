-- ============================================================================
-- Migration 013: Embedding Space Registry (Production Grade)
-- Date: 2026-05-27
-- Purpose: First-class, governed registry for all embedding spaces used
--          in the platform (Euclidean, Hyperbolic, future types).
--          This enables clean multi-space support and RAG readiness.
-- ============================================================================

CREATE TABLE IF NOT EXISTS embedding_space (
    space_id            TEXT PRIMARY KEY,
    name                TEXT NOT NULL UNIQUE,
    space_type          TEXT NOT NULL CHECK (space_type IN ('euclidean', 'hyperbolic', 'other')),
    dimensionality      INTEGER NOT NULL,
    curvature           DOUBLE PRECISION,                    -- only for hyperbolic
    model_name          TEXT,                                -- e.g. "GOSPConeMapper-v4", "text-embedding-ada-002"
    model_version       TEXT,
    description         TEXT,
    is_active           BOOLEAN DEFAULT TRUE,
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_embedding_space_type ON embedding_space(space_type);
CREATE INDEX IF NOT EXISTS idx_embedding_space_active ON embedding_space(is_active);

-- Optional: Version history for embedding spaces if models evolve
CREATE TABLE IF NOT EXISTS embedding_space_version (
    version_id          TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    space_id            TEXT NOT NULL REFERENCES embedding_space(space_id) ON DELETE CASCADE,
    model_version       TEXT NOT NULL,
    changelog           TEXT,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- NOTES
-- ============================================================================
-- - This registry is referenced by all embedding fact tables.
-- - Supports the locked requirement for native multi-space (Euclidean + Hyperbolic) support.
-- - Designed to be extensible for future embedding models and RAG use cases.