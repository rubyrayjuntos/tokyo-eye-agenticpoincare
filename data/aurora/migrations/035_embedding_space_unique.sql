-- =============================================================================
-- Migration 035: Add unique constraint on embedding_space.name
-- Date: 2026-05-28
-- Purpose: The Normalizer uses ON CONFLICT (name) DO NOTHING which requires
--          a unique constraint on the name column.
-- =============================================================================

CREATE UNIQUE INDEX IF NOT EXISTS idx_embedding_space_name_unique
    ON embedding_space (name);
