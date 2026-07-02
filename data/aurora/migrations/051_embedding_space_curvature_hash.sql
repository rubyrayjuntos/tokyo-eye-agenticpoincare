-- ============================================================================
-- Migration 051: curvature_hash for SSOT verification (Phase 1a)
-- Date: 2026-07-02
-- Purpose: Store SHA256(repr(c)) for hyperbolic embedding_space rows so
--          curvature_loader can fail-loud on drift. Non-breaking column add.
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS pgcrypto;

ALTER TABLE embedding_space
    ADD COLUMN IF NOT EXISTS curvature_hash VARCHAR(64);

COMMENT ON COLUMN embedding_space.curvature_hash IS
    'SHA256 hex of Python repr(curvature) — must match science.dtie.common.curvature_loader.curvature_hash_for()';

UPDATE embedding_space
SET curvature_hash = encode(digest(CAST(curvature AS TEXT), 'sha256'), 'hex')
WHERE name = 'gospconemapper_v6_hyp128'
  AND curvature IS NOT NULL
  AND curvature_hash IS NULL;
