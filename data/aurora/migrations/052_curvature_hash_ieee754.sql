-- ============================================================================
-- Migration 052: curvature_hash uses IEEE 754 float64 bytes (not repr)
-- Date: 2026-07-03
-- Purpose: SHA256(struct.pack('>d', curvature)) is stable across Python versions.
--          Replaces migration 051 repr/CAST population for CI + cross-env SSOT.
-- ============================================================================

COMMENT ON COLUMN embedding_space.curvature_hash IS
    'SHA256 hex of big-endian IEEE754 float64 bytes — curvature_loader.curvature_hash_for()';

-- lever_a pin: 0.7026273608207703 → 90383c360f60edddafebcb6c119b4b62366543eb733d7a9536bcb54d92c2e0e8
UPDATE embedding_space
SET curvature_hash = '90383c360f60edddafebcb6c119b4b62366543eb733d7a9536bcb54d92c2e0e8'
WHERE name = 'gospconemapper_v6_hyp128'
  AND curvature IS NOT NULL;
