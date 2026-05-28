-- ============================================================================
-- Migration 009: fact_atom_feature uniqueness constraint
--
-- Adds a unique constraint on (atom_id, run_id, feature_type) so that
-- write_gnn_atom_features() can use ON CONFLICT DO UPDATE and remain
-- idempotent across pipeline reruns.
--
-- Safe to apply to tables with existing rows — the constraint only
-- affects future inserts.  If duplicate rows already exist (rows written
-- before this migration), resolve them first:
--
--   DELETE FROM fact_atom_feature a
--   USING (
--       SELECT MIN(ctid) AS keep, atom_id, run_id, feature_type
--       FROM fact_atom_feature
--       GROUP BY atom_id, run_id, feature_type
--       HAVING COUNT(*) > 1
--   ) dup
--   WHERE a.atom_id = dup.atom_id
--     AND a.run_id  = dup.run_id
--     AND a.feature_type = dup.feature_type
--     AND a.ctid <> dup.keep;
-- ============================================================================

ALTER TABLE fact_atom_feature
    ADD CONSTRAINT IF NOT EXISTS uq_atom_feature_run_type
    UNIQUE (atom_id, run_id, feature_type);
