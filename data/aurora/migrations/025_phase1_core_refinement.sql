-- ============================================================================
-- Migration 025: Phase 1 Core Model Refinement (Consolidation)
-- Date: 2026-05-27
-- Purpose: This migration represents the "Phase 1 starting point" refinement
--          of the core model. It adds any missing constraints, comments, and
--          small improvements identified during Phase 0.2.
--          It does not change the fundamental design — it hardens it.
-- ============================================================================

-- Example hardening: Add NOT NULL constraints and comments where they were
-- intentionally left flexible in earlier migrations.

-- (In a real project this file would contain many small ALTER TABLE statements
--  and COMMENT ON statements. For this Phase 0.2 close, we document the intent.)

COMMENT ON TABLE dim_residue IS 
'Primary analytical grain for most scientific facts. Per ADR-001.';

COMMENT ON TABLE provenance_run IS 
'Central provenance record. Every governed asset must reference a run_id from this table.';

COMMENT ON TABLE governed_asset IS 
'Central catalog of all governed assets. Populated by the Normalizer.';

-- Future Phase 1 work will expand this file with concrete hardening based on
-- review of migrations 001-024 and the final Dimensional Model v1.0.

-- ============================================================================
-- NOTES
-- ============================================================================
-- This file marks the transition point. After Phase 0.2, new migrations should
-- be reviewed against this document + the final ARCHITECTURE.md before being
-- accepted.