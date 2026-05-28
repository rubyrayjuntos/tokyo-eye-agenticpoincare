-- ============================================================================
-- Migration 023: Governed Write Path Foundations (Notes + Scaffolding)
-- Date: 2026-05-27
-- Purpose: This migration serves as a placeholder and design note for the
--          future "Normalizer" / single authorized write path component.
--          No tables are created here yet — this documents the intended
--          direction for Phase 1/2.
-- ============================================================================

/*
GOVERNED WRITE PATH DESIGN NOTES (Phase 0.2)

Core Principle (from ARCHITECTURE.md):
- There should be a single, controlled, auditable path for writing into the
  governed data layer.

Intended Future Components (to be designed in Phase 1):
- A "Normalizer" service or module that is the only authorized writer.
- All compute services (dehydron detection, GNN inference, phase runners, etc.)
  will call the Normalizer instead of writing directly to Aurora.
- The Normalizer will:
    1. Validate inputs against schemas
    2. Enforce provenance requirements (every write must have a run_id)
    3. Register assets in the governed_asset catalog (see migration 022)
    4. Write to the correct fact tables
    5. Publish events to Pub/Sub after successful writes

This migration file is intentionally empty of DDL. It exists to mark the
intention in the migration history and to serve as a living design note
until the Normalizer design work begins in earnest.

When the Normalizer is implemented, this file may be updated or superseded
by concrete DDL and supporting objects.
*/

-- Placeholder comment only. No tables created in this migration.