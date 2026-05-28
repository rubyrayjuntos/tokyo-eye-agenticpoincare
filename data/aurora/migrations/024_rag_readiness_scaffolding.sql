-- ============================================================================
-- Migration 024: RAG Readiness Scaffolding (Design Only)
-- Date: 2026-05-27
-- Purpose: Placeholder migration marking the intention to support Vector RAG
--          and hybrid retrieval workloads. No production tables created yet.
--          This aligns with the "design for, implement later" decision.
-- ============================================================================

/*
RAG READINESS SCAFFOLDING NOTES (Phase 0.2)

Per locked decisions:
- Full RAG implementation is deferred to Phase 5.
- The core data model must be shaped to support RAG cleanly from the beginning.

Intended Future RAG Assets (to be designed in Phase 5):
- Chunk assets (with links to source residue/site)
- Retrieval context / query log tables
- Possibly dedicated retrieval-optimized embedding spaces

Current Enablers Already in Place (as of Phase 0.2):
- Residue and site as stable, queryable primary keys (ADR-001)
- Native support for multiple embedding spaces (migration 013)
- Strong provenance on all assets
- Flexible extensibility patterns

This migration is a design marker only. When Phase 5 RAG work begins,
this file may be updated or superseded with actual DDL.
*/

-- Placeholder comment only. No tables created in this migration.