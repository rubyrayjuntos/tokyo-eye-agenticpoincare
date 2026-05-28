# Migrated from: SRC_AGENT/pipeline-v3/gosp/ on 2026-05-27
"""GOSP Pipeline — services, normalizers, API routes, DB layer, and jobs.

This is the full agent pipeline infrastructure migrated from the ADK source.
It includes:
- API routes (v1 + v2): structures, analysis, annotations, assets, etc.
- Services: dehydron detection, energy calculation, ingestion, validation, etc.
- Normalizers: GNN writeback, dehydron, energy, folding, HDX, etc.
- Data layer: Aurora connection, models, repository
- Jobs: GNN inference, tier1/tier2 analysis

These will be progressively adapted to use the new governed data layer
(Normalizer, production views) rather than direct DB writes.
"""
