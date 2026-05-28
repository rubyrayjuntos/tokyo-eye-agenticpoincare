# Data Population and Backfill Strategy (v0.8)

**Status:** Phase 0.2 Complete – Design sufficient for transition to Phase 1
**Date:** May 27, 2026
**Related:** ARCHITECTURE.md, Roadmap Phase 1, Migration 022 (governed_asset)

## 1. Guiding Principles

- Every piece of historical or newly computed data that qualifies as a "governed asset" must eventually exist in the governed layer with proper provenance.
- Backfills must not corrupt or duplicate existing governed data.
- The process must be auditable and restartable.

## 2. Categories of Data Population

### A. New Runs (Forward Path)
- All future computation goes through the Normalizer (see NORMALIZER_DESIGN.md).
- This is the clean path.

### B. Historical Backfill (Existing Data)
Sources:
- Existing Aurora data from previous systems (ADK repo schemas, tokyo-eye-data, etc.)
- Object storage artifacts (old graphs, checkpoints, trajectories)
- External data (CDD annotations, HDX uploads, etc.)

### C. Science Code Integration Backfill
- When v3 and v4 science code is integrated (Phase 3), historical runs from those codebases will need to be re-executed or imported with proper provenance.

## 3. Recommended Backfill Approach (Phased)

### Phase 1 (Core Dimensions + High-Value Facts)
- Backfill `dim_structure`, `dim_chain`, `dim_residue`, `dim_atom` from existing PDB ingestion data.
- Backfill highest-value fact tables: `fact_gnn_node_embedding`, `fact_dehydron`, `fact_phase3_persistence`, etc.
- Use the new `governed_asset` catalog (migration 022) as the source of truth during backfill.

### Phase 2 (Full Provenance & Lineage)
- Ensure every backfilled asset has a proper `provenance_run` record (even if synthetic for historical data).
- Build lineage where possible from existing logs/checkpoint metadata.

### Phase 3+ (Full Science Code Re-execution where needed)
- For critical historical results, re-run through the new v3/v4 integrated pipelines so they emit governed data natively.

## 4. Technical Patterns

- **Idempotent Upserts**: All backfill scripts must be idempotent on `run_id` + natural keys.
- **Synthetic Runs**: For truly historical data where no original run record exists, create "synthetic" provenance_run records clearly marked as `run_type = 'historical_backfill'`.
- **Audit Trail**: Every backfill job should produce a report (rows processed, rows skipped, warnings) linked to a backfill run_id.
- **Chunking**: Large backfills should be chunked by structure or by date range for restartability.

## 5. Tooling Recommendations (Phase 1)

- A `backfill/` module or set of scripts under `data/`.
- CLI or notebook-friendly entrypoints (e.g., `backfill --structures 4OBE,7XKJ --tables fact_dehydron,fact_gnn_node_embedding`).
- Dry-run mode that only reports what would be written.

## 6. Risks & Mitigations

- Risk: Loss of original provenance during backfill → Mitigation: Store original metadata in JSONB on the synthetic run or asset.
- Risk: Inconsistent data between old and new systems during transition → Mitigation: Clear "source of truth" cutover dates per table/type.
- Risk: Performance impact on production queries during large backfills → Mitigation: Use separate backfill database/user or run during low-traffic windows; use the new materialized views strategically.

## 7. Current Status (Phase 0.2 Close)

- Strategy and principles defined.
- `governed_asset` catalog table exists as the future registration point.
- Starter provenance_run table exists.
- No production backfill tooling or large-scale import scripts written yet (this is expected in Phase 1).

---

This document is considered sufficient to mark item (f) as designed to a Phase 0 complete level. Detailed implementation and tooling will be built in Phase 1.