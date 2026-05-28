# Provenance Model Draft (v0.1)

**Date:** May 27, 2026  
**Status:** Early Draft  
**Related:** ADR-001, ADR-002, `data/ARCHITECTURE.md`

---

## 1. Goal

Every governed asset must carry strong, queryable, machine-readable provenance that allows full traceability of how the data was created, what inputs were used, which model/version produced it, and under what run/context.

This is non-negotiable per the core principles.

---

## 2. Core Provenance Spine

### Recommended Minimal Structure

A central `provenance` or `run` concept that every asset references:

- `run_id` (primary key, generated at the start of a logical computation or pipeline run)
- `structure_id` (when applicable)
- `model_version` (e.g., "GOSPConeMapper-v4", "DTIE-v3-full")
- `checkpoint_sha256` or `checkpoint_uri`
- `code_version` (git commit or equivalent)
- `pipeline_name` / `orchestrator`
- `source_type` (deterministic / probabilistic / external / derived)
- `started_at`, `completed_at`
- `parameters` (JSONB – e.g., n_landmarks, curvature override, effector sites)
- `warnings` (JSONB)
- `parent_run_id` (for hierarchical or sub-runs)

### Asset-Level Provenance Link

Every fact table / governed asset should include:
- `run_id` (strong link to the above)
- `source_type`
- `computed_at`
- Optional: `upstream_asset_ids` (array of other asset_ids that directly contributed)

This creates a traversable DAG of data lineage.

---

## 3. Granularity Considerations (Aligned with ADR-001)

- Provenance can (and should) be recorded at the most appropriate grain.
- For residue-level outputs (GNN node results, per-residue uncertainty, etc.), the `run_id` + `residue_id` combination gives very strong traceability.
- For higher-level constructs (`dim_site`), provenance can reference the contributing residues + the overall run.

---

## 4. v3 vs v4 Lineage Handling (Aligned with ADR-002)

- `model_version` is the primary differentiator.
- A single `run_id` can be associated with one lineage (v3 or v4), but downstream consumers can join across lineages when needed via shared `structure_id` / `residue_id`.
- Training runs should use a distinct `run_type` or be clearly marked so they don't pollute inference provenance graphs.

---

## 5. Concrete Implementation Start

A starting-point migration has been created:

- `data/aurora/migrations/003_provenance.sql`

It implements the core `provenance_run` table plus an optional `provenance_event` table for fine-grained steps. This directly supports the principles in `data/ARCHITECTURE.md` and the decisions in ADR-002 and ADR-003.

## 6. Open Questions

- How much detail to store in the core `run` record vs. in asset-specific JSONB?
- How to handle external data (e.g., RCSB fetches, CDD annotations) in the provenance model?
- Versioning strategy for the provenance schema itself.
- Best patterns for querying "full provenance of this residue-level embedding".

---

**This is an early draft (v0.1).** It will be refined as we move into Phase 1 schema work. The goal is to ensure provenance is designed in from the start rather than bolted on later.