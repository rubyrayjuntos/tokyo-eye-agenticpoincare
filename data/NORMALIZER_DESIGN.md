# Normalizer / Governed Write-Path Design (v0.9)

**Status:** Phase 0.2 Complete – Ready for Phase 1 detailed design and implementation
**Date:** May 27, 2026
**Related:** ARCHITECTURE.md §6, Migration 023, ADR-003

## 1. Purpose

Establish a single, controlled, auditable, and schema-enforcing path for writing data into the governed layer. This is a core governance requirement.

## 2. High-Level Architecture

```
Compute Services (dehydron, GNN inference, phase runners, etc.)
          │
          ▼  (call Normalizer, never write directly)
Normalizer Service / Module
          │
          ├── Validate against JSON Schemas
          ├── Enforce provenance (must have valid run_id)
          ├── Register asset in governed_asset catalog
          ├── Write to Aurora (dimensions + fact tables)
          └── Publish events (Pub/Sub / internal)
```

## 3. Core Responsibilities of the Normalizer

- **Authentication & Authorization**: Only trusted compute services may call it.
- **Schema Validation**: All incoming payloads validated against registered JSON Schemas.
- **Provenance Enforcement**: Every write must reference a valid `run_id` from `provenance_run`.
- **Asset Registration**: Every new governed asset is recorded in `governed_asset` (migration 022).
- **Idempotency / Upsert Semantics**: Clear rules for re-processing the same run.
- **Event Emission**: After successful write, emit structured events for downstream consumers (viewport, agent, future RAG indexers).

## 4. Interface Sketch (Python example – to be refined in Phase 1)

```python
from typing import Dict, Any

class Normalizer:
    def normalize_gnn_output(self, run_id: str, payload: Dict[str, Any]) -> str:
        """Accepts GNN node output for a structure and writes governed facts."""
        ...

    def normalize_phase_output(self, run_id: str, phase: str, payload: Dict[str, Any]) -> str:
        ...
```

## 5. Error Handling & Audit

- All validation failures must be logged with full context and linked to the `run_id`.
- Failed normalizations should not leave partial data in the governed layer.
- A `normalization_audit` table records every Normalizer attempt (success, validation error, write error).

### 5.1 Pipeline runtime audit (separate concern)

**Governed write audit** (`normalization_audit`) and **pipeline runtime audit** (`audit_pipeline_events`) serve different purposes:

| Table | Scope |
|-------|--------|
| `normalization_audit` | Normalizer only — did this payload persist? |
| `audit_pipeline_events` | Compute orchestration — preconditions, geometric enforcement, curvature, pathway lifecycle |

Pipeline audit is **not** a second write path for scientific facts. It is operational telemetry emitted from `shared/audit/` during job dispatch, ingest, and readiness checks. See `docs/audit/PIPELINE_AUDIT.md`.

## 6. Phasing

- **Phase 1**: Detailed interface design + initial implementation for the highest-volume paths (GNN output, key DTIE phases).
- **Phase 2**: Full rollout across all compute services + monitoring.
- **Ongoing**: Schema evolution support (new asset types must be registered before the Normalizer will accept them).

## 7. Open Questions (to resolve in Phase 1)

- Exact technology (FastAPI service vs in-process library vs both).
- Batching strategy for high-volume writes.
- How training runs (ADR-003) are handled differently from inference runs.
- Rate limiting and back-pressure mechanisms.

---

This document is considered sufficient to mark item (c) as "designed to Phase 0 completion level." Detailed implementation will occur in Phase 1/2.