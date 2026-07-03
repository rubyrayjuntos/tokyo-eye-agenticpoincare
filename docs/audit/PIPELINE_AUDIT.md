# Pipeline Runtime Audit

**Repository:** tokyo-eye-agenticpoincare  
**Date:** June 26, 2026  
**Status:** Implemented (migrations 049–050)  
**Related:** `docs/audit/WRITE_PATH_INVENTORY.md` (governed **write** audit), `science/contracts/onboard_contract.yaml` (geometric contract), `AGENTS.md`, [`DEVELOPER_ONBOARDING.md`](../DEVELOPER_ONBOARDING.md) (compliance guide)

---

## Purpose

The platform has two complementary audit layers:

| Layer | Table | What it records |
|-------|--------|-----------------|
| **Governed write audit** | `normalization_audit` | Every Normalizer call (success/failure, payload type, assets created) |
| **Pipeline runtime audit** | `audit_pipeline_events` | Compute orchestration: geometric validation, curvature passthrough, preconditions, pathway lifecycle, enforcement decisions |

`normalization_audit` answers *“did this scientific fact persist correctly?”*  
`audit_pipeline_events` answers *“what happened during compute over time, and why?”*

Runtime validation (contract checks, learned curvature, preconditions) previously appeared only in logs or single HTTP responses. Pipeline audit makes warnings and errors **queryable historical artifacts**.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Pipeline execution                                          │
│  runner_dispatch · preconditions · ingest · readiness        │
└───────────────────────────┬─────────────────────────────────┘
                            │ emit_audit_event()
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Audit event bus (shared/audit/)                             │
│  · Structured JSON logs (stdout / CloudWatch)                 │
│  · In-memory ring buffer (dev fallback, 500 events)          │
│  · Persistence sink → audit_pipeline_events                    │
└───────────────────────────┬─────────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
     GET /api/structures/{id}/audit   audit_pipeline_daily_summary
     GET /api/pipeline/audit          (after retention rollup)
```

**Code locations:**

| Module | Role |
|--------|------|
| `shared/audit/events.py` | `AuditEvent` model, event type constants |
| `shared/audit/bus.py` | `emit_audit_event()` — logging + persistence |
| `shared/audit/instrumentation.py` | Typed helpers per instrumentation point |
| `data/audit/pipeline_events.py` | INSERT + query helpers |
| `data/audit/retention.py` | Aggregate + prune older than retention window |
| `agent/coordinator/routers/audit.py` | Query API |

---

## Event model

```python
AuditEvent:
  event_id: str              # UUID
  timestamp: datetime
  event_type: str            # see table below
  severity: info | warning | error
  structure_id: str | None
  pipeline_job_id: str | None
  job_name: str | None
  contract_version: str      # from onboard_contract.yaml
  details: dict              # flexible payload
  correlation_id: str        # request_id or pipeline run_id
  enforcement_level: str     # snapshot of GEOMETRIC_ENFORCEMENT_* at emit time
```

### Event types

| `event_type` | Emitted from | Typical severity |
|--------------|--------------|------------------|
| `geometric_validation` | `runner_dispatch`, contract validation | warning |
| `curvature_learned` | `gnn_inference` success | info |
| `curvature_passthrough` | Downstream hyperbolic jobs | info / warning |
| `enforcement_decision` | Geometric enforcement apply | warning / error |
| `precondition_failed` | `preconditions.py` | error |
| `onboard_geometric_note` | Ingest prerequisites | info / warning |
| `geometric_readiness` | `data/readiness.py` (deduped) | info / warning |
| `pathway_started` | `execute_pathway` | info |
| `pathway_complete` | `execute_pathway` | info |
| `pathway_failed` | Dashboard background pipeline | error |
| `job_run_complete` | Per-job `runner_dispatch` | info / error |
| `contract_validation` | Startup / validation hooks | warning |

---

## Instrumentation points

| Location | Events |
|----------|--------|
| `science/compute/runner_dispatch.py` | geometric validation, curvature, enforcement, job completion |
| `science/compute/preconditions.py` | `precondition_failed` |
| `science/api/routers/ingest.py` | onboard geometric notes |
| `data/readiness.py` | `geometric_readiness` (fingerprint dedupe — no poll spam) |
| `science/compute/pathway_executor.py` | pathway started / complete |
| `agent/coordinator/routers/dashboard.py` | pathway failed |

### Correlation and `pipeline_job_id`

On ingest, the dashboard creates a `pipeline_job` row and runs compute in a background task. The job ID propagates:

```
_run_pipeline_background(job_id)
  → run_onboard_compute(pipeline_job_id=job_id)
  → execute_pathway(..., pipeline_job_id=...)
  → POST /compute/jobs/{id} { pipeline_job_id }
  → JobRunContext.pipeline_job_id
```

`audit_pathway_started` sets `RequestContext.run_id` to the pipeline job ID so subsequent events in the same pathway share a correlation ID.

### Hyperbolic geometry and learned curvature

Geometric semantics are declared in `science/contracts/onboard_contract.yaml` (v1.2+). Audit events capture runtime enforcement of that contract:

- **Curvature is learned** at `gnn_inference` and stored on `embedding_space.curvature` — never hardcoded in the contract.
- Downstream hyperbolic jobs emit `curvature_passthrough` (warning when missing).
- `enforcement_level` on each event snapshots `GEOMETRIC_ENFORCEMENT_LEVEL` / per-job overrides at emit time.

See `science/contracts/README.md` for the geometric mental model.

---

## Database schema

### `audit_pipeline_events` (migration 049)

Append-only. Indexed by `structure_id`, `pipeline_job_id`, `severity`, `event_type`, `correlation_id`.

### `audit_pipeline_daily_summary` (migration 050)

Daily rollups produced by retention. Preserves counts and trends after detailed rows are pruned.

---

## Configuration

| Variable | Default | Effect |
|----------|---------|--------|
| `AUDIT_PIPELINE_PERSIST` | `true` | Write warnings/errors to Postgres |
| `AUDIT_PIPELINE_MIN_SEVERITY` | `warning` | `info` events log only unless lowered |
| `AUDIT_RETENTION_DAYS` | `90` | Retention window for detail rows |
| `GEOMETRIC_ENFORCEMENT_LEVEL` | `warning` | `error` fails jobs on contract violation |
| `GEOMETRIC_ENFORCEMENT_ERROR_JOBS` | (empty) | Comma-separated jobs forced to `error` in CI/staging |

Apply migrations: `make migrate` (requires running DB).

---

## Query surfaces

### HTTP API (coordinator :8000)

| Endpoint | Description |
|----------|-------------|
| `GET /api/structures/{id}/audit` | Events for one structure (filters: `severity`, `event_type`, `since`, `limit`) |
| `GET /api/pipeline/audit` | Cross-structure query + optional `include_memory=true` |
| `GET /api/pipeline/audit/summary` | Daily aggregates from retention table |
| `POST /api/pipeline/audit/retention` | Run aggregate + prune (`retention_days`, `dry_run`) |

### CLI

```bash
make audit-structure STRUCTURE_ID=4obe SINCE=7d
make audit-summary SINCE=7d JOB_NAME=gnn_inference
make audit-retention RETENTION_DAYS=90
make audit-retention DRY_RUN=1    # preview prune count
```

### Python

```python
from data.audit.pipeline_events import get_audit_events_for_structure, query_audit_events
from data.audit.retention import run_audit_retention, query_daily_summary
```

---

## Frontend

**Discovery Cockpit → Control Console → Pipeline Audit**

When a structure is active, the panel shows:

- `geometric_readiness` from `GET /api/structures/{id}/readiness` (hyperbolic ready, learned κ, curvature ready)
- Last 12 events from `GET /api/structures/{id}/audit`

Implementation: `visualizer/frontend/src/components/cockpit/PipelineAuditReadout.tsx`

---

## Retention policy

1. Events older than `AUDIT_RETENTION_DAYS` (default 90) are rolled into `audit_pipeline_daily_summary` by `event_type`, `severity`, `job_name`, `structure_id`, and date.
2. Detailed rows in `audit_pipeline_events` are deleted after aggregation.
3. Run manually via `make audit-retention` or `POST /api/pipeline/audit/retention`. Schedule as cron in production.

---

## Relationship to other audit docs

- **`normalization_audit`** — governed write path only; written inside `data/normalizer/core.py`. Do not conflate with pipeline audit.
- **`docs/audit/WRITE_PATH_INVENTORY.md`** — inventory of SQL write bypasses; pipeline audit does not replace write-path enforcement.
- **`science/contracts/onboard_contract.yaml`** — declares geometric space and job metadata; pipeline audit records runtime drift and enforcement.

---

## Example queries

**All warnings for a structure (last 7 days):**

```bash
make audit-structure STRUCTURE_ID=4obe SINCE=7d SEVERITY=warning
```

**How often gnn_inference triggered enforcement:**

```bash
make audit-summary SINCE=30d JOB_NAME=gnn_inference
```

**Structures with repeated geometric warnings** — query `audit_pipeline_events` grouped by `structure_id` and `event_type`, or use daily summary after retention.
