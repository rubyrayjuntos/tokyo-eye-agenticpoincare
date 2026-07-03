# Developer Onboarding — Compliance Guide

**Repository:** tokyo-eye-agenticpoincare  
**Audience:** Engineers starting any new feature, job, API surface, or data write in this repo  
**Purpose:** Prevent spec drift by making the enacted platform contracts explicit and actionable

> **Living document:** Updated whenever new platform contracts or enforcement mechanisms are introduced.  
> **Last significant update:** June 26, 2026 — Master Onboard Contract v1.2, geometric enforcement, pipeline audit persistence (migrations 049–050), learned-curvature passthrough.

**If your change is not reflected in the onboard contract (or a linked spec you update in the same PR), it will drift and fail validation eventually.** Treat the contract as a living spec: re-read the relevant sections whenever you touch artifacts, jobs, readiness, geometry, or API shapes — not only on day one.

---

## Start here

Read in this order before your first PR. Total: **~35–45 minutes** for a full pass; skim the contract sections you will touch on every subsequent change.

| # | Document | Time | Why this order |
|---|----------|------|----------------|
| 1 | [`AGENTS.md`](../AGENTS.md) | ~8 min | Platform identity, architecture map, and non-negotiable rules — context for everything else |
| 2 | [`science/contracts/onboard_contract.yaml`](../science/contracts/onboard_contract.yaml) | ~15 min | **SSOT** for artifacts, jobs, readiness, geometry, API shapes — read `artifacts:`, `jobs:`, `geometric_constraints:` at minimum |
| 3 | [`docs/specs/ingest-compute-contract/requirements.md`](specs/ingest-compute-contract/requirements.md) | ~7 min | Why compute runs only at ingest and agent tools stay read-only |
| 4 | [`docs/audit/WRITE_PATH_INVENTORY.md`](audit/WRITE_PATH_INVENTORY.md) | ~5 min | Governed write path — no direct SQL to fact tables |
| 5 | [`docs/audit/PIPELINE_AUDIT.md`](audit/PIPELINE_AUDIT.md) | ~10 min | Runtime audit — how to record and query pipeline/geometric issues over time |
| 6 | [`docs/ENFORCEMENT_MATRIX.md`](ENFORCEMENT_MATRIX.md) | ~5 min | Which non-negotiable rules are **GATED** vs **MANUAL** today — read before assuming CI catches something |
| 7 | [`docs/AI_DEVELOPER_CONTRACT.md`](AI_DEVELOPER_CONTRACT.md) | ~5 min | **AI agents only** — STOP conditions, gate commands, anti-fabrication rules |

**After onboarding:** jump back to the contract YAML and this guide’s workflow section for your change type. Do not rely on memory.

> **Agents:** follow [`AI_DEVELOPER_CONTRACT.md`](AI_DEVELOPER_CONTRACT.md) instead of ticking checkboxes here. Humans use this doc; agents use the contract + paste gate output.

---

## When to update which spec doc

Update the matching spec **in the same PR** as your code change when the change is user-visible or normative.

| Your change | Also update |
|-------------|-------------|
| Ingest / DAG / agent compute boundary | `docs/specs/ingest-compute-contract/` |
| New science HTTP endpoint | `docs/specs/science-container-api/` |
| Cockpit UI panel | `docs/specs/discovery-cockpit-frontend/design.md` |
| New or bypassed write path | `docs/audit/WRITE_PATH_INVENTORY.md` + `scripts/audit_write_paths.py` (inventory: INSERT/UPDATE/COPY; `--gate` for CI) |
| New audit event types or query surfaces | `docs/audit/PIPELINE_AUDIT.md` |
| Major architecture decision | `docs/adr/` (new ADR) |
| Platform-wide rule | `AGENTS.md` + this document if the process changes |

---

## Platform mental model

```
User ingest (POST /api/ingest)
    → dims + scope (Normalizer)
    → pipeline_job created (dashboard)
    → discovery_story pathway (atomic jobs via science/compute)
    → governed facts (Normalizer only)
    → readiness + audit_pipeline_events
    → agent + UI consume pre-computed data (read-only tools)
```

| Authority | Location | Governs |
|-----------|----------|---------|
| **Write path** | `data/normalizer/core.py` | All inserts/upserts to fact tables |
| **Compute path** | `science/compute/` + ingest orchestrator | When and how jobs run |
| **Onboard contract** | `science/contracts/onboard_contract.yaml` | Artifact names, job metadata, geometry, readiness, API types |
| **Runtime audit** | `shared/audit/` + `audit_pipeline_events` | Queryable history of compute/orchestration issues |

---

## Non-negotiable rules (PR pre-flight)

> **Enforcement status:** each rule below is a requirement; whether CI fails on violation is documented in [`ENFORCEMENT_MATRIX.md`](ENFORCEMENT_MATRIX.md). Most data/provenance and agent-boundary rules are **MANUAL** today.

### Data & provenance

| Rule | Requirement |
|------|-------------|
| Write path | No direct `INSERT`/`UPDATE` on `fact_*`, `governed_asset`, embedding tables — Normalizer + payloads only. **Gated:** `tests/test_write_path_audit.py` |
| Provenance | `provenance_run` before persisting scientific outputs. **Gated:** `tests/test_provenance_gate.py` |
| IDs | `science/dtie/common/keys.py` for all canonical IDs |
| Idempotency | Natural keys + `ON CONFLICT`; re-ingest / re-pathway must be safe. **Gated:** ingest/graph/GNN idempotency property tests |
| Adapters | Science → adapter → Normalizer; no runner bypass. **Gated:** `tests/test_import_contracts.py` |

### Compute & agent boundary

| Rule | Requirement |
|------|-------------|
| When compute runs | Structure-scoped jobs **only** at ingest onboarding. **Gated:** `tests/test_import_contracts.py` |
| Agent tools | Read-only over governed artifacts (annotation/hypothesis writes via allowlisted Normalizer imports only). **Gated:** `tests/test_import_contracts.py` |
| New jobs | `registry.py` + `JOB_RUNNERS` + contract `jobs:` section. **Gated:** `validate_registry_job_keys_match_contract()` in `tests/test_onboard_contract.py` |
| Preconditions | `preconditions.py` — same for scheduler and `POST /compute/jobs/{id}`. **Gated:** `tests/test_compute_precondition_entrypoints.py` |
| Correlation | Pass `pipeline_job_id` through pathway dispatch |

### Geometry & hyperbolic space

| Rule | Requirement |
|------|-------------|
| Default space | Hyperbolic unless contract says `euclidean` or `mixed`. **Gated:** every artifact declares `geometric_space` in contract tests |
| Curvature | **Learned** at `gnn_inference` — never hardcode numeric κ. **Gated:** `tests/test_curvature_literal_lint.py` |
| Contract | Hyperbolic artifacts declare `curvature.source`; hyperbolic jobs in `hyperbolic_jobs` need `requires_hyperbolic: true` |
| Passthrough | Downstream jobs use `JobRunContext.learned_curvature` / `embedding_space.curvature` |
| Malformed coords | NaN/Inf/out-of-ball hyperbolic coordinates must **raise or quarantine** — never silently substitute origin `(0,0)` and report success. **Gated:** `tests/test_embedding_projection_gate.py` |
| MD validation | `fact_cryptic_site.md_validation_status='passed'` only after real SMD via `validate_site_md_in_process` → `run_smd` — never from dry-run, stubs, or synthetic success signals. **Gated:** `tests/test_md_validation_gate.py` |

### Contract & codegen

| Rule | Requirement |
|------|-------------|
| Edit order | `onboard_contract.yaml` **first**, then code |
| Codegen | `make contract-sync` after contract changes. **Gated:** `tests/test_contract_codegen_gates.py`, `scripts/lint_contract_sync.py` |
| Readiness | Tier/act lists from contract — not hardcoded in Python or UI. **Gated:** `tests/test_readiness_contract_sync.py` |
| Version | Bump contract `version` when semantics or shapes change. **Gated:** `scripts/lint_contract_version.py` |

### Audit & observability

| Rule | Requirement |
|------|-------------|
| Governed writes | `normalization_audit` (automatic via Normalizer) |
| Pipeline/runtime | Use `emit_audit_event()` or `shared/audit/instrumentation.py` helpers for anything you need to **query later** |
| Logging alone | OK for dev printf debugging; **not** sufficient for warnings/errors that operators must trace |
| Correlation | Set/pass `pipeline_job_id`; pathway start sets `RequestContext.run_id` |
| New events | Add stable `event_type` constant in `shared/audit/events.py`. **Gated:** `scripts/lint_audit_event_types.py` |

### Quality

| Rule | Requirement |
|------|-------------|
| Tests | `make test` before merge |
| Contract tests | Extend `tests/test_onboard_contract.py` when touching registry/geometry |
| Lint | `make lint` on Python; TypeScript compiles after `contract-sync` |

---

## Audit & observability (detailed)

Pipeline runs used to be visible only in ephemeral logs. **`audit_pipeline_events`** (migration 049) plus the event bus in `shared/audit/` make warnings, enforcement decisions, and curvature issues **historical and queryable**.

### Two audit layers (do not conflate)

| Layer | Table | Automatic? | Answers |
|-------|-------|------------|---------|
| **Governed write audit** | `normalization_audit` | Yes — every Normalizer call | Did this payload persist correctly? |
| **Pipeline runtime audit** | `audit_pipeline_events` | Emit from instrumentation | What happened during compute over time? Why did a job warn/fail? |

### Logging vs `emit_audit_event()`

| Situation | Use |
|-----------|-----|
| Transient dev debugging, verbose trace | `logger.info` / `logger.debug` |
| Precondition failure, geometric violation, enforcement decision, pathway failure | **`emit_audit_event()`** or instrumentation helper |
| Warning you want in **CI trends** or **structure history** | Audit event (`severity: warning` or `error`) |
| Info-only high-volume spam (e.g. every readiness poll) | Log only, or deduped audit (see `audit_geometric_readiness`) |

Persistence is controlled by `AUDIT_PIPELINE_PERSIST` (default `true`) and `AUDIT_PIPELINE_MIN_SEVERITY` (default `warning` — `info` events log but are not stored unless you lower the threshold).

### Event types and required context

All persisted events include: `event_type`, `severity`, `contract_version`, `correlation_id`, `enforcement_level` (snapshot at emit time), and optional `structure_id`, `pipeline_job_id`, `job_name`.

| `event_type` | Emit when | Typical `details` keys |
|--------------|-----------|-------------------------|
| `precondition_failed` | `check_job_preconditions` returns missing artifacts | `missing_artifacts`, `error` |
| `geometric_validation` | Contract geometry check messages | `message` |
| `curvature_learned` | `gnn_inference` succeeds with κ | `curvature` |
| `curvature_passthrough` | Downstream hyperbolic job loads κ | `learned_curvature` |
| `enforcement_decision` | `apply_enforcement()` after validation | `messages`, `failed_job` |
| `onboard_geometric_note` | Ingest prerequisite notes | `note` |
| `geometric_readiness` | Readiness assessment (deduped) | full `geometric_readiness` dict |
| `pathway_started` / `pathway_complete` / `pathway_failed` | Pathway lifecycle | `pathway_id`, `jobs_complete`, `error` |
| `job_run_complete` | After each job dispatch | `success`, `artifacts_produced`, `warning_count` |

Prefer helpers in `shared/audit/instrumentation.py` over raw `emit_audit_event()` when one exists.

```python
from shared.audit.instrumentation import (
    audit_precondition_failed,
    audit_geometric_messages,
    audit_enforcement_decision,
    audit_curvature_passthrough,
    audit_job_run_complete,
)
from shared.audit import emit_audit_event  # when no helper fits yet
```

### Querying pipeline history

| Method | Example |
|--------|---------|
| CLI | `make audit-structure STRUCTURE_ID=4obe SINCE=7d` |
| CLI summary | `make audit-summary SINCE=7d JOB_NAME=gnn_inference` |
| HTTP | `GET /api/structures/{id}/audit?severity=warning&limit=100` |
| HTTP (global) | `GET /api/pipeline/audit?job_name=source_leak_detection` |
| UI | Discovery Cockpit → **Control Console** → **Pipeline Audit** |
| Retention | `make audit-retention RETENTION_DAYS=90` → rolls up to `audit_pipeline_daily_summary` |

Full reference: [`docs/audit/PIPELINE_AUDIT.md`](audit/PIPELINE_AUDIT.md)

---

## Debugging geometric warnings & enforcement

Geometric contract checks run at job dispatch (`runner_dispatch` → `validate_job_run_result`). By default violations **warn** and append to `JobRunResult.warnings`; they are also audit-persisted when severity ≥ `AUDIT_PIPELINE_MIN_SEVERITY`.

### Environment controls

| Variable | Default | Effect |
|----------|---------|--------|
| `GEOMETRIC_ENFORCEMENT_LEVEL` | `warning` | `error` → failed job run on violation |
| `GEOMETRIC_ENFORCEMENT_ERROR_JOBS` | (empty) | Comma list — force `error` for specific jobs (use in CI/staging) |
| `AUDIT_PIPELINE_PERSIST` | `true` | Store audit events to Postgres |
| `AUDIT_PIPELINE_MIN_SEVERITY` | `warning` | What gets persisted |

Example CI/staging stricter profile (`.env`):

```bash
GEOMETRIC_ENFORCEMENT_LEVEL=warning
GEOMETRIC_ENFORCEMENT_ERROR_JOBS=gnn_inference,source_leak_detection
```

### Symptom → diagnosis → fix

| Symptom | Likely cause | What to do |
|---------|--------------|------------|
| `curvature is missing` on `gnn_inference` | Runner did not set `outputs.curvature` | Fix GNN runner; ensure `embedding_space.curvature` persisted |
| `without learned curvature passthrough` on downstream job | `gnn_hyp` exists but no κ in DB, or dispatch before GNN completes | Check preconditions; verify `load_structure_learned_curvature()` |
| `outputs.curvature != embedding_space learned curvature` | Runner hardcoded κ | Remove hardcode; use passthrough from context |
| `hyperbolic_jobs` / `requires_hyperbolic` mismatch | Contract drift | Align YAML `jobs:` with `geometric_constraints.hyperbolic_jobs` |
| `precondition_failed` + `learned_curvature` | GNN not finished or audit-only run | Re-ingest or wait for pathway; check `embedding_space` row |
| Job failed only in CI | `GEOMETRIC_ENFORCEMENT_ERROR_JOBS` | Fix root cause or adjust CI env intentionally |

### Debug workflow

1. **Reproduce** — run ingest or single job via science API with `pipeline_job_id` set when in pathway context.
2. **Query audit** — `make audit-structure STRUCTURE_ID=… SEVERITY=warning` or Control Console panel.
3. **Check readiness** — `GET /api/structures/{id}/readiness` → `geometric_readiness.learned_curvature`, `curvature_ready`.
4. **Check contract** — job and artifact `geometric_space` / `curvature.source` in `onboard_contract.yaml`.
5. **Run contract tests** — `pytest tests/test_onboard_contract.py -q`.

Hyperbolic quick reference:

| Question | Answer |
|----------|--------|
| Where declared? | `onboard_contract.yaml` → `geometric_constraints`, `artifacts:`, `jobs:` |
| Where κ comes from? | `gnn_inference` → `embedding_space.curvature` |
| Downstream κ? | `JobRunContext.learned_curvature` + `job_params["learned_curvature"]` |

Detail: [`science/contracts/README.md`](../science/contracts/README.md)

---

## Decision tree: what kind of change is this?

```
Are you persisting new scientific data?
  YES → Normalizer payload + adapter + contract artifact + migration if new table
  NO ↓

Are you adding/running structure-scoped computation?
  YES → Job registry + runner + contract jobs + preconditions + pathway DAG
        (trigger only from ingest orchestrator, not agent)
  NO ↓

Are you exposing new API or UI fields for readiness/ingest/hydrate?
  YES → onboard_contract api_surfaces + make contract-sync + router returns match
  NO ↓

Are you adding agent capabilities?
  YES → Read-only over artifacts? → tool in agent/tools/
        Schedules compute? → STOP — violates ingest–compute contract
  NO ↓

Operational / infra → still check logging, audit instrumentation, env docs
```

---

## Workflows by change type

### A. Add a new governed artifact

1. Add `artifacts:` entry in [`onboard_contract.yaml`](../science/contracts/onboard_contract.yaml) (`canonical_key`, `tier`, `discovery_act`, `geometric_space`, `representation`, `probe`, `producing_jobs`, `governed_tables`; hyperbolic → `curvature.source`).
2. Producing job lists artifact in `science/compute/registry.py` `produces`.
3. Wire probe via contract `probe` id in `data/readiness.py`.
4. Normalizer method + payload if new table shape.
5. `make contract-sync` if artifact appears in API responses.
6. `tests/test_onboard_contract.py`.

### B. Add a new atomic compute job

1. Register in [`science/compute/registry.py`](../science/compute/registry.py).
2. Runner in `science/compute/runners/` + [`runner_dispatch.py`](../science/compute/runner_dispatch.py) `JOB_RUNNERS`.
3. Contract `jobs:` block + `hyperbolic_jobs` if applicable.
4. Preconditions in [`preconditions.py`](../science/compute/preconditions.py) (include `learned_curvature` if post-GNN).
5. `make contract-sync`; validate with contract + precondition tests.
6. Consider audit helpers if new failure modes need historical trace.

**Do not** expose as an agent-callable compute tool.

### C. Change readiness, acts, or tiers

1. Edit contract `readiness:` / `acts:` only.
2. Confirm `data/readiness.py` / `data/act_readiness.py` use contract helpers.
3. `make test tests/test_readiness_contract_sync.py tests/test_act_readiness.py tests/test_structure_readiness.py`; `make contract-sync`.

### D. Change ingest or API response shapes

1. Contract `api_surfaces:` → `make contract-sync` → update routers.

### E. Frontend (Discovery Cockpit)

1. Generated types from `visualizer/frontend/src/lib/generated/onboard.ts`.
2. Readiness + audit from API — no hardcoded artifact lists.
3. Pipeline Audit panel already wired to `GET .../audit`.

---

## Local setup

```bash
cp .env.example .env
make up-db && make migrate    # migrations 049–050 = audit tables
uv sync --extra dev
make test
```

After contract edits: `make contract-sync` && `pytest tests/test_onboard_contract.py -q`

---

## PR submission checklist

```markdown
## Compliance checklist

- [ ] Onboard contract updated (if artifacts/jobs/readiness/API changed); version bumped
- [ ] `make contract-sync` run (if contract changed)
- [ ] No direct fact-table SQL writes outside Normalizer
- [ ] No new agent-scheduled compute paths
- [ ] Hyperbolic jobs/artifacts: geometry + curvature metadata (no hardcoded κ)
- [ ] Registry ↔ contract ↔ runners aligned (`test_onboard_contract` passes)
- [ ] Preconditions updated for new job dependencies
- [ ] Audit: queryable events for new warnings/failures (not log-only)
- [ ] `pipeline_job_id` passed if adding pathway-scoped dispatch
- [ ] `make test` and `make lint` pass
```

---

## Key files quick reference

| Concern | File(s) |
|---------|---------|
| Master contract | `science/contracts/onboard_contract.yaml` |
| Contract validation | `science/contracts/onboard_contract.py`, `science/contracts/validation.py`, `science/contracts/geometric_runtime.py` |
| Job dispatch | `science/compute/runner_dispatch.py`, `science/compute/dispatch_helpers.py` |
| Pathway | `science/compute/pathway_executor.py`, `science/dtie/ingest/orchestrator.py` |
| Preconditions | `science/compute/preconditions.py` |
| Normalizer | `data/normalizer/core.py` |
| Audit emit | `shared/audit/bus.py`, `shared/audit/instrumentation.py`, `shared/audit/events.py` |
| Audit persist/query | `data/audit/pipeline_events.py`, `data/audit/retention.py` |
| Audit API | `agent/coordinator/routers/audit.py` |
| Contract tests | `tests/test_onboard_contract.py`, `tests/test_readiness_contract_sync.py`, `tests/test_audit_pipeline.py` |

---

## Common mistakes (avoid drift)

| Mistake | Correct approach |
|---------|------------------|
| Direct `INSERT` into fact tables | Adapter → Normalizer |
| Agent schedules compute | Ingest re-run only |
| `curvature: 1.0` in contract or runner | Learned at inference; passthrough from `embedding_space` |
| Hardcoded tier lists in UI | `GET .../readiness` |
| Job in registry but not contract | Add `jobs:` + contract tests |
| `logger.warning` only for geometric issue | `emit_audit_event` / instrumentation helper |
| Forgot `make contract-sync` | Frontend/backend shape drift |
| `hyperbolic_jobs` out of sync | Each listed job: `requires_hyperbolic: true` |
| Debugging from logs alone | `make audit-structure` or `/api/structures/{id}/audit` |

---

## Getting help

| Problem | First step |
|---------|------------|
| Contract test failure | Read `tests/test_onboard_contract.py` output; fix YAML ↔ registry ↔ runners |
| Geometric warning | `make audit-structure STRUCTURE_ID=…` → see **Debugging geometric warnings** above |
| Write path uncertainty | Grep table name; see `docs/audit/WRITE_PATH_INVENTORY.md` |
| Missing pipeline history | Confirm migrations 049–050 applied; `AUDIT_PIPELINE_PERSIST=true` |

---

**This document is the onboarding entry point for compliance.** Update it when platform contracts or enforcement change. Deep references: `AGENTS.md`, `docs/audit/`, `docs/specs/`, `science/contracts/`.
