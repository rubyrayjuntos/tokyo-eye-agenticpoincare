# Enforcement Matrix — Non-Negotiable Rules

**Repository:** tokyo-eye-agenticpoincare
**Companion to:** `DEVELOPER_ONBOARDING.md` (replaces the prose checklist in *Non-negotiable rules* with an enforced/manual split)
**Purpose:** Make the gap between *stated* contract and *enforced* contract visible. A rule with no executable gate is a suggestion.
**Last updated:** June 29, 2026

---

## How to read this

Each rule carries a **Status** and an **Enforced by / Backlog** entry:

- **GATED** — a named test/script/compile-step fails the build when the rule is violated. *Trust, but verify the assertion actually checks the claim (see §0).*
- **PARTIAL** — a check exists but covers only part of the rule, or runs manually instead of in CI.
- **MANUAL** — nothing fails on violation today. The reader (human or agent) must remember. **These are the drift surface.**

The **Backlog** column names the specific check that would move a `MANUAL`/`PARTIAL` row to `GATED`. Every one is intended to be a property test, an import-contract, an AST/grep lint, or a CI diff gate — not a longer checklist.

---

## §0 — Precondition: confirm the GATED rows are real

> Before trusting this matrix, run each named test once and read its assertions. A test named `test_onboard_contract.py` that loads the YAML but never asserts `registry ⊆ contract` is `MANUAL` wearing a `GATED` badge. This audit is one afternoon and it is the highest-value item in the table, because false confidence is more dangerous than acknowledged absence.

| Claimed gate | Confirm it actually asserts |
|---|---|
| `test_onboard_contract.py` | registry keys ⊆ contract `jobs:`; every `produces` artifact exists in `artifacts:`; geometric consistency |
| `validate_runners_against_registry()` | every registry job has a `JOB_RUNNERS` entry **and** vice-versa (bidirectional) |
| `test_compute_preconditions.py` | scheduler path and `POST /compute/jobs/{id}` resolve preconditions through the *same* function |
| `test_act_readiness.py` / `test_structure_readiness.py` | tier/act artifact lists are read from contract, not literals in the module |
| `scripts/audit_write_paths.py` | flags `INSERT/UPDATE/UPSERT/COPY` against governed tables anywhere outside `data/normalizer/` |

### §0 staleness

The §0 audit has two layers:

| Layer | Enforced by | Catches |
|-------|-------------|---------|
| **Existence** | `audit_gate_claims()` in `data/audit/gate_claims.py` | Renamed/deleted gate files or assertion *symbols* |
| **Liveness** | `audit_gate_liveness_canaries()` in `data/audit/gate_canaries.py` | Symbols present but gate hollow (known-bad input no longer fails) |

**Existence ≠ liveness.** A symbol-presence check cannot detect `assert x or True`, unreachable assertions, or setup that trivializes the check — the precise adversary §0 prose describes. Liveness canaries cover a small fixed set of high-value rows; full mutation testing is out of scope for CI today.

**Three-way sync (intermediate state).** Machine authority: `GATE_CLAIMS` + gate tests. Human prose: matrix table. When they disagree, trust `GATE_CLAIMS`. End state: matrix shrinks to backlog-only.

CI: `tests/test_enforcement_matrix_gate_claims.py` + `scripts/lint_gate_claims.py` (both layers).

### §0 audit results (June 26, 2026 — codebase review; meta-gate June 29, 2026)

| Claimed gate | Verdict | Notes |
|---|---|---|
| `test_onboard_contract.py` | **GATED (confirmed)** | Registry `produces` ⊆ contract catalog; `validate_registry_job_keys_match_contract()`; geometric consistency; `test_all_artifacts_declare_geometric_space`. |
| `validate_runners_against_registry()` | **GATED** | One-way runner↔registry plus `validate_registry_job_keys_match_contract()` — registry keys (minus external foundation jobs) == contract `jobs:`. |
| `test_compute_preconditions.py` | **GATED** | Tests `check_job_preconditions()` in isolation. Entrypoint parity gated by `test_compute_precondition_entrypoints.py`. |
| `test_act_readiness.py` / `test_structure_readiness.py` | **GATED** | `tests/test_readiness_contract_sync.py` — tier/act/foundation/probe constants match contract reload. |
| `scripts/audit_write_paths.py` | **GATED (inventory)** | Scans `INSERT`/`UPDATE`/`COPY` on governed tables; `--gate` runs the same CI check as `scripts/lint_write_paths.py`. CI gate: `tests/test_write_path_audit.py`. |

---

## Data & provenance

| Rule | Status | Enforced by / Backlog |
|---|---|---|
| No direct writes to `fact_*`, `governed_asset`, embedding tables | **GATED** | `tests/test_write_path_audit.py` + `scripts/lint_write_paths.py` — zero bypass in `agent/tools/`, `agent/coordinator/`; runners limited to `provenance_run`; legacy bypasses frozen in `data/audit/write_paths.py`. |
| Provenance first — `provenance_run` before persisting | **GATED** | `tests/test_provenance_gate.py` — `_ensure_provenance_run()` before writes; `_assert_provenance_run_active()`; ingest dimensions ordering fixed. |
| Canonical IDs via `keys.py` | **MANUAL** | **Backlog:** (a) property test round-tripping `make_structure_id`/`make_residue_id`; (b) grep/AST lint flagging f-string ID construction outside `keys.py`; (c) optionally `NewType` IDs so raw `str` won't typecheck at write boundaries. |
| Idempotent upserts (natural keys + `ON CONFLICT`) | **GATED** | `tests/test_normalizer_ingest_properties.py` (ingest), `tests/test_graph_topology_normalizer.py` (graph), `tests/test_normalizer_integration.py` (GNN); registry in `tests/test_write_path_audit.py`. |
| Adapters only — runners never import `data.normalizer/core` | **GATED** | `tests/test_import_contracts.py` + `scripts/lint_import_contracts.py` — `science/compute/runners/**` may not import `data.normalizer`. Runners persist via `science.compute.persist`. |

## Compute & agent boundary

| Rule | Status | Enforced by / Backlog |
|---|---|---|
| Structure-scoped compute runs at ingest only — not from agent/dashboard | **GATED** | `tests/test_import_contracts.py` — `agent/tools/**` may not import `runner_dispatch`, `pathway_executor`, `dispatch_helpers`, or `scheduler`. Legacy cryptic on-demand persist paths frozen in allowlist. |
| Agent tools read-only over governed artifacts | **GATED** | Same import-contract — Normalizer imports limited to annotation/hypothesis allowlist (`data_tools`, `hypothesis/tools`, `hypothesis/contradiction`). |
| New jobs in registry + `JOB_RUNNERS` + contract `jobs:` | **GATED** | `validate_runners_against_registry()` + `validate_registry_job_keys_match_contract()` in `tests/test_onboard_contract.py`. |
| Preconditions in `preconditions.py`, same for scheduler + endpoint | **GATED** | `tests/test_compute_preconditions.py` + `tests/test_compute_precondition_entrypoints.py` — parametrized parity across `dispatch_job_in_process` and `dispatch_compute_job`; `PRECONDITION_RESOLVER` documents canonical function. |
| Pass `pipeline_job_id` through dispatch | **MANUAL** | **Backlog:** assert audit rows for pathway-scoped jobs carry non-null `pipeline_job_id`/`correlation_id`. Covered together with audit §. |

## Geometry & hyperbolic space

| Rule | Status | Enforced by / Backlog |
|---|---|---|
| Default space hyperbolic unless contract marks `euclidean`/`mixed` | **GATED** | `validate_geometric_contract()` + `test_all_artifacts_declare_geometric_space` — every artifact declares `geometric_space` (no silent default). |
| **Never hardcode numeric curvature** | **GATED** | `tests/test_curvature_literal_lint.py` + `scripts/lint_curvature_literals.py` — bans `curvature or 1.0`, `.get("curvature", 1.0)`, `?? 1.0`, etc. in `agent/`, `science/` (v5+), `data/`, `visualizer/`. Helpers: `science/dtie/common/curvature_values.py`. |
| Hyperbolic artifacts declare `curvature.source` | **GATED** | `validate_geometric_contract()` via `test_geometric_contract_valid` — §0 confirmed. |
| `hyperbolic_jobs` ⇒ `requires_hyperbolic: true` + `geometric_space: hyperbolic` | **GATED** | `test_hyperbolic_jobs_authoritative` — §0 confirmed. |
| Malformed hyperbolic coords (NaN/Inf/out-of-ball) **not silently coerced to origin** | **GATED** | `tests/test_embedding_projection_gate.py` — `parse_embedding_projection_xy()` raises; hydration quarantines bad rows. Module: `science/dtie/common/embedding_projection.py`. |

## MD validation gate

| Rule | Status | Enforced by / Backlog |
|---|---|---|
| `md_validation_status='passed'` only after real SMD, never synthetic/dry-run | **GATED** | `tests/test_md_validation_gate.py` — `smd_result_qualifies_for_passed()` rejects stub/bare-success payloads; `validate_site_md_in_process` integration; dry-run runner never emits `md_validation` artifact. Live path: `compute.py` → `md_validate_top_n` → `validate_site_md_in_process` → `run_smd`. |

### Coordinate coercion gate

| Rule | Status | Enforced by / Backlog |
|---|---|---|
| Malformed hyperbolic coords not silently coerced to origin | **GATED** | `tests/test_embedding_projection_gate.py` — `science/dtie/common/embedding_projection.py`; dashboard hydration quarantines via `projection_quarantined`. |

## Contract & codegen

| Rule | Status | Enforced by / Backlog |
|---|---|---|
| Edit `onboard_contract.yaml` first | **MANUAL (process)** | Not directly gateable, but the *effect* of forgetting (drift) is caught by downstream contract tests. Leave as process. |
| Run `make contract-sync` after contract changes | **GATED** | `tests/test_contract_codegen_gates.py` + `scripts/lint_contract_sync.py` — regenerated `job_schema.json` and `onboard.ts` must match contract. |
| Readiness loads from contract — no hardcoded tier/act lists | **GATED** | `tests/test_readiness_contract_sync.py` — `TIER1_ARTIFACTS`, act order, foundation, and `ARTIFACT_PROBE_KEYS` track contract (`get_artifact_probe_keys()`). |
| Bump contract `version` on shape/semantic change | **GATED** | `scripts/lint_contract_version.py` — if `onboard_contract.yaml` differs from base ref and `version:` line unchanged, fail. |

## Audit & observability

| Rule | Status | Enforced by / Backlog |
|---|---|---|
| Governed writes → `normalization_audit` (automatic) | **GATED (by construction)** | Automatic in Normalizer. **Backlog:** property test confirming every governed write emits exactly one audit row (guards against a future write path that skips it). |
| Compute/runtime issues → `emit_audit_event`, not logs | **MANUAL** | **Backlog (low):** lint warning on `logger.warning/error` inside `science/compute/**` for conditions that should be queryable events. Hard to fully gate; treat as advisory lint. |
| Instrumentation uses event types from `events.py` | **GATED** | `tests/test_contract_codegen_gates.py` + `scripts/lint_audit_event_types.py` — no string literals for `emit_audit_event` / `AuditEvent.event_type` outside `events.py`. |
| Preserve `correlation_id` / `pipeline_job_id` on pathway work | **MANUAL** | **Backlog:** same test as dispatch propagation above. |

## Quality

| Rule | Status | Enforced by / Backlog |
|---|---|---|
| `make test` passes | **GATED** | CI. |
| Extend `test_onboard_contract.py` for alignment | **GATED (self-enforcing)** | The test is the gate. |
| `make lint`; frontend types compile | **GATED** | CI / TS compile. |

---

## Prioritized backlog (extracted)

Ordered by *integrity/defensibility risk*, not effort. Each is one property test, one import-contract, or one CI gate.

**P0 — integrity (can report false science):**
1. ~~MD gate property~~ — **done** (`tests/test_md_validation_gate.py`)
2. ~~Malformed-coord property~~ — **done** (`tests/test_embedding_projection_gate.py`)
3. ~~Curvature literal lint~~ — **done** (`tests/test_curvature_literal_lint.py`, `scripts/lint_curvature_literals.py`)

**P1 — defensibility (audit-trail holes):**
4. ~~Provenance-required~~ — **done** (`tests/test_provenance_gate.py`)
5. ~~Idempotency property~~ — **done** (ingest/graph/GNN property tests; registry in `tests/test_write_path_audit.py`)
6. ~~Write-path audit in CI~~ — **done** (`tests/test_write_path_audit.py`, `scripts/lint_write_paths.py`)

**P2 — boundary integrity (drift over time):**
7. ~~Import-contract: `agent/tools/**` ⊥ compute dispatch + Normalizer writes~~ — **done** (`tests/test_import_contracts.py`, `scripts/lint_import_contracts.py`)
8. ~~Import-contract: `science/compute/runners/**` ⊥ `data.normalizer.core`~~ — **done** (same gate)
9. ~~Preconditions: prove scheduler + endpoint share one resolver~~ — **done** (`tests/test_compute_precondition_entrypoints.py`)

**P3 — codegen/version hygiene:**
10. ~~`contract-sync` diff gate in CI~~ — **done** (`tests/test_contract_codegen_gates.py`, `scripts/lint_contract_sync.py`)
11. ~~Contract `version` bump gate~~ — **done** (`scripts/lint_contract_version.py`)
12. ~~Audit `event_type` constant-only lint~~ — **done** (`scripts/lint_audit_event_types.py`)

**P-confirm — do first, costs an afternoon:**
0. ~~§0 audit~~ — **done** (see §0 audit results above). Re-run when new GATED rows are added.

**P4 — partial clearance (June 26, 2026):**
13. ~~Registry ↔ contract `jobs:` keys~~ — **done** (`validate_registry_job_keys_match_contract`, contract v1.4)
14. ~~Readiness constant drift~~ — **done** (`tests/test_readiness_contract_sync.py`, contract-derived `ARTIFACT_PROBE_KEYS`)
15. ~~Write-path inventory UPDATE/COPY~~ — **done** (`data/audit/write_paths.py`, `scripts/audit_write_paths.py --gate`)
16. ~~Artifact `geometric_space` required~~ — **done** (`test_all_artifacts_declare_geometric_space`)

---

## P0–P3 gate index (complete)

| Priority | Theme | Gates |
|---|---|---|
| **P0** | Integrity (false science) | `tests/test_md_validation_gate.py`, `tests/test_embedding_projection_gate.py`, `tests/test_curvature_literal_lint.py` |
| **P1** | Defensibility (audit holes) | `tests/test_provenance_gate.py`, idempotency property tests, `tests/test_write_path_audit.py` |
| **P2** | Boundary integrity | `tests/test_import_contracts.py`, `tests/test_compute_precondition_entrypoints.py` |
| **P3** | Codegen/version hygiene | `tests/test_contract_codegen_gates.py`, `scripts/lint_contract_sync.py`, `scripts/lint_contract_version.py`, `scripts/lint_audit_event_types.py` |
| **P4** | Partial-row clearance | `tests/test_readiness_contract_sync.py`, registry↔contract job keys, extended write-path inventory |
| **P-meta** | §0 claim freshness | `tests/test_enforcement_matrix_gate_claims.py`, `scripts/lint_gate_claims.py` |
| **P5** | Readiness probe integrity | `tests/test_readiness_probe_gate.py` |
| **P6** | Science container integration | `tests/test_science_container_integration.py`, `tests/test_pipeline_job_id_gate.py` |

**Remaining MANUAL rows:** canonical IDs via `keys.py`, normalization_audit property test, advisory compute→audit lint.

**P6 — science container integration (June 29, 2026):**
18. ~~Real DBAdapter ingest → gnn_inference preconditions~~ — **done** (`tests/test_science_container_integration.py`, `tests/helpers/science_container_fixtures.py`; `@pytest.mark.integration`).
19. ~~`pipeline_job_id` propagation through dispatch/audit~~ — **done** (`tests/test_pipeline_job_id_gate.py`; explicit `pipeline_job_id` on `check_job_preconditions`).

**P5 — readiness probe integrity (June 29, 2026):**
17. ~~Probe infra error ≠ artifact missing~~ — **done** (`tests/test_readiness_probe_gate.py`, `ProbeInfrastructureError`, `probe_errors` on `ReadinessResponse`; preconditions report probe errors separately from `missing_*`).

This matrix is a drift surface (it restates rules that live in the contract and the onboarding doc). **Preferred end state:** each row's authority migrates to the **test or lint** that enforces it; onboarding links to those tests by name. This file can then shrink to backlog-only or retire.

**Until remaining MANUAL rows clear**, keep this matrix as the honest map of what CI actually enforces vs what we only remember in review. Deletion is a milestone, not a deadline — a living reference that shrinks as gates land is fine.
