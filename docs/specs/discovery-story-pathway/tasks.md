# Tasks: Discovery Story Pathway

**Spec:** `requirements.md`, `design.md`  
**Status:** In progress — **16 peeled atomic jobs**; monolith = skip-flag shim; Act 01 fully atomic

### Implementation snapshot

| Layer | Path | Role |
|-------|------|------|
| Canonical catalog | `science/compute/registry.py` | 19 jobs, DAG validation |
| Scheduler | `science/compute/scheduler.py` | Monolith + peeled dispatch |
| Atomic runners | `science/compute/runners/` | 15 jobs via `POST /compute/jobs/{job_id}` |
| Monolith | `science/dtie/v5/orchestrator/pipeline.py` | GNN inference + skip flags |
| Spec (mirror) | `docs/specs/discovery-story-pathway/design.md` §1.1, §2.2 | Doc sync June 2026 |

---

## Phase 1 — Registry & documentation

- [x] Ratify `discovery-story-pathway` spec (review)
- [x] Link from `AGENTS.md` and `ingest-compute-contract`
- [x] Add `science/compute/registry.py` with `JOB_REGISTRY` from design §2.2
- [x] Unit test: DAG acyclic, all `requires` reference valid job IDs

## Phase 2 — Scheduler façade

- [x] Add `science/compute/scheduler.py` (topological sort + serial Phase A)
- [x] Update `science/dtie/ingest/orchestrator.py` to log `job_id` per module
- [x] Peel `source_leak_detection` into `science/compute/runners/source_leak_detection.py`
- [x] Add `POST /compute/jobs/{job_id}` + scheduler dispatch for peeled jobs
- [x] Provenance: `pathway`, `job_id`, `discovery_act` on new runs (parameters + events)
- [x] Peel `binding_site_scan` into `science/compute/runners/binding_site_scan.py`
- [x] Peel `md_validate_top_n` via `science/dtie/cryptic/smd_runner` (in-process)
- [x] Peel `pocket_pharmacophore_map` (pocket-scoped phase5 subset)
- [x] `fragment_screen` stub runner (provenance-only; hits deferred)
- [x] Peel `pharmacophore_identification` + `drug_candidate_ranking` (Act 04)
- [x] Peel Act 01 jobs: `graph_topology`, `witness_embedding`, `strain_vulnerability_scan`
- [x] Peel Act 02/05 jobs: `hyperbolic_motifs`, `topological_lift`, `resistance_pathway_map`, `allosteric_site_detection`
- [x] Extract hyperbolic motif discovery to `science/compute/motifs/discovery.py`
- [x] Peel `gnn_inference` (foundational — last monolith core)

## Phase 3 — Readiness act-scoping

- [x] Add `data/act_readiness.py` with `ACT_JOB_MAP` and `derive_act_status`
- [x] Extend `GET /api/structures/{id}/readiness` with `acts`, `current_act`, `pathway`
- [x] Artifact alias layer: `source_leaks` ↔ `dtie_core` in API responses
- [x] Tests: act-scoped readiness for acts 01–03

## Phase 4 — Naming cleanup

- [x] Deprecation notice in code comments where `DTIE` appears in user strings (agent prompts)
- [x] Rename readiness labels (`ARTIFACT_LABELS`) to discovery vocabulary (partial — aliases)
- [x] Update agent tool category strings (DTIE → discovery signals)
- [x] Sync `design.md` §2.2 job table + implementation status with `registry.py`
- [x] Sync `ingest-compute-contract/requirements.md` §6 (atomic dispatch live, tier-2 keys)
- [x] Update `AGENTS.md` architecture line for peeled-job model
- [x] Refactor standalone `POST /compute/graph-topology` to shared Normalizer job path

## Phase 5 — UI alignment

- [x] Port Discovery Story act rail to onboard workbench (`DiscoveryStoryActRail`, `StructureOnboard`)
- [x] Remove user-facing "DTIE pipeline" / "Phase 5–6" copy in cockpit panels
- [x] Onboarding progress: act rail grouped by `discovery_act` (readiness poll + act label)
- [x] Briefing panel: rename phases to act titles (Signal, Persistent Leak, …)
- [x] Workbench/cockpit polls readiness `current_act` during onboard (`StructureOnboard`, `useAutoIngestPipeline`)

## Phase 6 — Fragment & MD completion

- [x] Implement `fragment_screen` job stub (planned — provenance only)
- [x] Normalizer adapter for binding scan (`normalize_binding_site_scan`)

## Deferred

- [ ] Science package rename (`science/dtie/` → `science/discovery/` or similar)
- [ ] DB tables `structure_computation_run` / `structure_computation_stage`
- [ ] Secondary pathways `viewport_explore`, `pocket_only`
