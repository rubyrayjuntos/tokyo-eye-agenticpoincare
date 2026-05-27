# MIGRATION MAP — Tokyo Eye Agentic Poincaré Consolidation

**Purpose:** Exact source → destination guidance for the hybrid repo.  
**Date:** May 2026  
**Status:** Initial version — will be updated during execution.

---

## Source Locations (Reference)

- **SRC_AGENT** = `/media/rswan/89fe6e19-ce87-46fd-9c56-aa09a5245e57/adk-samples/python/agents/data-science`
- **SRC_DEM**   = `/home/rswan/Documents/Tokyo-eye-demensional-investigator`
- **SRC_VIZ**   = `/home/rswan/Documents/tokyo-eyes-visualizer`

---

## High-Priority Migrations (Phase 1 Focus)

### Science / DTIE Core

| Category | Source Path | Destination | Priority | Notes / Conflicts |
|----------|-------------|-------------|----------|-------------------|
| **v3 Orchestrator** | `SRC_DEM/DTIE_GNN_ORCHESTRATION/dtie_pipeline.py` + supporting utils | `science/dtie/v3/orchestrator/` | Critical | Best full pipeline. Rename functions for clarity (e.g. `run_dtie_full_pipeline_v3`) |
| **v3 GNN** | `SRC_DEM/DTIE_GNN_ORCHESTRATION/Gnnv3.py` | `science/dtie/v3/gnn/` | High | `GOSPConeMapper` |
| **v3 All Phases** | `SRC_DEM/DTIE_GNN_ORCHESTRATION/phase*.py` (1–6d + 35) | `science/dtie/v3/phases/` | Critical | 13 phase files. Keep as close to original as possible initially |
| **v3 Supporting** | `SRC_DEM/.../ingestion.py`, `source_leak_scoring.py`, `spectral.py`, `compare_graphs.py`, `contracts.py`, `gnn_runner.py` | `science/dtie/common/` or `v3/` | High | Strong candidates for `common/` |
| **v4 GNN Core** | `SRC_VIZ/src/components/TokyoEyesv4/Gnnv4.py` | `science/dtie/v4/gnn/Gnnv4.py` | Critical | Most advanced model |
| **v4 Training** | `SRC_VIZ/src/components/TokyoEyesv4/train_v4.py`, `retrain_stage2.py`, `finetune_differential.py`, `diagnose_v4.py` | `experiments/training/v4/` | Critical | Training harness + loss functions |
| **v4 Phases** | `SRC_VIZ/src/components/TokyoEyesv4/phase1_witness_embedding_v4.py`<br>`phase3_witness_persistence_v4.py` | `science/dtie/v4/phases/` | High | Only partial v4 phases exist |
| **v4 Validation Docs** | `SRC_VIZ/src/components/TokyoEyesv4/VALIDATION_*.md`, `CHECKPOINT_STATUS.md`, `DTIE_v4_migration.md` | `docs/findings/v4/` | High | Gold for understanding current best model |
| **Shared Utils** | Both sources have overlapping `utils.py`, `contracts.py` etc. | `science/dtie/common/` | Medium | Requires reconciliation |

### Agent Layer

| Item | Source | Destination | Priority | Notes |
|------|--------|-------------|----------|-------|
| FastAPI Coordinator | `SRC_AGENT/main.py` | `agent/coordinator/main.py` | High | Heavy cleanup needed (remove old gosp references) |
| Viewport Models & Directives | `SRC_AGENT/main.py` (ViewportState, ViewportDirective, etc.) | `agent/models/` | High | Keep and improve |
| Session Management | `SRC_AGENT` (DynamoDB + in-memory) | `agent/sessions/` | Medium | Good starting point |
| DTIE Tool Wrappers | None (only planned in `docs/dtie-integration-plan.md`) | `agent/tools/dtie/` | Critical | Must be written fresh using the v3/v4 orchestrators |
| Aurora DB helpers | `SRC_AGENT` + schemas in `pipeline-v3/gosp` | `data/aurora/` + `agent/` | High | Existing schema work is valuable |

### Visualizer / Frontend

| Item | Source | Destination | Priority | Notes |
|------|--------|-------------|----------|-------|
| React Poincaré Viewer | `SRC_VIZ/src/` (App.tsx, Visualizer2D/3D, OverlayUI, lib/) | `visualizer/frontend/src/` | High | Core UI asset |
| Built assets | `SRC_VIZ/dist/` | `visualizer/frontend/dist/` (or rebuild) | Low | Prefer rebuilding |
| Any viewer backend | `SRC_VIZ` (scattered) | `visualizer/server/` | Medium | Likely needs to be rewritten cleanly |

### Data & Infrastructure

| Item | Source | Destination | Priority | Notes |
|------|--------|-------------|----------|-------|
| Aurora Schemas + Migrations | `SRC_AGENT/scripts/*.sql` + `pipeline-v3/gosp/db/` | `data/aurora/` | High | Very valuable |
| Data Governance | `SRC_AGENT/tokyoeyes-data-governance/` | `data/governance/` | Medium | Keep as-is |
| Terraform | `SRC_AGENT/terraform/` | `infra/terraform/` | High | **Critical cleanup required** — remove all .tfstate files |
| Dataset Config | `SRC_AGENT/tokyoeyes_dataset_config.json` | `data/` | Low | Reference only |

### Documentation

| Item | Source | Destination | Priority | Notes |
|------|--------|-------------|----------|-------|
| Findings (May 2026) | `SRC_AGENT/docs/findings/` | `docs/findings/` | Critical | Preserve with dates and version tags |
| Session Handoffs | `SRC_AGENT/docs/session-handoff-2026-05-14.md` | `docs/history/` or `docs/` | High | Best narrative of recent work |
| DTIE Integration Plan | `SRC_AGENT/docs/dtie-integration-plan.md` | `docs/architecture/` | High | Now becomes historical reference |
| v4 Spec | `SRC_DEM/docs/dtie_v4_source_leak_pipeline_spec.md` | `docs/specs/` | High | Important design document |
| Bad CLAUDE.md files | All old locations | **Delete / ignore** | Critical | Already documented as harmful |

---

## Secondary / Later Migrations

- `SRC_DEM/backend/ensemble_dtie_full_pipeline.py` → `science/dtie/v3/` or `experiments/`
- `SRC_DEM/protein_brain/` (agentic experiment) → `experiments/agentic/` (low priority)
- Older GNN trainer code from `SRC_DEM/gnn-dtie/` → archive or experiments
- Large result folders, checkpoints, caches → **never** committed (use .gitignore + external storage)

---

## Known Conflicts & Reconciliation Tasks

1. **GNN Interface Differences**
   - v3: `GOSPConeMapper` expects certain input features and returns specific dict structure.
   - v4: Keeps more computation in hyperbolic space, different output format, new losses.
   - **Action:** Create `science/dtie/common/interfaces.py` with versioned protocols.

2. **Phase Overlap**
   - Several phases have both v3 and v4 variants.
   - Decision needed: Which v4 phases are production-ready vs experimental?

3. **Orchestrator Philosophy**
   - v3 version is very complete (includes virtual screening).
   - v4 direction (per spec) wants tighter focus on source-leak detection.
   - Recommendation: Keep both but make v4 orchestrator intentionally narrower.

4. **Training vs Inference Split**
   - Many training scripts hardcode paths and assumptions.
   - Plan: Move all training into `experiments/training/` with clear entrypoints.

---

## Migration Principles

- **Copy first, refactor later** (first 4–6 weeks).
- Every migrated file gets a header comment: `# Migrated from: <source> on <date>`.
- No blind "move everything" — each module must justify its new home.
- v3 and v4 must be independently importable and testable.

---

**This document will be the working reference during consolidation.**

Update it after every significant migration decision or batch of ports.