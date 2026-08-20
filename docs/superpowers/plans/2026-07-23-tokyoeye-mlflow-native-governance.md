# TokyoEye MLflow-Native Governance Implementation Plan

> **For agentic workers:** Use this plan for remaining P0 pipeline work. Prefer Python MLflow API over Make/docker-exec.

**Goal:** Full Make-free day-to-day loop: train under taxonomy → evaluate → register pyfunc → `@experimental` (`@champion` explicit only).

## Confirmed status (2026-07-30)

| Area | Status |
|------|--------|
| Runtime restore `models:/TokyoEye@alias` (gnn_inference fail-closed) | Done |
| Artifact proxy (`--serve-artifacts`, `mlflow-artifacts:/`) | Done |
| evaluate / register / set-alias / import-weights / promote CLI | Done |
| Full PyFunc Equiformer+MoE load/predict | Done |
| Train joins taxonomy run + evaluate→register→@experimental | Done |
| Threshold packs as MLflow artifact SSOT | Done (`seed-thresholds` + resolve order) |
| Live `train --pipeline` smoke (@champion unchanged) | Done (v3 registered; aliases unchanged) |
| Legacy taxonomy FS roots auto-archive → proxied | Done (`ensure_taxonomy_experiment`) |
| Legacy experiments 8/9 host `models:/` download | Deprioritized (history under `.legacy_fs_*`) |

## Priority (locked)

1. Day-to-day train without `--skip-train` (optional next)  
2. `@champion` remains explicit approval  

## Global constraints

- No Make for MLflow governance  
- Host API via `http://localhost:5000` preferred for agent ops  
- Do not move `@champion` in the automated train pipeline  

## Remaining tasks

### Task A: Threshold packs as MLflow artifacts — DONE

- `science/tokyo_eye/governance/thresholds.py`
- CLI: `seed-thresholds`; evaluate/promote/train resolve via `--thresholds-run-id` → template → legacy path
- `GNN_LIFECYCLE.md` updated

### Task B: Day-to-day full smoke on live server — DONE

- Seeded template run `1d11723371834f2faf3eb22da79a0105`
- Pipeline run `9bd9c5f7e0ef4874ab9471d2af2a3c2c` → TokyoEye **v3** (no alias move with `--no-experimental-alias`)
- `@champion` stayed **v1**; `@experimental` stayed **v2**
- Canonical geometric/full-stack experiment is now **id 11** with `mlflow-artifacts:/`; old id 8 renamed `.legacy_fs_8`

### Task C (later): Optional affinity-head FS archive

- Same `ensure_taxonomy_experiment` path will archive chemical/affinity-head on first write
