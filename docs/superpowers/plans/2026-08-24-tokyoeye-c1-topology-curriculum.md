# C1 Topology Curriculum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Train 150 epochs over the locked 25-chain topology corpus from `@champion` with Equiformer fully frozen, then stamp holdout hygiene at τ=0.90.

**Architecture:** Manifest with `role=train|holdout`. C1 runner resolves champion, freezes `system.frontend` entirely, cycles 25 train chains each epoch, probes 6 holdouts every 25 epochs, writes best ckpt on epoch-mean loss with last-step hygiene.

**Tech Stack:** PyTorch, existing `run_epoch` / B0 `observe_one`, MLflow metrics only.

**Spec:** `docs/superpowers/specs/2026-08-24-tokyoeye-c1-topology-curriculum-design.md`

## Global Constraints

- Init: `resolve --alias champion`. Never Mode C. No affinity head train. No alias flags.
- `set_dehydron_wrap_max(1)` once. `tau_end=0.90`. Gumbel half_epochs = 75.
- One epoch = one pass over 25 train chains (not 150 singleton steps).
- Skip failed loads; abort if a theme drops below 2 train chains.
- `biology_pass: false`. Do not promote champion.

---

### Task 1: Manifest + split parse

- Create: `manifests/v8_c1_topology_curriculum_v1.json`
- Create: `experiments/training/v8/c1_topology_curriculum.py`
- Test: `tests/v8/test_c1_topology_curriculum.py`

- [x] 25 train + 6 holdout; disjoint; shock PDBs absent; theme counts 4/4/4/5/4/4.

### Task 2: Freeze + stamp hygiene

- [x] `freeze_entire_frontend` zeros frontend grads; optimizer has only spine params.
- [x] `c1_hygiene_pass(holdout_rows, kras_row)` implements spec Pass (finite; mean sat < 0.50 with `boundary_radius=0.80`; moe_load_min > 0 on ≥4/6 holdouts).

### Task 3: Train CLI

- Create: `experiments/training/v8/run_c1_topology_curriculum.py`
- Reuse `run_epoch`, B0 `observe_one`, champion load from B0 runner.
- [x] 150 epochs × 25 steps; holdout probe every 25; best ckpt; stamp; MLflow no alias.

### Task 4: Execute on CUDA

- [x] pytest
- [x] docker compose science, wrap_max=1, user 1000:1000
- Stamp: `data/gates/tokyo_eye_v8_c1_topology_curriculum.json` — `hygiene_pass: false` (H2 finite; mean sat ≈1.0 at τ=0.90; eval `moe_load_min=0` on 6/6). Best epoch 142, mean loss 1.996. MLflow `04a959c77ba34fc4b864b9a7156f7f64`. No alias.
