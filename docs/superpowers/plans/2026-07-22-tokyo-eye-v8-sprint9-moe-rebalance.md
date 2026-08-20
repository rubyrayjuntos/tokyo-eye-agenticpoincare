# TokyoEye-v8 Sprint 9 — MoE Rebalance Implementation Plan

> **For agentic workers:** Execute tasks in order. Spec SSOT:
> [`docs/superpowers/specs/2026-07-22-tokyo-eye-v8-sprint9-moe-rebalance-design.md`](../specs/2026-07-22-tokyo-eye-v8-sprint9-moe-rebalance-design.md)

**Goal:** Unstick E0–E3 monopoly via CV↑ + min-load quota + exponential Gumbel; log `moe_load_e*`; continue Mode C train.

**Frozen:** R0–R5 / DSSP / double-cone / `v8_biophys_s8` cache — do not touch.

## File map

| File | Responsibility |
|------|----------------|
| `science/tokyo_eye/v8/moe.py` | Raw `cv_loss` + `quota_loss`; no internal λ scale |
| `science/tokyo_eye/v8/engine.py` | Exp/linear `GumbelTemperatureSchedule`; scale CV+quota in `train_v8_step` |
| `science/tokyo_eye/v8/configs/equiformer_v3_weight_map.json` | Defaults: `cv_coeff=10`, quota, schedule |
| `experiments/training/v8/run_v8_experiment.py` | Scale losses once; `moe_load_e*`; `--init-ckpt`; schedule CLI |
| `Makefile` | `CV_COEFF`, `GUMBEL_SCHEDULE`, `MOE_QUOTA_COEFF`, `INIT_CKPT` |
| `tests/v8/test_moe_rebalance_sprint9.py` | Quota hinge + schedule unit tests |

---

### Task 1 — Quota hinge + CV SSOT (`moe.py`)
- [x] `min_load_quota_loss`; unscaled CV/quota in aux

### Task 2 — Exponential Gumbel (`engine.py`)
- [x] exp + linear schedules

### Task 3 — Harness + config + Makefile
- [x] `moe_load_e*`, CLI, config defaults

### Task 4 — Mode C continue train
- [x] `tokyo_eye_v8_mode_c_moe_rebalance_s9` (24 epochs) — **PASS** all `moe_load_e*≥0.05`; no rematch

### Task 5 — Re-export viewers (optional closeout)
- [ ] After accept (optional)
