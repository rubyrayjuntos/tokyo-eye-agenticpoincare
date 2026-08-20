# Cold Hyp MP funnel angfill continue — implementation plan

> **For Claude:** execute task-by-task. Prereg: `docs/specs/tokyo-eye-v7/cold-hyp-mp-funnel-angfill-prereg.md`

**Goal:** Continue from cold best with rim_fanout + geom prior, majority+load-floor specialization, disc ER > 1.5, MLflow on.

## Tasks

### Task 1: Train module
- Create `experiments/training/v7/cold_hyp_mp_funnel_angfill_train.py`
- Resume cold best only; forbid HEALTHY_V7/Fix-1
- Build/load TokyoEye with rim_fanout + geometric_angular_prior; strict=False for new keys
- Loss coeffs per prereg; log disc_effective_rank, routing metrics, MLflow `tokyo-eyes-v7`

### Task 2: Makefile
- `train-v7-cold-hyp-mp-funnel-angfill`

### Task 3: Smoke + full train
- Short smoke then full continue with MLflow
