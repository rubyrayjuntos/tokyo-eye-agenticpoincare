# Tokyo Eye v7 — B′ core radial floor continue (prereg)

**Status:** PREREG  
**Date:** 2026-07-21  
**Decision:** Soft **core radial floor** (weight ∝ ρ·(1−τ), `min_r=0.15`) to lift e1 off origin without rim-push; keep Option B save hygiene + mild disc occupancy.

## Why

Disc continue_v1: eligible saves OK; `disc_r_mean` peaked ~0.22 (bar 0.25). e2/e3 already ≥0.25; **e1** (τ≈0, ρ≈20, disc_r≈0.07) drags the mean. Viewers show full angular spread + mild core collapse; expert colors look scattershot because niches are **feature** (τ/ρ/SS), not geographic wedges.

## Resume

| Field | Value |
|-------|--------|
| Checkpoint | `checkpoints/v7/runs/tokyo_eye_v7_bprime_disc_continue_v1/v7_best_disc.pt` |
| Locks | `hyp_mp_primary=True`, `se3_aux=False`, `hyperbolic_mp_graph=False` |

## Stack delta

| Knob | Value |
|------|--------|
| `core_radial_floor_coeff` | **1.0** |
| `core_radial_floor_min_r` | **0.15** (near-origin floor, not rim) |
| `disc_occupancy_coeff` | 0.70 (carry Option B) |
| `disc_depth_scale_coeff` / target | 1.60 / 0.60 |
| Save | sparsity-style mean_H ∈ [0.25, 0.90]; no legacy H(f̄)≤1.21 |
| `routing_entropy_sparsity_coeff` | 0 |

## Health Pass bars

Same as disc-health continue: `disc_r_mean≥0.25`, hyp-MP audit, finite curvature, no NaN. Prefer eligible `v7_best`. **No biology / promote.**

## Commands

```bash
make train-v7-bprime-core-floor-continue DEVICE=cuda EPOCHS=24
```
