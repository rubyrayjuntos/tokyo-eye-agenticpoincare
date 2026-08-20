# Tokyo Eye v7 — B′ uncertainty heads recovery (prereg)

**Status:** CLOSED — FAIL (rematch-0 + rematch-1). **Do not extend.**  
**Unpark:** [`bprime-uncertainty-unpark.md`](bprime-uncertainty-unpark.md)  
**Date:** 2026-07-21  
**Depends on:** sealed health [`v7_healthy_sealed.pt`](../../../checkpoints/v7/runs/tokyo_eye_v7_bprime_core_floor_continue_v1/v7_healthy_sealed.pt)  
**Gate stamp:** [`data/gates/tokyo_eye_v7_bprime_uncertainty_heads_prereg.json`](../../../data/gates/tokyo_eye_v7_bprime_uncertainty_heads_prereg.json)

## Why (historical)

Health bank (`epoch_041` / sealed) clears `disc_r_mean≥0.25`, but ale/epi are collapsed. Uncertainty was never in the B′ health stack. This prereg authorized a **heads-only** recovery that **started from inherited P4 coeffs** without a hyp-MP formula benchmark — methodologically insufficient; see unpark brief.

## Locks

| Field | Value |
|-------|--------|
| Resume | `checkpoints/v7/runs/tokyo_eye_v7_bprime_core_floor_continue_v1/v7_healthy_sealed.pt` only |
| `hyp_mp_primary` | `True` |
| `se3_aux` | `False` |
| `hyperbolic_mp_graph` | `False` |
| Train mode | `epistemic_uncertainty_only_train=True` (trunk / gate / hyp_mp frozen) |
| Save | sparsity-style mean_H ∈ [0.25, 0.90] (hygiene; not biology) |
| Abort | Fail if `disc_r_mean` &lt; **0.25** for 2 consecutive epochs |

## Phase (rematch-0)

Adapted from `p4_uncertainty_calibration` — milder, no v3 ale hinge:

| Knob | Value |
|------|--------|
| LR | `5e-5` |
| Epochs | 20–24 |
| `epistemic_decoupling_coeff` | 0.50 |
| `epi_ale_decorrelation_coeff` | 0.40 |
| `epistemic_anticollapse_coeff` | 0.10 |
| `evidential_coeff` | 0.01 |
| v3 ale hinge | **off** (rematch-1 only if authorized) |
| Disc occupancy / feeler geom pressure | **off** (heads-only) |

## Uncertainty Pass bars (rematch-0)

1. `aleatoric_std_mean` ≥ **0.02** (stretch 0.05)
2. `epistemic_std_mean` ≥ **0.01** and `evidence_nu_cv_mean` ≥ **0.02**
3. `|probe_r_epi_ale|` ≤ **0.70**
4. `uncertainty_tau_ale_elevated` **or** relative τ–ale lift ≥ **0.20**
5. Final `disc_r_mean` ≥ **0.25**
6. No NaN; hyp-MP audit unchanged

## Rematch policy (pre-authorized)

- Ale/epi still flat but disc held → one rematch with higher anticollapse / decorrelation (no trunk unfreeze)
- Disc_r breaks → **Fail**, restore sealed health; do not chase uncertainty into biology
- Formal biology only after closeout Pass **or** explicit user waiver

## Non-claims

Biology vs Fix-1; promote; KRAS hubs; chem-MVP; investigation viewer default coloring.

## Commands

```bash
make seal-v7-bprime-healthy
make train-v7-bprime-uncertainty-heads DEVICE=cuda EPOCHS=24
make grade-v7-bprime-uncertainty-heads
# If rematch-0 FAIL with disc held: one authorized rematch
make train-v7-bprime-uncertainty-heads-rematch DEVICE=cuda EPOCHS=24
make grade-v7-bprime-uncertainty-heads RUN_ID=tokyo_eye_v7_bprime_uncertainty_heads_rematch_v1 \
  OUT=data/gates/tokyo_eye_v7_bprime_uncertainty_heads_closeout.json
```
