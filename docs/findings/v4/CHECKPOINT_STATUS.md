# Tokyo Eyes v4 Checkpoint Status

## 2026-05-19 Training Summary

Two checkpoints with complementary strengths. Architectural limitation prevents
unification — radial and angular geometry share the same degree of freedom.
GNN development at stable resting point. Move to pipeline validation.

---

## Checkpoint 1: Scientific Claims (SBIR)

**File:** `checkpoints_v4_cone_fix2/checkpoint_stage_1.pt`
**Epoch:** 35 | **Date:** 2026-05-19 | **Curvature:** 0.7293

**Use for:** Switch-I differential finding, biological validation, paper figures.

| Probe | Result |
|-------|--------|
| Probe 3 (neighbors) | G12 → Switch-I 33,34,36 (functional, non-local) ✓ |
| Probe 5 (differential) | Switch-I ratio = 2.26x vs null (p95=1.84) ✓ |
| Probe 5 (top movers) | Res 34, 33 are #2, #3 (effector-binding) ✓ |
| Probe 5 (Switch-II) | Res 57,58,60,64 all in top 10 ✓ |
| Probe 4 (experts) | Expert 1: ρ=2.3 (dehydron), Expert 2: ρ=14.0 (exposed) ✓ |
| Probe 1 (burial) | Catalytic DEEPER than surface (Δ=+0.41) ✓ |
| Radial geometry | COLLAPSED: |p|=1.156±0.000 (boundary shell) ✗ |

**Known limitation:** All points on boundary shell. TDA/witness complex will
find artifacts, not genuine topological features. Do not use for Phase 3.

---

## Checkpoint 2: Pipeline (Phase 1 + Phase 3)

**File:** `checkpoints_v4_retrain/checkpoint_stage_2b.pt`
**Epoch:** 75 | **Date:** 2026-05-19 | **Curvature:** 0.7752 | **hyper_scale:** 0.5145

**Use for:** Witness complex, TDA, radial hierarchy, production pipeline.

| Probe | Result |
|-------|--------|
| Probe 3 (neighbors) | G12 → Switch-I 33,34,36 (functional, non-local) ✓ |
| Probe 5 (differential) | Switch-I ratio = 1.13x (not significant) ✗ |
| Probe 4 (experts) | Expert 1: ρ=5.4 (buried), Expert 2: ρ=19.2 (exposed) ✓ |
| Radial geometry | |p|=1.08±0.05, min=0.92, genuine spread ✓ |
| 2D structure | 9/11 proteins at peak (epoch 74), stable 4-7/11 ✓ |
| Burial correlation | Inverted (needs more training with corrected cone loss) ✗ |

**Known limitation:** Switch-I differential lost during angular reorganization.
Burial geometry inverted (corrected cone loss applied but not converged).

---

## Invariant Across Both Checkpoints

These findings are robust and reproducible regardless of training configuration:

- **Functional neighbor topology (Probe 3):** G12 retrieves Switch-I allosteric
  partners (33, 34, 36) as hyperbolic neighbors despite 7-11Å physical distance.
  Q61 retrieves Switch-II cluster (59, 60, 63). This is the strongest single
  result for platform claims.
- **Expert routing specialization:** Buried/dehydron vs exposed separation held
  across all training runs.
- **Angular community structure:** Non-local functional relationships encoded
  in hyperbolic geometry, not spatial adjacency.

---

## Pipeline Assignment

| Pipeline Phase | Checkpoint | Reason |
|---------------|------------|--------|
| Phase 1 v4 (residue hyperbolic) | Checkpoint 2 | Genuine radial spread for depth/cone_width |
| Phase 3 v4 (witness persistence) | Checkpoint 2 | TDA needs interior geometry, not boundary shell |
| SBIR differential figure | Checkpoint 1 | Switch-I 2.26x, null-calibrated, validated |
| Platform claims (Probe 3) | Either | Functional topology identical in both |

---

## What Requires v5 Architecture

Unifying radial spread + Switch-I differential requires decoupling the radial
dimension from the angular dimension. Current architecture conflates them —
domain separation loss reorganizes angular structure, which scrambles the
WT/G12D differential axis. A v5 fix would use separate geometric subspaces
(e.g., radial controlled by one head, angular by another) with orthogonal
loss terms.

## Architecture Changes Made (v4.3)

In `Gnnv4.py` forward pass, Step 3:
- L2 normalize tangent vectors before expmap0 (not layer_norm — that gives √dim norm)
- `hyper_scale` parameter controls radius via `softplus(hyper_scale)` multiplier
- After `mobius_proj`, re-normalize by MEAN norm (not per-vector) to preserve
  radial variance from direction-dependent Möbius transform
- Cone loss target flipped: `target_depth = (ρ/30)` (high ρ = buried = deep)

## Files

```
Gnnv4.py                  — Model with v4.3 architectural fixes
train_v4.py               — Original 4-stage training script
retrain_stage2.py         — Stage 2 retraining (produced checkpoint 2)
finetune_differential.py  — Differential loss fine-tune (did not recover signal)
diagnose_v4.py            — 5-probe interpretability diagnostics
CHECKPOINT_STATUS.md      — This file
```


## What Requires v5 Architecture

Unifying radial spread + Switch-I differential requires decoupling the radial
dimension from the angular dimension. Current architecture conflates them —
domain separation loss reorganizes angular structure, which scrambles the
WT/G12D differential axis. A v5 fix would use separate geometric subspaces
(e.g., radial controlled by one head, angular by another) with orthogonal
loss terms.

## Architecture Changes Made (v4.3)

In `Gnnv4.py` forward pass, Step 3:
- L2 normalize tangent vectors before expmap0 (not layer_norm — that gives √dim norm)
- `hyper_scale` parameter controls radius via `softplus(hyper_scale)` multiplier
- After `mobius_proj`, re-normalize by MEAN norm (not per-vector) to preserve
  radial variance from direction-dependent Möbius transform
- Cone loss target flipped: `target_depth = (ρ/30)` (high ρ = buried = deep)

## Files

```
Gnnv4.py                  — Model with v4.3 architectural fixes
train_v4.py               — Original 4-stage training script
retrain_stage2.py         — Stage 2 retraining (produced checkpoint 2)
finetune_differential.py  — Differential loss fine-tune (did not recover signal)
diagnose_v4.py            — 5-probe interpretability diagnostics
CHECKPOINT_STATUS.md      — This file
```
