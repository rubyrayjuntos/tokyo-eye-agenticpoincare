# TokyoEye-v8 Sprint 9 — MoE Expert Rebalance (Mode C)

**Date:** 2026-07-22  
**Status:** APPROVED 2026-07-22 — implement per plan  
**Depends on:** Sprint 8 biophys harden + Mode C KRAS baseline  
**Scope:** **Option A only** — unstick E0–E3 hard MoE monopoly. **No** PDB-Bind affinity head (deferred to Sprint 10).  
**Frozen layers:** R0–R5 graph, Kabsch–Sander / double-cone wrap, `v8_biophys_s8` graph cache — **do not touch**.

---

## Problem

Mode C checkpoint `tokyo_eye_v8_mode_c_kras` shows hard-routing collapse:

| Structure | E0 | E1 | E2 | E3 |
|-----------|----|----|----|----|
| 4LPK | ~92% | ~8% | **0%** | **0%** |
| 5US4 | ~87% | ~13% | **0%** | **0%** |
| 6GOD | **100%** | 0% | **0%** | **0%** |
| 6GOF | ~86% | ~10% | **0%** | ~4% |

E1 partially localizes dehydrons (4LPK: 0.92 vs 0.45 on E0) — biophys labels are usable — but E2/E3 are starved. Soft max for E2 is 0.0 (gate never assigns). Winner-take-all under hard Gumbel + weak CV + fast linear cool-down.

---

## Non-goals

- PDB-Bind / affinity regression (Sprint 10)  
- Changing hyp attention, SE(3)-lite, or dehydron BCE head  
- Soft MoE (keep `hard=True` STE at train; argmax at eval)  
- Re-opening Sprint 8 wrap τ / DSSP constants  

---

## Architecture (aux path only)

```
topology_gate_features → gate logits → Gumbel-Softmax (hard STE)
                                      ↓
                         routing ∈ {0,1}^{N×4}  (row one-hot)
                                      ↓
              L_cv = cv(load) · λ_cv
              L_quota = Σ_e ReLU(ρ_min − load_e)² · λ_quota
                                      ↓
                    total += L_cv + L_quota
```

`load_e = mean_i routing[i,e]` (batch / structure mean).

---

## Frozen open points

### 1. CV load-balance scale

| Param | Sprint 8 | Sprint 9 default | Rematch |
|-------|----------|-----------------|---------|
| `cv_coeff` (`λ_cv`) | 1.0 | **10.0** | **50.0** if any `moe_load_e* < 0.05` after ≥12 epochs |

**Bugfix (in scope):** today `TopologyAwareHardMoE` multiplies CV by `self.cv_coeff`, and `run_epoch` multiplies `moe_aux["cv_loss"]` by `cv_coeff` again. Sprint 9 makes **one** application SSOT: raw CV (+ quota) in `moe_aux`, harness scales by config `cv_coeff` once. Document in closeout.

### 2. Soft minimum-load quota

\[
L_{\mathrm{quota}} = \sum_{e=0}^{3} \mathrm{ReLU}(\rho_{\min} - \mathrm{load}_e)^2
\]

| Param | Value | Notes |
|-------|-------|-------|
| `ρ_min` (`moe_quota_floor`) | **0.05** | matches acceptance ≥5% |
| `λ_quota` (`moe_quota_coeff`) | **5.0** | primary rematch lever if CV alone fails |
| Applied on | hard STE routing (train) | same tensor as CV |

Uniform target 0.25 is **not** forced — only a floor. Experts may specialize above 5%.

### 3. Gumbel temperature schedule

**Default (Sprint 9):** exponential

\[
\tau(t) = \max\bigl(\tau_{\min},\; \tau_{\mathrm{init}} \cdot \exp(-\alpha\, t)\bigr)
\]

| Param | Value |
|-------|-------|
| `τ_init` (`gumbel_tau_start`) | 1.0 |
| `τ_min` (`gumbel_tau_end`) | 0.3 |
| `α` (`gumbel_exp_alpha`) | \(\ln(\tau_{\mathrm{init}}/\tau_{\min}) / T_{\mathrm{half}}\) with \(T_{\mathrm{half}}=12\) → **α ≈ 0.1002** so τ≈0.3 near epoch 12, then clamps |

**Fallback flag (required):** `--gumbel-schedule {exponential,linear}` (Makefile `GUMBEL_SCHEDULE=`).  
`linear` = Sprint 8 cool-down: \(\tau = \tau_{\mathrm{init}} + (\tau_{\min}-\tau_{\mathrm{init}})\, t/T_{\mathrm{epochs}}\) — for A/B comparison runs only; not the default.

Keep `hard=True` Gumbel-Softmax at train; eval remains argmax.

---

## MLflow telemetry (acceptance monitor)

Every epoch, log to experiment `tokyo-eyes-v8`:

| Metric | Definition |
|--------|------------|
| `moe_load_e0` … `moe_load_e3` | `routing.mean(dim=0)[e]` on the step batch |
| `moe_load_min` | \(\min_e \mathrm{load}_e\) |
| `moe_quota_loss` | unscaled \(L_{\mathrm{quota}}\) |
| `loss_cv` | scaled CV term actually added to total loss |
| `gumbel_temperature` | τ used this epoch |
| `gumbel_schedule` | param tag `exponential` \| `linear` |

Console: print `moe_load=[e0,e1,e2,e3]` beside existing structure line.

---

## Training protocol

- Corpus: `manifests/v8_kras_nucleotide_basin_v1.json` (Mode C)  
- Wrap τ: keep Sprint 8 retune path (4OBE median-then-descend); do not change biophys code  
- Epochs: **≥24** (enough for exp schedule + load recovery)  
- Init: continue from `checkpoints/v8/runs/tokyo_eye_v8_mode_c_kras/v8_best.pt` **or** cold live-backbone — **prefer continue** to isolate MoE vs backbone (CLI `--init-ckpt`)  
- Run name: `tokyo_eye_v8_mode_c_moe_rebalance_s9`

---

## Acceptance

| Gate | Pass |
|------|------|
| **`MOE_LOAD_FLOOR`** | On **all four** KRAS structures (eval argmax or train hard mean — report both): `moe_load_e* ≥ 0.05` |
| **`AUPRC_HELD`** | Per-structure `val_dehydron_auprc` mean ≥ Mode C baseline mean − **0.05** (baseline ≈ 0.508 from epochs 0–3 of `tokyo_eye_v8_mode_c_kras`) |
| **`NO_NAN`** | No NaN abort |
| Unit tests | Quota hinge fires below floor / zero above; exp schedule monotonic; linear flag matches Sprint 8 formula |

Fail → rematch once at `cv_coeff=50` and/or `moe_quota_coeff=10`. Second fail → new pre-reg (do not invent soft MoE in this sprint).

---

## Files to touch

| File | Change |
|------|--------|
| `science/tokyo_eye/v8/moe.py` | `min_load_quota_loss`; aux fields; stop double-scaling CV inside module |
| `science/tokyo_eye/v8/engine.py` | `GumbelTemperatureSchedule` exp mode + linear preserved |
| `science/tokyo_eye/v8/configs/equiformer_v3_weight_map.json` | `cv_coeff=10`, `moe_quota_*`, `gumbel_schedule`, `gumbel_exp_alpha` |
| `experiments/training/v8/run_v8_experiment.py` | log `moe_load_e*`; schedule CLI; optional `--init-ckpt` |
| `Makefile` | `CV_COEFF=`, `GUMBEL_SCHEDULE=`, `MOE_QUOTA_COEFF=` |
| `tests/v8/test_moe_rebalance_sprint9.py` | new |

---

## Open decisions (freeze on approval)

1. **Continue vs cold:** default **continue** from Mode C best.  
2. **Linear fallback:** **yes** — `--gumbel-schedule linear`.  
3. **Quota floor 5%** aligned with acceptance (not the earlier 10% sketch).  

---

## Docs after approval

- Plan: `docs/superpowers/plans/2026-07-22-tokyo-eye-v8-sprint9-moe-rebalance.md`  
- Then implement + Mode C rebalance train + re-export KRAS viewers  
- Affinity / PDB-Bind → Sprint 10 (separate spec)
