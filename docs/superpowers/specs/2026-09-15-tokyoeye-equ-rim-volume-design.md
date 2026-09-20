# Tokyo Eye EQU — Hyperbolic Volume Pressure & Equivariance Alignment Design

**Gate ID:** `tokyo_eye_equ_rim_volume`  
**Display Lineage:** Tokyo Eye EQU  
**Status:** APPROVED_LOCKED  
**Date:** 2026-09-15  
**Approver:** Ray Swan (operator proceed; Bot amendments accepted)  
**Experiment (Domain Charter):** `tokyoeye/equiformer-v3-moe/geometric/hyperbolic-spine`  
**Governance Standard:** `tokyo_eye_equ_governance`  
**Amended Freeze:** `tokyo_eye_equ_pure_hyp_v1`  
**Predecessor Run:** `tokyo_eye_equ_correct_start` (Sealed Failure: μ_sat≈1.00, Δ_equiv=0.675; MLflow `19115256357c4a41ba81178fbf5e35dc`)

---

## 1. Executive Summary & Diagnostic Root Cause

`tokyo_eye_equ_correct_start` established manifold purity (`pure_hyp_pass=1.0`) and healthy MoE routing (`H_norm=0.876`) but failed:

1. **Rim collapse** — no interior volume pressure; message passing packs mass near the ball boundary / curriculum shell.
2. **Equivariance leakage** — `RadialAngularProjector` injected vector *direction* `â` into Poincaré coordinates (frame-dependent). Separately, the frozen Linear frontend is **not** a true e(3) feature map; full-graph `R·x` invariance is out of scope for this card.

Per governance: discard correct_start θ. Fresh spine + pinned MPtrj bank. No remediation theater on Fail weights.

### Operator amendments (2026-09-15)
- Seal **μ_sat** as mean Euclidean ball radius `< 0.50` (not fraction `r > 0.80`).
- Seal **σ_radius** as `std(r_H)` of hyperbolic radii `> 0.15`.
- Seal **Δ_equiv** as **lift+spine** invariance: hold scalars `s`, rotate vector channels `v ← v Rᵀ`, require `‖z(s,v) − z(s,vRᵀ)‖_∞ < 1e-5`. Full-system `R·x` residual is logged as diagnostic only.
- Frontend pin = locked MPtrj path (not missing `checkpoints/v8/...`).
- `τ_ceiling` remains an upper clamp (0.60→0.95 quadratic); interior pressure comes from `L_vol`, not from lowering τ below the sat threshold.

---

## 2. Architectural Modifications

### 2.1 Hyperbolic Interior Volume Loss (`L_vol`)

$$
\mathcal{L}_{\mathrm{vol}} = -\frac{1}{N}\sum_i \log(1 - c\|z_i\|_2^2) + \lambda_{\mathrm{spread}}\max\bigl(0,\ \sigma_{\mathrm{target}} - \mathrm{std}(r_H(z_i))\bigr)
$$

with `σ_target = 0.20`, `λ_spread = 0.5`, applied on `z_hyp` each train step (`volume_coeff` default 1.0).

### 2.2 Invariant e(3) Scalar Lift (`RadialAngularProjector`)

Forbidden: mapping directional `â = v/‖v‖` into Poincaré tangent directions.

Required:

$$
s_{\mathrm{inv}} = s \oplus \bigl(\|v\|_2,\ \text{optional pairwise }\langle v_a,v_b\rangle\bigr),\quad
z_0 = \exp_0^c\bigl(\mathrm{MLP}_{\mathrm{inv}}(s_{\mathrm{inv}})\bigr)
$$

scaled into the curriculum radius via a bounded tangent before `exp_0`.

---

## 3. Pre-Registered Advance Gates

| Gate ID | Metric | Threshold |
| :--- | :--- | :--- |
| `pure_hyp_pass` | Static + live AST | `= 1.0` |
| `probe_sat_gate` | `μ_sat = mean ‖z‖_2` | **`< 0.50`** |
| `radius_spread_gate` | `σ_radius = std(r_H)` | **`> 0.15`** |
| `finite_h2_gate` | Finite hyp features | `= 1.0` |
| `moe_liveness_gate` | `H_norm` | **`≥ 0.60`** |
| `equiv_residual_gate` | Lift+spine `‖z(s,v)−z(s,vRᵀ)‖_∞` | **`< 1e-5`** |

Any single Fail → DISQUALIFIED / FAILED.

---

## 4. Execution Scope

- **MLflow experiment:** `tokyoeye/equiformer-v3-moe/geometric/hyperbolic-spine`
- **Run name:** `tokyo_eye_equ_rim_volume`
- **Stamp:** `data/gates/tokyo_eye_equ_rim_volume.json`
- **Frontend pin:** `checkpoints/tokyoeye/pretrained/hf/checkpoint/mptrj_gradient.pt`  
  sha256 `59c6c23573a3b05b347662f473209d1bb1ccb5b85f6624a8f87072e7c397addf`  
  (symlink `checkpoints/tokyoeye/pretrained/equiformer_v3_baseline.pt` OK)
- **Panels:** same cold-boot / correct_start train 8 / probe 6 / holdout 4
- **Init:** fresh hyp spine + MoE; **no** correct_start / champion / affinity / C1 θ
- **Epochs:** 20; quadratic τ 0.60→0.95; AdamW spine `3e-4`, wd `1e-4`; frontend frozen (geometry Pass)
- **PDB dir in science:** `/tmp/dtie_pdb_cache`

---

## 5. Exit Criteria

- All 6 gates Pass → `QUALIFIED`; tag `tokyo_eye_equ_next_open` → `tokyo_eye_equ_affinity_coupling`
- Else → `FAILED`; evidence in MLflow; **do not** patch Fail θ

---

## 6. Sign-off

**Approver:** Ray Swan  
**Date:** 2026-09-15  
**Amendments:** Bot technical vetoes accepted; proceed amended
