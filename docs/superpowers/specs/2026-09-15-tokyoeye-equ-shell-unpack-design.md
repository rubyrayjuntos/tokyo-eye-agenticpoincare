# Tokyo Eye EQU — Shell Unpack Design

**Gate ID:** `tokyo_eye_equ_shell_unpack`  
**Display Lineage:** Tokyo Eye EQU  
**Status:** APPROVED_LOCKED  
**Date:** 2026-09-15  
**Approver:** Ray Swan (operator: proceed as deemed correct)  
**Experiment:** `tokyoeye/equiformer-v3-moe/geometric/hyperbolic-spine`  
**Governance:** `tokyo_eye_equ_governance` · **Freeze:** `tokyo_eye_equ_pure_hyp_v1`  
**Predecessor:** `tokyo_eye_equ_rim_volume` FAILED (MLflow `1b18631c983549cf9fde2ff0c4f1ca9f`)

---

## 1. Diagnostic (sealed, no remediation theater)

`rim_volume` kept purity, MoE liveness, and **lift+spine Δ_equiv ≈ 1.8e-7**, but failed:

- `μ_sat = mean ‖z‖₂ ≈ 0.95` (gate `< 0.50`)
- `σ_radius = std(r_H) ≈ 0` (gate `> 0.15`)

Train telemetry showed `r ≈ τ` every step (mass glued to the curriculum shell). `L_vol` barrier+spread hinge alone did not unpack the shell while **per-stage τ clamps** after lift / every attn layer / MoE repeatedly re-projected outliers onto the ceiling.

Discard rim_volume θ. Fresh spine. Keep invariant projector (proven).

---

## 2. Architectural modifications

### 2.1 Once-per-forward τ clamp (`tau_clamp_mode = final_only`)

Between stages: `project_to_ball` only (numerical manifold safety).  
Apply `τ_ceiling` clamp **once** on the MoE output (final `z_hyp`).

Default `per_stage` preserved for other cards.

### 2.2 Radial unpack loss (`L_rad`)

$$
\mathcal{L}_{\mathrm{rad}} =
\lambda_{\mathrm{mean}}(\overline{\|z\|_2}-\mu^\star)^2
+\lambda_{\mathrm{spread}}\max(0,\sigma^\star-\mathrm{std}(r_H))
-\frac{\lambda_{\mathrm{bar}}}{N}\sum_i\log(1-c\|z_i\|_2^2)
$$

Sealed knobs: `μ* = 0.35`, `σ* = 0.20`, `λ_mean = 3.0`, `λ_spread = 1.0`, `λ_bar = 0.5`, `volume_coeff = 1.0` (multiplies whole `L_rad`).

### 2.3 Retained from rim_volume

- Invariant scalar lift (no directional `â`)
- Lift+spine Δ_equiv sealed gate (rotate `v`, hold `s`)
- Sat / spread formulas: mean ball radius; std hyp radius
- MPtrj frontend pin; frontend frozen

---

## 3. Pre-registered advance gates

| Gate | Metric | Threshold |
| --- | --- | --- |
| `pure_hyp_pass` | static+live | `= 1.0` |
| `probe_sat_gate` | mean ‖z‖₂ | `< 0.50` |
| `radius_spread_gate` | std(r_H) | `> 0.15` |
| `finite_h2_gate` | finite | `= 1.0` |
| `moe_liveness_gate` | H_norm | `≥ 0.60` |
| `equiv_residual_gate` | lift+spine ‖Δz‖_∞ | `< 1e-5` |

---

## 4. Execution

- Run name / stamp: `tokyo_eye_equ_shell_unpack`
- Pin: `checkpoints/tokyoeye/pretrained/hf/checkpoint/mptrj_gradient.pt` sha256 `59c6c235…`
- Panels: cold-boot 8 / 6 / 4
- Init: fresh spine; **forbid** rim_volume / correct_start / champion / affinity / C1
- Epochs 20; τ quadratic 0.60→0.95; AdamW spine `3e-4`; PDB `/tmp/dtie_pdb_cache`

## 5. Exit

All Pass → QUALIFIED → next open `tokyo_eye_equ_affinity_coupling`.  
Else → FAILED; evidence in MLflow; do not patch Fail θ.
