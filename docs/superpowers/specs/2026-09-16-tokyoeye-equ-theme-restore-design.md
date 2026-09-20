# Tokyo Eye EQU — Theme Biology on Restore Spine

**Gate ID:** `tokyo_eye_equ_theme_restore`  
**Status:** APPROVED_LOCKED  
**Date:** 2026-09-16  
**Approver:** Bot (Ray sit-back lead; CATH/PDB-Bind/OMat deferred)  
**Predecessor:** `tokyo_eye_equ_geoopt_restore` QUALIFIED (MLflow `b0480b85e87f492f965f2993a9298dba`)  
**Supersedes attempt:** `tokyo_eye_equ_theme_biology` FAILED on SE(3)-lite / lift_radius spine — do **not** patch that Fail θ  
**Experiment:** `tokyoeye/equiformer-v3-moe/geometric/hyperbolic-spine`

## 1. Intent

Geometry QUALIFIED on EquiformerV3-to-pool + geoopt PoincaréBall. This card asks whether that **restored** spine carries dehydron / mechanism signal across Stage-A fold themes — not affinity, not Pearson, not CATH hierarchy, not multimers, not OMat24.

## 2. Non-claims

- Not affinity Pass; Pearson parked until ligand inference returns.
- Not CATH / PDB-Bind / AF-cluster / multimer corpus expand (later cards).
- Not MoE hard-route remediation theater.
- Not a visual / Poincaré gate.
- Not continuation of theme_biology Fail θ.

## 3. Init / corpus / frontend

- **Frontend:** `EquiformerPoolFrontend` (official cut before `energy_block`); forbid SE(3)-lite / Stub live_backbone.
- **Lift:** `geoopt.manifolds.PoincareBall` (already in projector).
- **Init:** warm-start from QUALIFIED `eqf_equ_geoopt_restore_20260916/tokyoeye_best.pt` (epoch 19). Fresh MLflow run; no champion alias.
- **Forbid:** theme_biology / theme_signal Fail θ; lift_radius / rim / shell / correct_start as silent continue; stub frontend.
- **Panels:** cold-boot Stage-A (train 8 / probe 6 themes); wrap_max=**1**.
- Freeze entire Equiformer pool; `tau_clamp_mode=final_only`; pure-hyp strict.

## 4. Sealed gates

### Geometry HOLD

| Gate | Threshold |
|------|-----------|
| pure_hyp_pass | = 1.0 |
| probe_sat_gate | mean ‖z‖₂ < 0.50 |
| radius_spread_gate | std(r_H) > 0.15 |
| finite_h2_gate | = 1.0 |
| equiv_residual_gate | lift+spine ‖Δz‖_∞ < 1e-5 |
| restore_frontend_gate | frontend_mode = equiformer_v3_pool |

`moe_liveness_gate` logged; watch only (not Fail bar).

### Biology NEW (probe themes)

| Gate | Threshold |
|------|-----------|
| theme_auprc_coverage | ≥ 4 / 6 themes AUPRC ≥ 0.55 |
| mean_theme_auprc | mean ≥ 0.60 |
| theme_auprc_floor | no theme AUPRC < 0.40 (finite) |

Scores = σ(mechanism_score); labels = dehydron_labels (wrap_max=1).

## 5. Train budget

20 epochs; probe every 5; τ quadratic 0.60→0.95; same L_rad / volume knobs as geoopt_restore. Seal on restored best.

## 6. Exit

- QUALIFIED → next card `tokyo_eye_equ_corpus_expand` (PPI / non-KRAS diversity) under new design — **not** automatic PDB-Bind.
- FAILED → new card; no Fail-θ patch; Pearson stays parked.
