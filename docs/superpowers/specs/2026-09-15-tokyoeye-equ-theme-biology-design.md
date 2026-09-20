# Tokyo Eye EQU — Theme Biology Design

**Gate ID:** `tokyo_eye_equ_theme_biology`  
**Status:** APPROVED_LOCKED  
**Date:** 2026-09-15  
**Approver:** Bot (Ray: sit-back lead for ~200 runs; proceed)  
**Predecessor:** `tokyo_eye_equ_lift_radius` QUALIFIED (MLflow `09f422b04335450d9cbc55ad2f675181`)  
**Experiment:** `tokyoeye/equiformer-v3-moe/geometric/hyperbolic-spine`

## 1. Intent

Geometry hygiene Pass is not biology. This card asks whether the lift_radius spine carries **dehydron / mechanism signal across fold themes** on the locked Stage-A probe panel — not KRAS memorization, not affinity, not Pearson.

## 2. Non-claims

- Not affinity Pass; not ligand binding; Pearson parked.
- Not corpus expansion; not proteome.
- Not MoE remediation (hard-route E0/E1 collapse = diagnostic only).
- Not a visual / Poincaré gate.

## 3. Init / corpus

- **Init:** load QUALIFIED `eqf_equ_lift_radius_*/tokyoeye_best.pt` (epoch 17 lineage). Fresh run id; do not alias champion.
- **Forbid:** champion / C1 Pearson θ; rim_volume / shell_unpack / correct_start Fail θ as silent continue.
- **Panels:** cold-boot pins unchanged (train 8 / probe 6 themes / holdout shock).
- **wrap_max:** **1** (train-era). Refuse wrap_max=19 all-positive dehydron caches for seal metrics.
- Frontend: MPtrj pin sha `59c6c235…`; freeze entire frontend; `tau_clamp_mode=final_only`; pure-hyp strict.

## 4. Sealed gates

### Geometry HOLD (must not regress)

| Gate | Threshold |
|------|-----------|
| pure_hyp_pass | = 1.0 |
| probe_sat_gate | mean ‖z‖₂ < 0.50 |
| radius_spread_gate | std(r_H) > 0.15 |
| finite_h2_gate | = 1.0 |
| equiv_residual_gate | lift+spine ‖Δz‖_∞ < 1e-5 |

`moe_liveness_gate` (soft load H_norm ≥ 0.60) remains logged; **not** a Fail bar on this card (watch item).

### Biology NEW (probe themes only)

| Gate | Threshold |
|------|-----------|
| theme_auprc_coverage | ≥ **4 / 6** probe themes with dehydron AUPRC ≥ **0.55** |
| mean_theme_auprc | mean AUPRC over loaded themes ≥ **0.60** |
| theme_auprc_floor | **no** loaded theme with AUPRC < **0.40** (and finite) |

Scores = σ(mechanism_score); labels = dehydron_labels under wrap_max=1. Train-set AUPRC is telemetry only.

### Diagnostics (never Fail)

- Hard MoE argmax mass on E2+E3; per-token routing entropy.
- Full-system Δ_equiv.
- Per-theme AUPRC table in MLflow + stamp.

## 5. Train budget

20 epochs; probe every 5; τ quadratic 0.60→0.95; same L_rad + volume knobs as lift_radius. Best by finite train mean loss (geometry-eligible), seal on final restored best.

## 6. Exit

- QUALIFIED → next card `tokyo_eye_equ_corpus_expand` (PPI / oncogenic / non-KRAS P-loop diversity) under a new pre-registered design.
- FAILED → new card; no patch of Fail θ; no Pearson revival.
