# Tokyo Eye EQU — Spread HOLD on Restore Spine

**Gate ID:** `tokyo_eye_equ_spread_hold`  
**Status:** APPROVED_LOCKED  
**Date:** 2026-09-16  
**Approver:** Bot (Ray sit-back lead; Jobberworkee owns job hunt)  
**Predecessor:** `tokyo_eye_equ_geoopt_restore` QUALIFIED (MLflow `b0480b85e87f492f965f2993a9298dba`)  
**Context:** `tokyo_eye_equ_theme_restore` FAILED — biology AUPRC gates PASSED (mean≈0.688, 5/6≥0.55) but `radius_spread_gate` failed (std(r_H)≈0.134 &lt; 0.15). **Do not** patch theme_restore Fail θ.  
**Experiment:** `tokyoeye/equiformer-v3-moe/geometric/hyperbolic-spine`

## 1. Intent

Prove Geometry HOLD (esp. radial **spread**) on the Equiformer-pool + geoopt spine after a biology-oriented warm path compressed spread. Keep theme AUPRC as **telemetry** (logged, not Fail bar on this card) so we do not thrash biology vs geometry in one AND-gate.

## 2. Non-claims

- Not a silent continue of theme_restore Fail θ.
- Not affinity / Pearson / CATH / multimers / OMat24.
- Not MoE hard-route remediation.
- Not a visual gate.

## 3. Init / stack

- **Frontend:** `EquiformerPoolFrontend`; forbid SE(3)-lite.
- **Lift:** geoopt PoincareBall.
- **Init:** warm from geoopt_restore QUALIFIED best (`eqf_equ_geoopt_restore_20260916/tokyoeye_best.pt`). Optional ablation: cold spine (flag) — default warm restore.
- **Forbid:** theme_restore / theme_biology / theme_signal Fail θ as spine_init.
- wrap_max=1; freeze Equiformer pool; `tau_clamp_mode=final_only`; pure-hyp strict.
- Train knobs: raise volume/spread pressure vs theme_restore (L_rad / σ target / lam_spread) — pin exact coeffs in pins.json; no ad-hoc mid-run edits.

## 4. Sealed gates

### Geometry HOLD (Fail bar)

| Gate | Threshold |
|------|-----------|
| pure_hyp_pass | = 1.0 |
| probe_sat_gate | mean ‖z‖₂ &lt; 0.50 |
| radius_spread_gate | std(r_H) &gt; **0.15** |
| finite_h2_gate | = 1.0 |
| equiv_residual_gate | lift+spine &lt; 1e-5 |
| restore_frontend_gate | equiformer_v3_pool |

### Biology TELEMETRY (logged; not Fail)

- per-theme dehydron AUPRC table; mean / coverage vs 0.55/0.60/0.40 thresholds as watch metrics only.

## 5. Budget

20 epochs; probe every 5; τ 0.60→0.95 quadratic; AdamW spine.

## 6. Exit

- QUALIFIED → biology Pass card that AND-gates theme AUPRC + HOLD (or corpus_expand if biology already strong enough as watch).
- FAILED → new card; no Fail-θ patch.
