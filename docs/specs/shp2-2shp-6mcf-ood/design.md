# SHP2 2SHP → 6CRF OOD migration gate

**Status:** **PHASE COMPLETE (PASS)** — `2SHP → 6CRF` on sparsity champion  
**Date:** 2026-07-20  
**Amended:** 2026-07-20 — active PDB corrected `6MCF` → `6CRF` (identity error; bars unchanged)  
**Checkpoint SSOT:** `FIX1_SPARSITY_CHAMPION_CKPT`  
**Prereg:** [`data/gates/shp2_2shp_6mcf_ood_prereg.json`](../../../data/gates/shp2_2shp_6mcf_ood_prereg.json)  
**Make:** `make grade-v66-fix1-shp2-2shp-6crf-ood`  
**Probe:** `experiments.diagnostics.shp2_2shp_6mcf_ood`  
**Artifact:** `checkpoints/v66/diagnostics/routing_sparsity/shp2_2shp_6crf_ood.json`  
**Result:** Spearman=0.763; perturb Jaccard H₁₀%=1.0×3; G(6CRF)=0.188; max/med=2.29 — all arms Pass.

## PDB identity

| Role | PDB | Notes |
|------|-----|-------|
| Inactive | `2SHP` | Autoinhibited WT SHP2 |
| Active (locked) | **`6CRF`** | Open E76K (Fodor/Stams Nat Commun 2018) |
| Rejected label | `6MCF` | **Not SHP2** — RCSB deposit is 7SK RNA + HIV-1 Tat RBD |

Historical Tier-2 docs used `6MCF` as “open SHP2”; that label is wrong. Corrected **before** the first Pass/Fail stamp on this gate. Bars were not retuned.

## Intent

Confirm that sparsity-champion routing is **state-adaptive** on SHP2: inactive autoinhibited (`2SHP`) → open/active (`6CRF`), without open-state monopoly collapse and without highway fragility to contact-graph perturbations.

Mechanistic axes language: **N-SH2 / PTP coupling** (not Src C-lobe / αC).

Prerequisite: full-chain Gini reduction stamp (`grade-v66-fix1-gini-reduction-analysis`).

## Locked bars

| Arm | Metric | Bar |
|-----|--------|-----|
| 1. State-transition | Spearman(`out_effect`) on shared auth_resseq | **≥ 0.50** |
| 1b (report) | Jaccard(top-10% hubs) on aligned vectors | report only |
| 2. Perturbation (on `6CRF`) | Jaccard(H₁₀%) vs unperturbed for cutoff ±0.5 Å and edge truncate keep=0.8 | **≥ 0.50** on **all** arms |
| 3. Calibration | Gini(`6CRF`) | **∈ [0.12, 0.25]** |
| 3 | max/median (`6CRF`) | **≤ 3.0** |
| 3 | Gini blow-up | **G(6CRF) ≤ G(2SHP) + 0.05** |

**Aggregate:** all three primary arms Pass.

## Fail modes

- Low cross-state Spearman → different shortcut on open state (not same axes).
- Highway Jaccard collapse under cutoff/truncate → sparsity / coordinate artifact.
- Gini or max/median blow-up on `6CRF` → open-state structural monopoly.
