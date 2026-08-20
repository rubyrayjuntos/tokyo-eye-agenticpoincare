# Path 2 — Explicit directionality objective (diam≤9)

**Status: ORPHANED (2026-07-18 cleanup)**  
**Do not train. Do not warm-start. Do not treat as authorized next bet.**

This file documents an **agent-initiated** Move 3 pilot that was registered and
trained without clearing the project’s usual methodology gates (Move 2 causal
Pass was soft / over-claimed; Path 2 grade never run; train-surrogate metrics
had logging defects). Cleanup keeps the **code opt-in at λ=0** and this note for
provenance; it **withdraws** authorization.

---

## What happened (facts)

| Item | Detail |
|------|--------|
| Parent | `chem_mvp_tau_abs_dist_stage_a12_cold_v1` (`|ρ−TAU|` + z-norm feeler) |
| Objective | Train surrogate: maximize pairwise asym of \(I(a\to b)=\|h_a\|\mathrm{softplus}(\langle\hat h_a,\hat h_b\rangle)\) on `encoder_h` |
| Mask | diam≤9 PDBs only (1UBQ, 1TEN, 1HHP, 1LYZ, 4OBE, 1TIM, 1MBN) |
| λ | 0.05 |
| Run | `path2_dir_diam9_swap_warm_v1` (ge11→20 warm) — **checkpoint directory removed in cleanup** |
| Grade | **Never completed** on pre-registered knockout / late≫ep1 gates |
| Code left | `science/dtie/common/directionality_objective.py` + loss hook; **default coeff = 0.0** |

## Why orphaned

1. Authorized from a Move 2 “Pass” that fails the clear Δ_median ≥ +0.10 bar
   (methodology regrade → Partial).
2. Train-surrogate asym ≠ causal directionality; `directionality_asym_index`
   logged NaN; `directionality_asym_raw` diluted by diam>9 zeros.
3. No user-owned re-registration against standing flow-influence / feeler SSOTs.

## Locked trunk (resume here)

- `checkpoints/v66/runs/chem_mvp_tau_abs_dist_stage_a12_cold_v1/v66_best.pt`
- Matched baseline: `checkpoints/v66/runs/chem_mvp_znorm_stage_a12_cold_v1/v66_best.pt`
- Jacobian on z-norm-on: still **forbidden**
  ([`JACOBIAN_ZNORM_DEFECT.md`](JACOBIAN_ZNORM_DEFECT.md))

## If Path 2 is ever reconsidered

Must be a **fresh user registration**: clear success/fail, knockout grading plan,
metrics that average only diam≤9 eligibles, no soft Pass from prior agent moves.
Makefile target `train-v66-path2-dir-diam9` now **errors** and points here.

## Historical references

- Parents: [`ablation.md`](ablation.md), [`../rho-tau-abs-dist-swap/ablation.md`](../rho-tau-abs-dist-swap/ablation.md)
- Diameters: `checkpoints/v66/diagnostics/learned_flow_influence/asymmetry_vs_diameter.json`
