# Ledger B — 2SHP Focal-Hub Sensitivity Check (pre-reg)

**Status:** LOCKED before run  
**Date:** 2026-07-20  
**Target:** `2SHP` chain A  
**Checkpoint SSOT:** `FIX1_SPARSITY_CHAMPION_CKPT`  
**Focal hubs:** R32, I310, N308, V457  
**Artifact:** `checkpoints/v66/diagnostics/routing_sparsity/ledger_b_sensitivity_check.json`  
**Make:** `make grade-v66-fix1-ledger-b-sensitivity-check`

## Bars (locked)

| Arm | Metric | Bar |
|-----|--------|-----|
| Coordinate jitter (σ = 0.1 Å Gaussian on Cα; rebuild contact graph) | Spearman ρ of full `out_effect` ranks (baseline vs each jitter seed) | **> 0.85** (mean and all seeds) |
| Sparsity trajectory | Focal hubs in top-10% `out_effect` on epochs **46 / 48 / 50** of the banked λ=0.0075 run | All four hubs on every available epoch |
| Wrapping / B-factor | ρ vs τ=13; Cα B-factor | **Report-only** (underwrap supports source-leak reading; not a hard gate) |

## Notes

- Jitter bar **0.85** is locked from the protocol; do not retune after seeing results.
- “Varying λ” is implemented as a **realized-sparsity trajectory** (same train λ, different epochs / mean residue-H). A true λ rematch retrain is out of scope for this check.
- Does not change Ledger B recall@0.25 or pre-reg interface set *I*.
