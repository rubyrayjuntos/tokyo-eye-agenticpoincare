# Generic flow–centrality concordance (platform grade)

**Status:** HISTORICAL — smoke graded **FAIL**; primary Pass claim **retired** (hard-Fail closeout, 2026-07-21)  
**Policy closeout:** [`platform-concordance-policy-closeout.md`](platform-concordance-policy-closeout.md) · stamp `data/gates/platform_concordance_policy_hard_fail.json`  
**Checkpoint graded:** `FIX1_SPARSITY_CHAMPION_CKPT`  
**Make (report-only re-runs):** `make grade-v66-fix1-general-hub-alignment`  
**Artifact:** `checkpoints/v66/diagnostics/routing_sparsity/general_hub_alignment_smoke.json`

> Do **not** lower the 0.50 bar to manufacture Pass. Gradient / Jacobian interpretation questions remain open research — they do not reopen this Fail.

## Principle

Decouple **model signal** (forward-knockout `out_effect`) from **interpretation** (classical betweenness \(C_B\)). Pass/Fail uses rank-order concordance only — **never** hardcoded residue IDs.

## Default smoke panel (`TRAINING_TARGETS`, non-KRAS)

| PDB | Role |
|-----|------|
| `3PP0` | SRC kinase |
| `2SHP` | SHP2 phosphatase |
| `2HHB` | Haemoglobin (blind fold; small) |

**Not used:** `1STP` / `1PTP` / `1B0N` — absent from `TRAINING_TARGETS`; `1STP` is streptavidin, not a kinase.  
`2Z6H` (ARM, n≈533) reserved for extended panel — too slow for default smoke knockout.

## Pass bar

1. Full-structure Spearman(\(out\_effect\), \(C_B\)) **> 0.50**
2. Stability: \(|\Delta\rho| \le 0.05\) when \(C_B\) recomputed at 7.5 Å and 8.5 Å cutoffs

Top-10% hub enrichment is **report-only**. KRAS 81/114/156 → `audit_reports/` only (`--with-kras-audit`).

## Dual ledger (2026-07-20)

| Ledger | Spec |
|--------|------|
| A — Geometric \(C_B\) | This document; bar **0.50** unchanged |
| B — Interface alignment | [`ledger-b-interface-prereg.md`](ledger-b-interface-prereg.md) — SRC/SHP2 sets locked **before** hub extraction |
| Tier-2 historical | [`../kras-pathway-tier2/README.md`](../kras-pathway-tier2/README.md) |

**Ordering:** do not extract champion top-k% flow hubs for literature mapping until `data/gates/ledger_b_interface_prereg_src_shp2.json` is present (it is).
