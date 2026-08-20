# v7 — KRAS G12D hub migration (compare-only vs Fix-1)

**Status:** OPEN — first grade on sealed Θ  
**Date:** 2026-07-21  
**Θ:** `HEALTHY_V7_CKPT` (`v7_healthy_sealed.pt`)  
**Make:** `make grade-v7-kras-g12d-hub-migration`  
**Artifact:** `checkpoints/v7/diagnostics/hub_migration/kras_hub_migration_4obe_4dso.json`

## Why this probe

v66 sparsity champion **Passed** localized conductance migration: under G12D (4DSO vs 4OBE), residue **163** becomes a stronger causal out-hub than on WT.  
v7 replaces Euc MP with **hyperbolic message passing**. Question: does the same physics-native hub migration still appear on sealed disc-health Θ?

This is **B0-style inventory / compare-only** — not a B1 teleconnections rematch, not Ledger B.

## What we are *not* doing

| Forbidden / excluded | Why |
|----------------------|-----|
| Jacobian flow-centrality | Standing defect on z-norm-on trunks ([`JACOBIAN_ZNORM_DEFECT.md`](../learned-flow-influence/JACOBIAN_ZNORM_DEFECT.md)) |
| Disc-only (`hyp_projections_2d`) Pass | Disc ≠ pathway trunk ([`DISC_PROJECTION_NOT_TRUNK_PROXY.md`](../../audit/DISC_PROJECTION_NOT_TRUNK_PROXY.md)) |
| Epistemic doorways / NIG | Uncertainty PARKED on v7 |
| Changing Pass bar after seeing numbers | Bars locked below |

## Panel + identity locks

| PDB | State | G12 | 151 | 163 |
|-----|-------|-----|-----|-----|
| **4OBE** | WT GDP | GLY | GLY | ILE |
| **4DSO** | G12D GDP | ASP | GLY | ILE |

Refuse grade on residue-identity mismatch (same as v66).

## Method (locked)

1. Load sealed TokyoEye-v7; forward once → baseline \`x_hyp\` ∈ ball (post Hyp MP).  
2. For each residue \(i\): zero input \`data.x[i]\`, forward, measure per-residue geodesic displacement  
   \(\delta_i(j) = d_{\mathbb{B}}(x_{\mathrm{hyp}}^{(i)}[j],\, x_{\mathrm{hyp}}^{0}[j];\, c)\).  
3. **Out-effect** of \(i\): \(\mathrm{out}(i) = \mathrm{mean}_{j \neq i} \delta_i(j)\).  
4. Normalize: \(R_{163} = \mathrm{out}(163) / \mathrm{median}_i \mathrm{out}(i)\).  
5. Report ranks and raw out for attribution (median-drift check).

**Primary Pass:** \(R_{163}(4\mathrm{DSO}) > R_{163}(4\mathrm{OBE})\).

**Secondary (report-only):** rank(163) improvement; raw \(\mathrm{out}(163)\) rise (not only median drop); optional in-effect at 163 from knocking G12.

## Fix-1 compare baseline (not Pass/Fail)

From sparsity-champion knockout / hub-migration closeout (v66):

- Pass form identical: \(R_{4\mathrm{DSO}} > R_{4\mathrm{OBE}}\)  
- Champion artifact: `checkpoints/v66/diagnostics/routing_sparsity/kras_hub_migration_4obe_4dso.json`  
- Soft monitor: ΔR and rank(163) 4OBE→4DSO

v7 Fail with live geometry ≠ “Hyp MP is broken”; it means **this sealed health bank has not yet recovered the Fix-1 migration signal**.
