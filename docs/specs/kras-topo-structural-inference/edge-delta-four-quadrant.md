# KRAS edge ΔE — four-quadrant rematch

**Date locked:** 2026-07-20  
**Status:** Primary closeout for classical topology rewiring claim  
**Make:** `make grade-v66-kras-topo-edge-four-quadrant`  
**Artifact:** `checkpoints/v66/diagnostics/routing_sparsity/kras_topo_edge_delta_four_quadrant.json`

---

## Science claim

G12D **rewires inactive-state allosteric networks to act like WT active** — not merely “jammed ON.” Graph form: smaller topological distance from G12D-OFF to WT-ON than from WT-OFF to WT-ON, plus a compressed G12D OFF↔ON span vs WT OFF↔ON.

## Locked roster (inhibitor-free)

| State | Wild Type | G12D |
|-------|-----------|------|
| Inactive (OFF) GDP | **4LPK** | **5US4** |
| Active (ON) GppNHp | **6GOD** | **6GOF** |

Isogenic ON pair (6GOD/6GOF); nucleotide + Mg²⁺ only. Avoid inhibitor-bound deposits (e.g. MRTX1133 / BI-2865 class).

## Pass bars (Cα contact @ 8 Å)

1. **Mimetic inactive:** \(|\Delta E(5US4, 6GOD)| < |\Delta E(4LPK, 6GOD)|\)
2. **Compressed mut span:** \(|\Delta E(5US4, 6GOF)| < |\Delta E(4LPK, 6GOD)|\)

Shared deposited residue numbers across all four structures.

## Report-only

- Dehydron-wrapper complementarity ΔE on the same pairs
- Coupled-lock Cα distances 12–32 / 12–60 / 12–61 across the four PDBs

## Supersedes

Historical triad complementarity Pass  
\(|\Delta E_{\mathrm{comp}}(4DSO,5VQ2)| < |\Delta E_{\mathrm{comp}}(4OBE,5VQ2)|\)  
is **report-only** on `kras_topo_matrix_4obe_4dso_5vq2.json` (5VQ2 is G12V active — wrong allele for this claim).
