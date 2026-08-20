# KRAS G12 neighborhood / conduit graft — closeout

**Date locked:** 2026-07-21  
**Decision:** OFF-arm **Pass** — neighborhood Δρ exceeds single-site and beats scramble.  
**Machine stamp:** [`data/gates/kras_g12_neighborhood_graft_closeout.json`](../../../data/gates/kras_g12_neighborhood_graft_closeout.json)  
**Pre-reg:** [`kras-g12-neighborhood-graft-prereg.md`](kras-g12-neighborhood-graft-prereg.md)  
**Artifact:** `checkpoints/v66/diagnostics/routing_sparsity/kras_g12_neighborhood_graft.json`  
**Make:** `make grade-v66-kras-g12-neighborhood-graft`  
**Workstream:** `gnn_perturbation_boundary` child 2

---

## Verdict (OFF: `4LPK ← 5US4`)

| Quantity | Value |
|----------|-------|
| \|N₁₂\| | **17** |
| N₁₂ members | 9–17, 32, 34–35, 58–59, 81–82, 89 |
| ρ(WT, mut) | 0.792 |
| ρ(single-site, mut) | 0.793 |
| ρ(neighborhood, mut) | **0.799** |
| ρ(scramble, mut) | 0.753 |
| Δρ_single | +0.0015 |
| Δρ_neighborhood | **+0.0075** (~5× single-site) |

**Pass gates:** Δρ_neigh > Δρ_single **and** ρ(neigh) > ρ(scramble).

Note: Switch lock 60/61 were not in the deposited WT∩mut intersection used for N₁₂ (32 present).

## Interpretation

- **Conduit hypothesis supported at the margin:** expanding the graft from site 12 to the 1-hop + Switch-touching neighborhood increases directional movement toward mut more than single-site alone, and the matched neighborhood beats a same-cardinality distal scramble.
- **Absolute effect remains small** vs backbone inertia (WT–mut already ρ≈0.79). Seeds + local conduits nudge; they do not fully rewrite the field.
- Still **physics/geometry** (ρ/τ/ss + Cα), not AA identity.

## Claims

**Allowed:** neighborhood/conduit flex increases allele-site directional sensitivity vs single-node graft on this OFF arm.  
**Forbidden:** full state rewrite; AA recognition; CB concordance / Ledger B Pass; silent expansion to G12V/C without new pre-reg.
