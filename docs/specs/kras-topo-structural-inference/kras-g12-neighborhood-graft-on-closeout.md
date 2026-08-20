# KRAS G12 neighborhood graft ON arm — closeout (Child 3)

**Date locked:** 2026-07-21  
**Decision:** ON-arm **Pass** — conduit-flex multiplier replicates on active scaffold.  
**Machine stamp:** [`data/gates/kras_g12_neighborhood_graft_on_closeout.json`](../../../data/gates/kras_g12_neighborhood_graft_on_closeout.json)  
**Pre-reg:** [`kras-g12-neighborhood-graft-on-prereg.md`](kras-g12-neighborhood-graft-on-prereg.md)  
**Artifact:** `checkpoints/v66/diagnostics/routing_sparsity/kras_g12_neighborhood_graft_on.json`  
**Make:** `make grade-v66-kras-g12-neighborhood-graft-on`  
**Workstream:** `gnn_perturbation_boundary` child **3**

---

## Verdict (ON: `6GOD ← 6GOF`)

| Quantity | Value |
|----------|-------|
| \|N₁₂\| | **20** (recomputed; includes 60/61) |
| ρ(WT, mut) | 0.902 |
| ρ(single-site, mut) | 0.902 |
| ρ(neighborhood, mut) | **0.905** |
| ρ(scramble, mut) | 0.836 |
| Δρ_single | +0.00043 |
| Δρ_neighborhood | **+0.00308** (~7× single-site) |

**Pass gates:** Δρ_neigh > Δρ_single **and** ρ(neigh) > ρ(scramble).

## Cross-arm comparison (four-quadrant conduit baseline)

| Arm | \|N₁₂\| | Δρ_single | Δρ_neigh | Multiplier | Scramble beaten? |
|-----|---------|-----------|----------|------------|------------------|
| OFF `4LPK←5US4` | 17 | +0.0015 | +0.0075 | ~5× | Yes |
| ON `6GOD←6GOF` | 20 | +0.00043 | +0.00308 | ~7× | Yes |

Active state has higher WT–mut baseline ρ (0.90 vs 0.79), so absolute Δρ is smaller; the **conduit-flex multiplier still holds** (and is at least as strong).

## Claims

**Allowed:** state-symmetric conduit sensitivity on inhibitor-free four-quadrant ON/OFF neighborhood grafts.  
**Forbidden:** full fold rewrite; AA identity; hop-2 or G12V/C without new pre-reg; CB / Ledger B Pass.
