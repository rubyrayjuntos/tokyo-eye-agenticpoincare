# KRAS G12 single-site graft — closeout

**Date locked:** 2026-07-21  
**Decision:** Panel **Pass** retained with honest effect-size read; single-site claim sealed.  
**Machine stamp:** [`data/gates/kras_g12_residue_graft_closeout.json`](../../../data/gates/kras_g12_residue_graft_closeout.json)  
**Pre-reg:** [`kras-g12-residue-graft-prereg.md`](kras-g12-residue-graft-prereg.md)  
**Artifact:** `checkpoints/v66/diagnostics/routing_sparsity/kras_g12_residue_graft.json`  
**Make:** `make grade-v66-kras-g12-residue-graft`

---

## Verdict

**Panel Pass** (both OFF and ON arms):  
ρ(graft, mut) > ρ(WT, mut) **and** ρ(graft, mut) > ρ(scramble, mut).

| Arm | ρ(WT,mut) | ρ(graft,mut) | Δρ | scramble |
|-----|-----------|--------------|-----|----------|
| OFF `4LPK←5US4` | 0.792 | 0.793 | +0.0015 | 0.772 |
| ON `6GOD←6GOF` | 0.902 | 0.902 | +0.0004 | 0.893 |

## What this claims

- **Allele-site physics/geometry sensitivity:** the champion responds to the localized ρ/τ/ss + Cα signature copied from the mut deposit at resseq 12, not to an abstract amino-acid identity token (GNN inputs are not AA one-hots).
- **Directional, not generic:** the matched graft beats a distal scramble (donor resseq 80 → WT site 12).
- **Boundary:** single-node local physics alone cannot override WT backbone / contact-graph inertia — tiny Δρ is expected under distributed allostery and multi-hop message passing.

## What this does not claim

- Electronic or chemical recognition of the Asp side-chain string  
- Full WT→mut state rewrite from site 12 alone  
- Reversal of platform CB concordance Fail or Ledger B interface Fail  
- Hollywood-scale network flip as a success criterion  

## Next probe (not this stamp)

Neighborhood / conduit graft on OFF arm `4LPK ← 5US4` — expand beyond node 12 to 1-hop contacts and Switch edges. Spec: [`kras-g12-neighborhood-graft-prereg.md`](kras-g12-neighborhood-graft-prereg.md) (drafted after this closeout).
