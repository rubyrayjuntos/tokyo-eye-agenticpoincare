# KRAS G12 residue-12 graft validation (four-quadrant)

**Status:** GRADED — Panel **Pass**; closeout sealed 2026-07-21  
**Closeout:** [`kras-g12-residue-graft-closeout.md`](kras-g12-residue-graft-closeout.md) · stamp `data/gates/kras_g12_residue_graft_closeout.json`  
**Date locked (pre-reg):** 2026-07-21  
**Checkpoint:** `FIX1_SPARSITY_CHAMPION_CKPT`  
**Make:** `make grade-v66-kras-g12-residue-graft`  
**Stamp (pre-reg):** `data/gates/kras_g12_residue_graft_prereg.json`  
**Artifact:** `checkpoints/v66/diagnostics/routing_sparsity/kras_g12_residue_graft.json`

> Single-site claim is sealed. Next causal probe: neighborhood/conduit graft — [`kras-g12-neighborhood-graft-prereg.md`](kras-g12-neighborhood-graft-prereg.md).

---

## Claim under test

Matched-state **local transplant** of mut residue **12** (node features + Cα) onto the WT scaffold moves knockout `out_effect` **toward** the real mutant relative to WT, and beats a distal scramble control.

This probes whether champion weights carry a usable allele map — not whether disc layout or CB concordance Pass.

## Roster (inhibitor-free four-quadrant)

| Arm | WT scaffold | Mut donor / target |
|-----|-------------|--------------------|
| OFF (GDP) | **4LPK** | **5US4** |
| ON (GppNHp) | **6GOD** | **6GOF** |

## Graft definition

1. Load WT and mut graphs (chain A).  
2. **Graft:** copy mut `data.x` row + Cα at resseq **12** onto WT index of resseq **12**; rebuild Cα contacts @ 10 Å.  
3. **Scramble control:** same, but donor resseq = **80** (distal) onto WT site **12**.  
4. Metric: forward-knockout `out_effect` (Jacobian forbidden). Align on shared deposited resseqs across WT / graft / scramble / mut for that arm.

**Honesty:** GNN node inputs are ρ/τ/ss (topology), not AA one-hots. Asp vs Gly enters via deposited local physics/geometry at site 12 on the mut deposit — not a literal Gly→Asp token.

## Pass form (per arm)

1. ρ(graft, mut) > ρ(WT, mut)  
2. ρ(graft, mut) > ρ(scramble, mut)  

**Panel Pass:** both OFF and ON arms Pass.

## Report-only

- Spearman of Δ fields: (graft − WT) vs (mut − WT)  
- Top |Δ out_effect| residues after graft  
- Coupled-lock neighborhood 12–32 / 60 / 61 ranks  

## Explicit non-goals

- Does not reopen platform CB concordance or Ledger B interface Fail.  
- Does not claim unconstrained OOD runtime.  
- Other G12 alleles / inhibitor-bound deposits are expansion — out of scope for this stamp.
