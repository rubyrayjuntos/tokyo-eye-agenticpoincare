# KRAS G12 neighborhood / conduit graft — pre-registration

**Status:** GRADED — OFF-arm **Pass**; closeout sealed 2026-07-21  
**Closeout:** [`kras-g12-neighborhood-graft-closeout.md`](kras-g12-neighborhood-graft-closeout.md) · stamp `data/gates/kras_g12_neighborhood_graft_closeout.json`  
**Prerequisite closeout:** [`kras-g12-residue-graft-closeout.md`](kras-g12-residue-graft-closeout.md) (single-site Pass sealed)  
**Date locked (pre-reg):** 2026-07-21  
**Checkpoint:** `FIX1_SPARSITY_CHAMPION_CKPT`  
**Make:** `make grade-v66-kras-g12-neighborhood-graft`  
**Stamp:** `data/gates/kras_g12_neighborhood_graft_prereg.json`  
**Artifact:** `checkpoints/v66/diagnostics/routing_sparsity/kras_g12_neighborhood_graft.json`

---

## Motivation

Single-site graft (`4LPK←5US4` / `6GOD←6GOF`) showed allele-site physics sensitivity with **tiny** Δρ. Hypothesis: a true mut-like field rewrite requires the **transport conduit** (local contact geometry) to flex, not only node-12 features.

## Scope (this stamp)

**Primary arm only:** OFF matched-state **`4LPK ← 5US4`** (chain A).  
ON arm and other alleles (G12V/C) are **out of scope** until this arm is graded.

## Graft definition (conduit)

On the WT scaffold (`4LPK`):

1. Identify WT graph index of resseq **12**.  
2. Let \(N_{12}\) = resseq 12 ∪ all residues with a Cα contact edge to 12 on the **mut** graph (`5US4`) at the standard training cutoff (10 Å), intersected with residues present on WT.  
3. Optionally include Switch lock partners **32, 60, 61** if present (union into \(N_{12}\)).  
4. **Neighborhood graft:** for every resseq \(r \in N_{12}\), copy mut `data.x` row + Cα onto the matching WT index; rebuild full Cα contact graph on the WT scaffold.  
5. **Controls:**
   - **Single-site baseline:** re-run / cite site-12-only graft on the same arm (sealed closeout numbers).  
   - **Scramble neighborhood:** same cardinality set of distal residues on mut (exclude 12 and Switch 25–40 / 57–75), grafted onto WT’s \(N_{12}\) positions (or onto WT distal sites — lock one scheme below).

**Locked scramble scheme:** sample \(|N_{12}|\) mut residues with resseq ∉ {12} ∪ Switch-I(25–40) ∪ Switch-II(57–75), sorted by resseq; map in sorted order onto sorted \(N_{12}\) on WT (features + Cα); rebuild edges.

## Metric

Forward-knockout `out_effect` (Jacobian forbidden). Align on shared deposited resseqs among WT / single-site / neighborhood / scramble / mut.

## Pass form (OFF arm)

**Primary (both required):**

1. ρ(neighborhood, mut) − ρ(WT, mut) **>** ρ(single-site, mut) − ρ(WT, mut)  
   i.e. neighborhood improves toward mut **more** than single-site did.  
2. ρ(neighborhood, mut) > ρ(scramble_neighborhood, mut)

**Secondary report-only (not Pass):**

- Absolute Δρ_neigh = ρ(neighborhood, mut) − ρ(WT, mut)  
- Spearman of Δ fields: (neighborhood − WT) vs (mut − WT)  
- |N₁₂| and member resseqs  

**Interpretation if Fail:** global WT contact inertia dominates even with local conduit flex → next hypothesis is larger conformational / multi-hop paste or active-backbone shift (new pre-reg required).

## Explicit non-goals

- Allele fan-out (G12V/C)  
- ON arm in this stamp  
- AA identity claims  
- Reopening CB concordance / Ledger B interface
