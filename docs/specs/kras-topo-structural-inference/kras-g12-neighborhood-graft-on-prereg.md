# KRAS G12 neighborhood graft — ON arm (Child 3)

**Status:** GRADED — ON-arm **Pass**; closeout sealed 2026-07-21
**Closeout:** [`kras-g12-neighborhood-graft-on-closeout.md`](kras-g12-neighborhood-graft-on-closeout.md) · stamp `data/gates/kras_g12_neighborhood_graft_on_closeout.json`  
**Date locked:** 2026-07-21  
**Workstream:** `gnn_perturbation_boundary` child **3**  
**Depends on:** Child 1 (single-site) + Child 2 (OFF neighborhood) Pass closeouts  
**Checkpoint:** `FIX1_SPARSITY_CHAMPION_CKPT`  
**Make (to wire):** `make grade-v66-kras-g12-neighborhood-graft-on`  
**Stamp:** `data/gates/kras_g12_neighborhood_graft_on_prereg.json`  
**Artifact:** `checkpoints/v66/diagnostics/routing_sparsity/kras_g12_neighborhood_graft_on.json`

---

## Motivation

OFF neighborhood (`4LPK←5US4`) showed ~5× Δρ vs single-site and beat scramble — conduit flex helps propagation. Child 3 tests **state symmetry**: does the same conduit definition replicate on the active GppNHp scaffold?

## Scope

**Arm:** ON matched-state **`6GOD ← 6GOF`** (chain A).  
OFF arm is already closed (Child 2). G12V/C and hop-2 are **out of scope**.

## Graft definition (same protocol as Child 2)

1. Build \(N_{12}\) on **mut** (`6GOF`): seed 12 ∪ Cα neighbors @ 10 Å ∪ Switch lock partners {32, 60, 61} if present, intersected with residues on WT (`6GOD`).  
2. **Do not** hard-code the OFF arm’s 17-member list — membership is recomputed (may differ under active Switch geometry).  
3. **Neighborhood graft:** for each \(r \in N_{12}\), copy mut `data.x` + Cα onto WT; rebuild Cα contacts.  
4. **Single-site baseline:** graft resseq 12 only (same arm).  
5. **Scramble:** \(|N_{12}|\) distal mut donors (exclude 12 ∪ Switch-I 25–40 ∪ Switch-II 57–75), sorted → sorted \(N_{12}\) on WT.

Metric: forward-knockout `out_effect` (Jacobian forbidden). Align shared deposited resseqs.

## Pass form (ON arm)

Both required:

1. Δρ_neigh = ρ(N, mut) − ρ(WT, mut) **>** Δρ_single = ρ(site12, mut) − ρ(WT, mut)  
2. ρ(N, mut) > ρ(scramble, mut)

**Report-only:** |N₁₂| and members; ratio Δρ_neigh / Δρ_single vs OFF (~5×); absolute Δρ.

**Interpretation:**
- **Pass:** conduit-flex multiplier holds in active state (four-quadrant bidirectional baseline closes).  
- **Fail:** active Switch is more rigid / less graft-sensitive → state asymmetry; document before allele fan-out.

## Explicit non-goals

- Hop-2 depth scaling  
- G12V/C  
- AA identity claims  
- Reopening CB / Ledger B Fails
