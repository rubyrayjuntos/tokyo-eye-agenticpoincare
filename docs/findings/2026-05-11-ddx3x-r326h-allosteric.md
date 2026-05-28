# DDX3X R326H Allosteric Finding

**Date:** 2026-05-11
**Method:** DTIE structural physics — dehydron analysis + Phase 2 + Phase 4b conductance
**GNN status:** NOT APPLIED to interpretation. GNN provided epistemic filtering only.

## Reference Structure

- **5E7I** — DDX3X catalytic core, WT apo, 2.22Å, Chain A (430 residues, positions 134-577)
- Full two-domain DEAD-box helicase (RecA-like domain 1 + domain 2)

## Comparisons

| Comparison | PDB | Resolution | Identity | Common Core | RMSD |
|-----------|-----|-----------|----------|-------------|------|
| Apo vs Inhibitor | 5E7J | 2.23Å | 99.76% | 419 | 1.72Å |
| WT vs R326H | 9E2C | 2.30Å | 99.77% | 420 | 0.87Å |

## Core Finding: R326H Creates a Novel Doorway at RNA-Binding Motif Ia

**Single state-selective doorway: GLY 261** (ρ=1.0 in WT, wrapped in R326H)

- GLY 261 is the first residue of **Motif Ia** (GxxRRK, positions 261-265)
- Motif Ia directly contacts RNA 2'-OH during unwinding
- Distance from R326 to GLY 261: **21.8Å** — genuine long-range allosteric effect
- GLY 261 does NOT appear in the apo vs inhibitor comparison (0 doorways)
- Therefore: R326H creates a novel doorway at the RNA-binding motif that does not exist in normal helicase cycling

**Interpretation:** The R326H cancer mutation allosterically tightens the RNA-binding Motif Ia, potentially disrupting RNA engagement. This is a gain-of-rigidity at the RNA interface caused by a mutation in the helicase core.

## Phase 4b: Inter-Domain Coupling Lock

| Metric | WT (5E7I) | Inhibitor (5E7J) | R326H (9E2C) |
|--------|-----------|------------------|--------------|
| λ₂ (spectral gap) | 0.0476 | 0.1695 | 0.2074 |
| Top flux hub | LEU 313 | ASP 350 | ASP 350 |

**The low WT λ₂ (0.048) is biologically correct.** DEAD-box helicases require inter-domain flexibility for their catalytic mechanism — the two RecA-like domains must open and close during ATP-coupled RNA unwinding. A near-zero spectral gap reflects this genuine conformational flexibility (the contact graph nearly splits into two components at the inter-domain interface).

**R326H increases λ₂ by 331%** (0.048 → 0.207). The inter-domain bottleneck disappears — the two domains become rigidly coupled. The helicase cannot cycle because the domains cannot separate. This is a loss-of-function mechanism through over-connectivity: the mutation locks the dynamic interface that needs to be flexible.

**Hub migration: LEU 313 → ASP 350.** The conductance hub shifts from a hydrophobic core residue to an acidic surface residue 24 positions downstream of R326H. The mutation creates a new allosteric hub in its immediate neighborhood.

## Constitutive Dehydron Cluster: RNA-Binding Surface (473-479)

| Residue | ρ (WT) | ρ (R326H) | Identity |
|---------|--------|-----------|----------|
| 473 | 3.0 | constitutive | GLY |
| 474 | 1.0 | constitutive | ASP |
| 475 | 1.0 | constitutive | ARG |
| 476 | 0.0 | constitutive | SER |
| 477 | 0.0 | constitutive | GLN |
| 478 | 0.0 | constitutive | ARG |
| 479 | 2.0 | constitutive | ASP |

This 7-residue cluster (ρ = 0-3, all far below TAU=13) is a **permanent PPI landing pad** on the C-terminal RNA-binding surface. It exists in both WT and R326H — the mutation doesn't create or destroy it.

**Significance for the PPI bridge hypothesis:** DDX3X was identified as an inter-community bridge protein connecting the spliceosome, ribosome, NF-κB, and HR repair complexes. Those bridges are protein-protein interactions. The constitutive dehydron cluster at 473-479 is the structural explanation: a permanently underwrapped surface that is thermodynamically available for protein contacts regardless of ATP or RNA binding state.

This is the physical basis for DDX3X's multi-complex bridging role — it has a standing PPI landing pad that enables simultaneous engagement with multiple protein partners.

## Apo vs Inhibitor: No Wrapping Change

The apo (5E7I) vs inhibitor-bound (5E7J) comparison produced:
- 0 state-selective doorways
- 18 constitutive dehydrons
- λ₂ increase: 0.048 → 0.170

The inhibitor changes the conductance network (increased coupling) without changing dehydron topology. The backbone wrapping is identical between apo and inhibitor-bound states. This means the ATP-binding pocket occupation affects allosteric communication but not surface exposure — a purely dynamic effect rather than a structural one.

## Summary of DDX3X Mechanisms

1. **R326H locks the inter-domain interface** (λ₂: 0.048 → 0.207) — loss of catalytic flexibility
2. **R326H creates a novel doorway at Motif Ia** (GLY 261) — allosteric disruption of RNA binding
3. **Constitutive dehydron cluster at 473-479** — permanent PPI landing pad explaining multi-complex bridging
4. **Normal inhibitor binding** does not change wrapping — ATP pocket occupation is a dynamic, not structural, perturbation

## Comparison to SPOP Pattern

| Feature | SPOP | DDX3X |
|---------|------|-------|
| Convergence | 4 hotspots → same 2 residues | Single mutation → single doorway |
| Mechanism | Allosteric occlusion of regulatory sites | Inter-domain lock + RNA motif disruption |
| λ₂ change | +18% (0.519→0.614) | +331% (0.048→0.207) |
| Constitutive sites | E47 (ρ=7, single residue) | 473-479 (7-residue cluster, ρ=0-3) |
| Therapeutic implication | Restore wrapping at SER 33/THR 56 | Restore inter-domain flexibility |

DDX3X and SPOP represent two distinct allosteric cancer mechanisms:
- SPOP: convergent occlusion (multiple mutations → same downstream effect)
- DDX3X: domain locking (single mutation → global rigidification + focal RNA disruption)


## Pipeline Run IDs

- Apo vs Inhibitor (5E7I vs 5E7J): 4dcb829c-fda8-4d0d-8d0f-da6c4fe4531a
- WT vs R326H (5E7I vs 9E2C): 2225abf9-43cf-4dfa-bfa6-bb60067f0957 (final run)

## Reproducibility

All runs: pipeline_mode=source_leak_v4, n_landmarks=200, target_chain=A, normalization=seqres_fallback (conn=None), TAU=13.0, epistemic filter > median, GNN checkpoint: robust_experts.pt
