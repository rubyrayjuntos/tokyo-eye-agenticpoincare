# KRAS G12D Allosteric Finding (Corrected)

**Date:** 2026-05-12
**Method:** DTIE structural physics — dehydron analysis + Phase 2 + Phase 4b conductance
**GNN status:** NOT APPLIED to interpretation. GNN provided epistemic filtering only.
**Correction:** This document supersedes the previous KRAS finding which incorrectly assigned 4DSO as WT. 4DSO is G12D (ASP at position 12). The previous run compared G12D (4DSO) vs G12H (7KFU) — a mutant-vs-mutant comparison, not WT-vs-mutant.

## PDB Identity Verification

| PDB | Residue 12 | Identity | Nucleotide | Resolution |
|-----|-----------|----------|-----------|-----------|
| **4OBE** | GLY | **WT** | GDP | 1.22Å |
| **4DSO** | ASP | **G12D** | GDP | 1.24Å |
| 7KFU | HIS | G12H | — | — |

Verified by direct inspection of ATOM records at position 12.

## Corrected Comparison: 4OBE (WT GDP) vs 4DSO (G12D GDP)

| Metric | Value |
|--------|-------|
| Normalization | RMSD=1.46Å, 98.21% identity, 157 common core — **good** |
| Method | uniprot_guided |
| Doorways | 7 state-selective (WT-exposed, G12D-wrapped) |
| Constitutive | 154 |
| λ₂ | 0.241 → 0.476 (**+97%**) |
| Hub shift | GLY 151 → ILE 163 |

## Core Finding: G12 is a Constitutive Dehydron

**GLY 12 is constitutively underwrapped (ρ=9.0) in WT KRAS.** The G12D mutation does NOT create the dehydron at position 12. G12 is already thermodynamically exposed in the wild-type protein.

This is the same pattern as SPOP E47K (ρ=7.0 constitutive): the cancer mutation exploits a pre-existing wrapping deficit rather than creating one. The G→D substitution changes the chemical character (adds a charged carboxylate) at an already-exposed backbone H-bond.

**Updated Purple Cage interpretation:** The Purple Cage as a topological description of the dehydron network around the nucleotide pocket remains valid. However, the mutation does not CREATE the cage — it EXPLOITS a pre-existing structural vulnerability. The allosteric propagation downstream (P-loop wrapping, Switch II wrapping) is the cage effect, triggered by chemical modification at a constitutive dehydron.

## State-Selective Doorways (WT-exposed, G12D-wrapped)

| Residue | ρ (WT) | Epistemic | Structural Context | Distance from G12 |
|---------|--------|-----------|-------------------|-------------------|
| **17 (LYS)** | 10.0 | 0.594 | P-loop (phosphate binding) | ~5 residues, local |
| **68** | 9.0 | 0.581 | Switch II (effector binding) | ~20Å |
| **85** | 8.0 | 0.612 | α3 helix | ~25Å |
| **99** | 9.0 | 0.529 | β5 strand | — |
| **126** | 9.0 | 0.538 | α4 helix | — |
| **163 (ILE)** | 10.0 | 0.521 | C-terminal (G12D hub) | 24.9Å |
| **168** | 0.0 | 1.866 | C-terminus (flexible tail) | — |

### Key Doorway Interpretations

**Residue 17 (P-loop):** LYS 17 forms part of the phosphate coordination network. G12D causing increased wrapping at position 17 (5 residues away) is the local conformational consequence of replacing glycine with the bulkier aspartate, propagating through backbone geometry.

**Residue 68 (Switch II):** The effector-binding surface becomes more wrapped in G12D. This means the RAF/PI3K engagement surface changes its thermodynamic accessibility — not just conformation, but wrapping state. This is a more precise structural explanation for altered effector selectivity in G12D.

**Residue 163 (C-terminal, G12D hub):** This residue is both a doorway AND the new conductance hub in G12D. The mutation creates a new allosteric center 25Å from the mutation site.

## Phase 4b: Conductance Network

| Metric | WT (4OBE) | G12D (4DSO) |
|--------|-----------|-------------|
| λ₂ | 0.241 | 0.476 |
| Top hub | GLY 151 | ILE 163 |
| Classification | — | PATHOLOGICAL (+97%) |

Hub migration from GLY 151 to ILE 163: the conductance center shifts from the flexible C-terminal region to a more structured position. G12 → hub 151: 20.2Å. G12 → hub 163: 24.9Å.

## Archetype Classification

KRAS G12D aligns with **Archetype I: Convergent Occlusion** (SPOP pattern):

| Feature | SPOP E47K | KRAS G12D |
|---------|-----------|-----------|
| Pre-existing dehydron | E47, ρ=7.0 | G12, ρ=9.0 |
| Mutation effect | Charge reversal (Glu→Lys) | Geometry + charge (Gly→Asp) |
| Allosteric propagation | 30Å to SER 33/PHE 57 | 20-25Å to Switch II/α3 |
| λ₂ direction | +18% | +97% |
| Functional consequence | Substrate entry occlusion | Effector surface wrapping |

Both exploit pre-existing dehydrons and propagate allosterically to functional surfaces. KRAS has stronger λ₂ increase (+97% vs +18%) suggesting more global network rewiring.

**Subtype distinction:** KRAS doorways are in closer spatial proximity to the mutation site (P-loop at 5 residues, Switch II at ~20Å) compared to SPOP (30Å propagation). This may represent a "local cage" subtype of convergent occlusion where the allosteric effect is concentrated near the mutation rather than propagating to a distant regulatory surface.

## Retired Data (Previous Incorrect Run)

The previous dossier values (λ₂: 0.112 → 0.289, +158%, PDB pair 4DSO/7KFU) represent a G12D vs G12H comparison — two different mutants at the same position. This is a valid experiment (differential allosteric consequences of different G12 substitutions) but is NOT a WT baseline comparison. Those values must not appear as "WT vs G12D" in any document.

**G12D vs G12H as a separate finding:** The λ₂ difference between G12D (0.112 as reference) and G12H (0.289) shows that G12H produces even greater network rigidification than G12D. This is consistent with histidine being bulkier than aspartate and creating more steric disruption at the P-loop. Worth noting but not part of the WT baseline analysis.

## Pipeline Run ID

- WT vs G12D (4OBE vs 4DSO): recorded in Aurora, 2026-05-12

## Reproducibility

pipeline_mode=source_leak_v4, n_landmarks=200, target_chain=A, normalization=uniprot_guided, TAU=13.0, epistemic filter > median, GNN checkpoint: robust_experts.pt
