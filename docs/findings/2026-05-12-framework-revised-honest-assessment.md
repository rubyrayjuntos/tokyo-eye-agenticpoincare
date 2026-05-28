# Revised Framework: What DTIE Actually Measures

**Date:** 2026-05-12
**Status:** Honest reassessment after critical audit

## The Claim That Died

**Original claim:** "λ₂ direction predicts whether a structural perturbation is cancer-pathological (increase) or physiologically regulatory (decrease)."

**Status: ABANDONED.** This binary discriminator is scientifically indefensible because:
1. Normal substrate binding (KEAP1 Kelch + NRF2, Kelch-only: +8.0%) overlaps with lethal cancer mutations (TP53 R273H: +5.1%)
2. The kinase activation data (CDK2 -29%, SRC -35%) compares macro-state transitions to point mutations — a category error
3. λ₂ does not know if a change is "good" or "evil" — it measures effect size of network rewiring

## What λ₂ Actually Measures

**λ₂ quantifies the magnitude of allosteric rewiring** caused by a structural perturbation. It classifies *how* a mutation breaks a protein, not *whether* it causes cancer.

### Validated Classification (Mechanism of Action)

| Δλ₂ | Mechanism | Examples | Interpretation |
|-----|-----------|----------|----------------|
| +18% to +97% | Distal allosteric clamping | SPOP +18%, KRAS +97% | Mutation propagates to remote regulatory sites |
| +331% | Domain fusion/locking | DDX3X +331% | Mutation abolishes inter-domain dynamics |
| +5% to +8% | Local contact perturbation | TP53 R273H +5.1%, R273C +7.1%, R280K +5.1% | Mutation affects immediate neighborhood only |
| +1% | Noise floor | DAXX same-state +1% | No structural change detected |

### What λ₂ Cannot Do

- Distinguish pathogenic from benign missense mutations (no benign crystallographic control exists)
- Serve as a standalone cancer diagnostic
- Compare point mutations to macro-state transitions on the same scale

## What Survives: Two Robust Physical Findings

### Finding 1: Constitutive Dehydrons Predict Mutation Vulnerability

The strongest, most defensible finding in the entire dossier. Wild-type proteins contain pre-existing thermodynamic "fault lines" — residues with low wrapping (ρ) that are maximally vulnerable to chemical perturbation:

| Protein | Residue | ρ (WT) | Cancer Mutation | Frequency | Mechanism |
|---------|---------|--------|----------------|-----------|-----------|
| TP53 | R248 | **5** | R248W | #1 most common | Exploits deepest dehydron → global unfolding |
| SPOP | E47 | **7** | E47K | Hotspot | Exploits dehydron → allosteric propagation |
| KRAS | G12 | **9** | G12D/V/C | #1 most common | Exploits dehydron → effector surface clamping |

**This is pure structural physics.** No narrative gymnastics required. The ρ value of a WT residue predicts its vulnerability to oncogenic exploitation. Lower ρ = more exposed backbone H-bond = more susceptible to chemical perturbation by missense substitution.

**Testable prediction:** Across all cancer genes, the most frequently mutated residues will have lower ρ values than non-mutated residues in the same protein. This is a genome-wide falsifiable hypothesis.

### Finding 2: Allosteric Effect Size Stratifies Mutation Mechanisms

While λ₂ cannot diagnose pathology, it cleanly separates mutation mechanisms:

- **High Δλ₂ (>15%):** The mutation rewires the allosteric network. Functional consequences are *distal* from the mutation site. Drug targets are the doorway residues, not the mutation itself.
- **Low Δλ₂ (<8%):** The mutation acts locally. Functional consequences are at or near the mutation site. The mutation itself is the relevant pharmacological target.

This distinction has direct therapeutic implications:
- KRAS G12D (+97%): Target the allosteric doorways (Switch II, P-loop wrapping changes)
- TP53 R273H (+5.1%): Target the mutation site directly (restore DNA contact)

## Graph Boundary Rule (Mandatory)

All λ₂ comparisons MUST use the same set of structurally aligned Cα nodes. Violations:
- ❌ Including a bound peptide/partner in one state but not the other
- ❌ Comparing structures with different domain compositions
- ❌ Comparing a monomer to a complex
- ✓ Same chain, same domain, apo vs apo (or both with partner excluded)
- ✓ Same chain extracted from different complexes

The KEAP1 Kelch artifact (-47% → +8.0% when NRF2 excluded) is the cautionary example.

## Remaining Open Questions

1. **Benign missense control:** No crystallographic structure of a known benign TP53/KRAS/SPOP variant exists. Until one is tested, we cannot claim λ₂ has diagnostic specificity for pathogenicity.

2. **ρ genome-wide validation:** The constitutive dehydron hypothesis needs testing across all CGC Tier 1 genes. Does low ρ at WT residues correlate with COSMIC mutation frequency?

3. **Kinase activation interpretation:** CDK2 (-29.2%) measured the kinase domain alone in its activated conformation (cyclin excluded from graph). This is methodologically clean but represents a conformational change, not a point mutation. It cannot be compared to missense mutation data on the same axis.

## Data Summary (All Persisted to Aurora)

| Run | GDP (ref) | GTP (alt) | λ₂ ref | λ₂ alt | Δλ₂ | Type |
|-----|-----------|-----------|--------|--------|------|------|
| KRAS | 4OBE WT | 4DSO G12D | 0.241 | 0.476 | +97% | Missense |
| SPOP | 3IVV WT | 7LIN D140G | 0.519 | 0.614 | +18% | Missense |
| DDX3X | 5E7I WT | 9E2C R326H | 0.048 | 0.207 | +331% | Missense |
| TP53 R273H | 2XWR WT | 4IBS | 0.619 | 0.650 | +5.1% | Missense (contact) |
| TP53 R273C | 2XWR WT | 4IBQ | 0.619 | 0.663 | +7.1% | Missense (contact) |
| TP53 R280K | 2XWR WT | 6FF9 | 0.619 | 0.651 | +5.1% | Missense (contact) |
| KEAP1 BTB | 7EXI WT | 4CXJ C151W | 0.173 | 0.137 | -21% | Missense (sensor) |
| KEAP1 Kelch | 1ZGK apo | 2FLU Kelch-only | 0.848 | 0.916 | +8.0% | Substrate binding |
| CDK2 | 1HCK inactive | 1FIN active | 0.332 | 0.235 | -29% | State transition |
| DAXX | 4H9N | 4H9O | 0.281 | 0.285 | +1% | Null control |

## Method Notes

- All missense comparisons: same protein, same domain, apo vs apo, single chain
- Graph boundary: Cα contact graph (10Å cutoff), weighted by GNN cone depth
- Dehydron threshold: TAU = 13.0
- Pipeline: DTIE Phase 4b conductance spectral analysis (Fiedler eigenvalue)
- GNN: Epistemic filtering only — no learned representations used for classification
