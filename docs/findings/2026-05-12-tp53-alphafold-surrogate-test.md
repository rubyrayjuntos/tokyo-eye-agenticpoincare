# TP53 AlphaFold2/ESMFold Surrogate Test: Negative Result

**Date:** 2026-05-12
**Method:** DTIE Phase 4b on ESMFold-predicted structures
**Structures:** ESMFold WT vs ESMFold R248W, ESMFold WT vs ESMFold R175H
**Result:** NEGATIVE — surrogates unsuitable for structural mutants

## Summary

ESMFold-predicted structures for TP53 R248W and R175H show negligible λ₂ changes (+0.9% and +1.4% respectively), both within the DAXX null control range (+1%). This is NOT a falsification of the framework — it is a demonstration that single-sequence structure prediction methods cannot model the conformational consequences of destabilizing mutations.

## Results

| Comparison | λ₂ (WT) | λ₂ (mutant) | Δλ₂ | Hub | Doorways |
|-----------|---------|-------------|------|-----|----------|
| ESM WT vs ESM R248W | 0.584 | 0.589 | +0.9% | A:48 (stable) | 1 |
| ESM WT vs ESM R175H | 0.584 | 0.593 | +1.4% | A:48 (stable) | 2 |
| Crystal WT vs Crystal R273H | 0.619 | 0.650 | +5.1% | A:141 (stable) | 11 |
| DAXX null control | 0.281 | 0.285 | +1.0% | — | — |

## Structural Analysis

| Metric | ESM WT vs R248W | ESM WT vs R175H |
|--------|----------------|----------------|
| Cα RMSD | 0.878Å | 0.926Å |
| Max displacement | 1.37Å (pos 116) | 3.69Å (pos 90) |
| Mean displacement | 0.822Å | 0.806Å |
| Residues >1Å | 62/200 | 50/200 |

The ESMFold predictions produce structures that are nearly identical to WT. The RMSD values (~0.9Å) are comparable to crystal packing differences between identical structures (DAXX: 0.198Å RMSD). While there IS some structural variation, it is distributed uniformly rather than concentrated at functional sites — hence no differential doorway signal.

## Why This Failed

ESMFold (and AlphaFold2) predict the **ground-state fold** — the single most likely 3D structure for a given sequence. They do NOT model:

1. **Thermodynamic destabilization** — R248W and R175H have ΔΔG > 3 kcal/mol. In reality, these proteins exist as partially unfolded ensembles, not well-defined single structures.

2. **Conformational heterogeneity** — The actual R248W/R175H mutant proteins sample multiple conformational states including locally unfolded intermediates. A single predicted structure cannot represent this.

3. **Kinetic trapping** — Crystal structures of rescue mutants (with stabilizing background mutations) capture a specific folded state that the naked mutant cannot maintain at physiological temperature.

The DTIE pipeline is working correctly — it reports minimal λ₂ change because the input structures ARE nearly identical. The pipeline detects what's there; it cannot detect destabilization that isn't represented in the input coordinates.

## Comparison with Crystallographic Result

The crystallographic R273H result (+5.1%, 11 doorways) is valid because:
- R273H is a **contact mutant** — it maintains the fold
- The crystal structure captures the **actual mutant conformation**
- The structural difference is real (different crystal, different lab, different year)
- The doorway pattern is biologically interpretable (proximal to DNA-binding surface)

## Implications for the Framework

1. **AF2/ESMFold surrogates are NOT suitable** for testing structural mutants (R248W, R175H, Y220C, etc.)
2. **Contact mutants** (R273H, R273C, R280K) remain testable via crystallography
3. **Molecular dynamics** would be the appropriate surrogate for structural mutants — MD can sample the partially unfolded ensemble
4. **AlphaFold2 Missense** pathogenicity scores (AlphaMissense) predict R248W and R175H as pathogenic, but the 3D structure predictions do not capture the mechanism

## Revised Falsification Strategy

The original proposal to use AF2 models as surrogates for uncrystallizable mutants is invalidated for structural mutants. Alternative approaches:

| Approach | Suitable for | Limitation |
|----------|-------------|-----------|
| Crystal structures | Contact mutants (R273H, R273C) | Limited availability |
| ESMFold/AF2 | Contact mutants only | Cannot model destabilization |
| MD simulation (100ns+) | Structural mutants | Computationally expensive |
| Rescue mutant crystals | Structural mutants (with caveats) | Background mutations confound |
| Cryo-EM | Large complexes | Resolution limits |

**Recommended next steps:**
1. Run DTIE on R273C (4IBQ) — second contact mutant, crystallographic
2. Run DTIE on R280K (6FF9) — third contact mutant, crystallographic
3. If MD infrastructure becomes available, simulate R248W (100ns) and extract representative conformations for DTIE

## Method Notes

- ESMFold API: `https://api.esmatlas.com/foldSequence/v1/pdb/`
- Prediction time: 0.6–25.8s per 200-residue sequence
- DBD sequence: residues 94–293 of UniProt P04637
- R248W: position 155 in DBD sequence (R→W)
- R175H: position 82 in DBD sequence (R→H)
- All predictions single-chain, no templates, no MSA
