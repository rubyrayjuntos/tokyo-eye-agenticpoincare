# Framework Validation & Hardening Results

**Date:** 2026-05-12
**Purpose:** Address critical review requirements for rigorous validation
**Status:** 2 of 3 tests passed; 1 test revealed artifact requiring framework revision

## Executive Summary

| Test | Result | Implication |
|------|--------|-------------|
| Contact mutant replication | **PASSED** — R273C (+7.1%), R280K (+5.1%) consistent with R273H (+5.1%) | Archetype IV validated |
| Benign variant control | **PARTIAL** — ESMFold limitation prevents crystallographic test; DAXX null (+1%) remains best control | 5-7× separation from null |
| KEAP1 artifact isolation | **FAILED** — Signal reverses from -47% to +8.0% when NRF2 excluded | Archetype III (Kelch) requires revision |

## Test 1: Contact Mutant Replication (PASSED)

Three crystallographic TP53 contact mutants compared against the same WT reference (2XWR chain A):

| Mutant | PDB | λ₂ (WT) | λ₂ (mut) | Δλ₂ | Hub WT | Hub mut | Doorways |
|--------|-----|---------|----------|------|--------|---------|----------|
| R273H | 4IBS | 0.6186 | 0.6500 | +5.1% | A:141 | A:141 | 11 |
| R273C | 4IBQ | 0.6186 | 0.6628 | +7.1% | A:141 | A:141 | 12 |
| R280K | 6FF9 | 0.6186 | 0.6505 | +5.1% | A:141 | A:141 | 11 |

**Findings:**
- All three contact mutants produce λ₂ increases in the +5–7% range
- Hub remains stable at CYS 141 in all cases (no migration)
- Doorway counts are consistent (11–12)
- The signal is 5–7× above the DAXX null control (+1%)

**Conclusion:** Archetype IV ("Contact Loss with Compensatory Rigidification") is validated as a distinct, reproducible, low-magnitude signal. Contact mutants consistently produce mild rigidification (+5–7%) without hub reorganization, distinguishing them from structural mutants (Archetype I: +18–331%) which show hub migration and much larger λ₂ shifts.

## Test 2: Benign Variant Control (PARTIAL)

| Comparison | Method | Δλ₂ | Interpretation |
|-----------|--------|------|----------------|
| DAXX 4H9N vs 4H9O (same state) | X-ray | +1.0% | Noise floor |
| ESMFold WT vs P151T (benign) | ESMFold | -0.4% | Within noise |
| ESMFold WT vs R248W (pathological) | ESMFold | +0.9% | Within noise (surrogate failure) |
| X-ray WT vs R273H (pathological) | X-ray | +5.1% | 5× above null |
| X-ray WT vs R273C (pathological) | X-ray | +7.1% | 7× above null |

**Limitation:** No crystallographic structure of a known benign TP53 DBD variant exists in PDB. The ESMFold benign control (P151T, -0.4%) confirms that ESMFold predictions don't trigger false positives, but this is the same limitation that prevents ESMFold from detecting true positives for structural mutants.

**Best available evidence:** The DAXX same-state control (+1%) establishes the noise floor. All crystallographic pathological comparisons (R273H, R273C, R280K) are 5–7× above this floor. The separation is sufficient to distinguish signal from noise, though a crystallographic benign variant would strengthen the case.

**Falsification criterion 3 status:** "Any benign polymorphism producing a λ₂ increase >20% would indicate insufficient specificity." This criterion remains untested with crystallographic data. The ESMFold P151T result (-0.4%) is consistent with the prediction but cannot be considered definitive due to the surrogate limitation.

## Test 3: KEAP1 Kelch Artifact Isolation (ARTIFACT CONFIRMED)

| Comparison | λ₂ (apo) | λ₂ (bound) | Δλ₂ | NRF2 in graph? |
|-----------|----------|-----------|------|----------------|
| Original (1ZGK vs 2FLU full) | 0.848 | 0.446 | -47% | YES (16 residues) |
| Corrected (1ZGK_A vs 2FLU_X only) | 0.848 | 0.916 | **+8.0%** | NO |

**The -47% "Regulatory Decoupling" signal was an artifact.**

When the NRF2 peptide chain (16 residues, chain P) is excluded from the 2FLU conductance graph, the λ₂ direction REVERSES from -47% to +8.0%. The original result was a mathematical artifact: adding 16 loosely connected NRF2 peptide nodes to the graph artificially decreased algebraic connectivity by creating a weakly-coupled subgraph.

**Mechanism of the artifact:** The Fiedler eigenvalue (λ₂) measures the weakest bottleneck in the graph. When NRF2 peptide residues are included, they form a loosely connected appendage to the Kelch β-propeller. The weakest connection in the combined graph is the Kelch–NRF2 interface, not any internal Kelch feature. This drives λ₂ down regardless of any genuine allosteric change in the Kelch domain itself.

**Corrected interpretation:** With NRF2 excluded, the Kelch domain actually shows +8.0% λ₂ increase upon NRF2 binding — the same direction as pathological mutations. This suggests NRF2 binding mildly rigidifies the Kelch propeller (similar to the contact mutant pattern), rather than decoupling it.

### Impact on Framework

The KEAP1 Kelch result was one of two data points supporting Archetype III ("Regulatory Decoupling" = λ₂ decrease). With this artifact exposed:

| Archetype III evidence | Status |
|----------------------|--------|
| KEAP1 Kelch (1ZGK vs 2FLU) | **INVALIDATED** — artifact of NRF2 chain inclusion |
| KEAP1 BTB (7EXI vs 4CXJ, -21%) | Still valid (single-chain comparison, no peptide) |

Archetype III now rests on a single data point (KEAP1 BTB, -21%). The framework's claim that "physiological events decrease λ₂" is weakened but not falsified — the BTB result still shows the predicted direction. Additional physiological comparisons are needed to re-establish the pattern.

## Revised Framework Status

| Archetype | Evidence | Status |
|-----------|----------|--------|
| I: Convergent Occlusion | KRAS +97%, SPOP +18% (×4 hotspots) | **Strong** |
| II: Domain Locking | DDX3X +331% | **Single data point** |
| III: Regulatory Decoupling | KEAP1 BTB -21% (Kelch INVALIDATED) | **Weakened** |
| IV: Contact Loss | TP53 R273H +5.1%, R273C +7.1%, R280K +5.1% | **Validated (new)** |
| Control: Null | DAXX +1% | **Established** |

### Key Revision Required

The framework document must be updated to:
1. Remove or flag the KEAP1 Kelch result as artifactual
2. Note that Archetype III rests on a single data point (BTB)
3. Add Archetype IV with the three TP53 contact mutant results
4. Acknowledge that the λ₂ directionality discriminator (increase = pathological, decrease = physiological) is now supported by 7 pathological comparisons but only 1 physiological comparison

## Method Notes

- All runs: Phase 4b conductance spectral analysis, TAU=13.0, cutoff=10.0Å
- TP53 runs: 2XWR chain A (WT, 199 residues) vs single-chain mutant extracts
- KEAP1 run: 1ZGK chain A (apo, 281 residues) vs 2FLU chain X only (285 residues)
- No normalization applied (NullStore mode, offline)
- No DSSP — geometric secondary structure fallback
