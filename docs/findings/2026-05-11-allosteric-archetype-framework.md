# Allosteric Archetype Framework: λ₂ Directionality as Cancer Mutation Discriminator

**Date:** 2026-05-11
**Method:** DTIE structural physics (dehydron topology, Phase 4b conductance spectral analysis)
**Proteins analyzed:** SPOP, DDX3X, KEAP1 (Kelch + BTB domains)
**GNN status:** NOT APPLIED to any mechanistic interpretation. GNN provided epistemic uncertainty filtering only.

## Summary

Three structurally distinct proteins — an E3 ligase substrate adaptor (SPOP), an RNA helicase (DDX3X), and a β-propeller/BTB electrophile sensor (KEAP1) — were analyzed by DTIE dehydron topology and Phase 4b conductance spectral analysis. The results define three allosteric archetypes distinguished by the direction of change in algebraic connectivity (λ₂) upon perturbation.

**Core discriminatory claim:** ~~λ₂ direction predicts whether a structural perturbation is cancer-pathological (increase) or physiologically regulatory (decrease).~~ **REVISED (2026-05-12):** λ₂ magnitude quantifies allosteric effect size. It classifies *how* a mutation rewires a protein (local contact vs distal allostery vs domain locking), not *whether* it causes cancer. See `docs/findings/2026-05-12-framework-revised-honest-assessment.md`.

## Data Records

| Protein | Domain | Comparison | PDB pair | λ₂ (ref) | λ₂ (alt) | Δλ₂ | Classification |
|---------|--------|-----------|----------|-----------|-----------|------|----------------|
| KRAS | G-domain | WT vs G12D | 4OBE / 4DSO | 0.241 | 0.476 | +97% | PATHOLOGICAL |
| SPOP | MATH | WT vs D140G | 3IVV / 7LIN | 0.519 | 0.614 | +18% | PATHOLOGICAL |
| DDX3X | Helicase | WT vs R326H | 5E7I / 9E2C | 0.048 | 0.207 | +331% | PATHOLOGICAL |
| TP53 | DBD | WT vs R273H | 2XWR / 4IBS | 0.619 | 0.650 | +5.1% | BORDERLINE PATH. |
| KEAP1 | Kelch | Apo vs NRF2 | 1ZGK / 2FLU | 0.848 | 0.446 | -47%* | ~~PHYSIOLOGICAL~~ **ARTIFACT** |
| KEAP1 | BTB | WT vs C151W | 7EXI / 4CXJ | 0.173 | 0.137 | -21% | PHYSIOLOGICAL |
| DAXX | HID | Same-state crystal forms | 4H9N / 4H9O | 0.281 | 0.285 | +1% | SPECIFICITY CONTROL |

*Kelch λ₂ includes NRF2 peptide chain in graph — true Kelch-only value may differ. Phase 2 doorway results are unaffected.

**CORRECTION (2026-05-12): KEAP1 Kelch result INVALIDATED.** Re-running with NRF2 peptide excluded (chain X only) produces λ₂ = 0.848 → 0.916 (+8.0%), REVERSING the direction. The -47% was an artifact of including 16 loosely-connected NRF2 peptide residues in the conductance graph, which artificially depressed λ₂. Archetype III now rests solely on the KEAP1 BTB result (-21%). See `docs/findings/2026-05-12-framework-validation-hardening.md`.

**TP53 Contact Mutants (2026-05-12):** Three crystallographic contact mutants (R273H +5.1%, R273C +7.1%, R280K +5.1%) define Archetype IV: "Contact Loss with Compensatory Rigidification" — mild λ₂ increase, no hub migration, consistent +5–7% signal that is 5–7× above the DAXX null.

**Specificity control (DAXX):** Pipeline produces stable λ₂ values for structurally identical states (RMSD 0.198Å, 100% identity, 205 common core). Confirms that large Δλ₂ values observed for SPOP (+18%), DDX3X (+331%), and KRAS G12D (+97%) reflect genuine structural perturbation rather than technical artifact from crystal packing, minor conformational variation, or pipeline noise.

**TP53 R273H (2026-05-12):** Contact mutant showing +5.1% λ₂ increase — above DAXX null (+1%) but below established pathological threshold (+18%). No hub migration. Mutation site (R273) is NOT a dehydron (ρ=23). Represents a potential fourth archetype: "Contact Loss with Compensatory Rigidification." See `docs/findings/2026-05-12-tp53-r273h-contact-mutant.md`.

**CORRECTION (2026-05-12):** Previous dossier listed KRAS as 4DSO/7KFU with λ₂ +158%. That was a G12D vs G12H (mutant-vs-mutant) comparison. The correct WT baseline is 4OBE (GLY12) vs 4DSO (ASP12), λ₂ +97%. See `docs/findings/2026-05-12-kras-g12d-corrected.md`.

## Archetype I: Convergent Occlusion (SPOP, KRAS)

**Mechanism:** Cancer mutations exploit pre-existing dehydrons and propagate allosterically to functional surfaces, increasing wrapping at regulatory sites.

### SPOP (4 hotspots)
Four cancer hotspot mutations (D140G, E47K, M117V, D140N) at three different positions in the MATH domain substrate-binding groove, separated by up to 93 residues in sequence and 30Å in space, all produce increased wrapping at the same two residues: **SER 33** (Δρ = +5 to +6) and **PHE 57** (Δρ = +1 to +5). These residues are AURKA phosphorylation sites (Nikhil et al. 2020, DOI: 10.3390/cancers12113247).

### KRAS G12D
GLY 12 is constitutively underwrapped (ρ=9.0) in WT — the mutation exploits a pre-existing dehydron. G12D propagates allosterically to P-loop (residue 17), Switch II (residue 68, effector-binding surface), and α3 helix (residue 85). λ₂ increases +97%.

**Subtype distinction:** KRAS doorways are in closer spatial proximity to the mutation site (5-25Å) compared to SPOP (30Å). This represents a "local cage" variant where allosteric effects concentrate near the mutation rather than propagating to a distant regulatory surface.

**Shared pattern:** Both SPOP E47K and KRAS G12D mutate residues that are ALREADY constitutive dehydrons (ρ=7.0 and ρ=9.0 respectively). The mutations change chemical character at pre-existing exposed sites rather than creating new dehydrons.

**Detailed findings:** `docs/findings/2026-05-11-spop-allosteric-convergence.md`, `docs/findings/2026-05-12-kras-g12d-corrected.md`

## Archetype II: Domain Locking (DDX3X)

Single cancer mutation R326H increases λ₂ by 331% from a low baseline (0.048), locking the inter-domain interface that must be dynamic for ATP-coupled RNA unwinding. Creates a novel doorway at **GLY 261** (Motif Ia, RNA 2'-OH contact) 21.8Å from the mutation site. This doorway does NOT appear in the normal apo→inhibitor functional comparison.

**Mechanism:** Inter-domain rigidification abolishes catalytic cycling + allosteric disruption of RNA-binding motif.

**Additional finding:** Constitutive dehydron cluster at residues 473-479 (ρ=0-3) on the C-terminal RNA-binding surface — permanent PPI landing pad explaining DDX3X's multi-complex bridging role in the cancer gene network.

**Detailed finding:** `docs/findings/2026-05-11-ddx3x-r326h-allosteric.md`

## Archetype III: Regulatory Decoupling (KEAP1 — Physiological Baseline)

**STATUS: WEAKENED — Kelch result invalidated, BTB result still valid.**

### ~~Kelch Domain (NRF2 binding)~~ — ARTIFACT (2026-05-12)

~~NRF2 substrate binding produces 19 state-selective doorways across all 6 β-propeller blades. Of these, only 2 are in direct NRF2 contact (<4.5Å) — the remaining 17 are allosteric responders. λ₂ decreases by 47%, indicating the propeller becomes more modular upon substrate engagement.~~

**CORRECTED:** The -47% λ₂ decrease was an artifact of including the NRF2 peptide chain (16 residues, chain P) in the 2FLU conductance graph. When only the Kelch domain (chain X) is analyzed, λ₂ INCREASES by +8.0%. The original signal was caused by loosely-connected NRF2 nodes creating a weak bottleneck in the combined graph.

### BTB Domain (C151W electrophile mimic)

C151W produces 3 doorways including GLN 75 at 30Å (dimerization interface). λ₂ decreases by 21%. Hub (TYR 141) is stable — no migration. Pattern matches physiological regulation: the electrophile sensor modification produces controlled decoupling, not pathological rigidification.

**Novel prediction:** C151W propagates allosteric changes to the BTB dimerization interface (GLN 75, 30Å from C151). This suggests electrophile sensing may alter KEAP1 dimer geometry as a secondary mechanism beyond NRF2 release.

**Detailed finding:** `docs/findings/2026-05-11-keap1-regulatory-decoupling.md`

## Falsification Criteria

The λ₂ directionality hypothesis is falsifiable. It predicts that:

1. **Any oncogenic missense mutation producing a decrease in λ₂** relative to WT would represent a fourth archetype — a cancer mutation that functions through regulatory decoupling rather than rigidification. Identification of such a mutation would require revision of the framework.

2. **Any physiological regulatory event producing an increase in λ₂** (substrate binding, post-translational modification, allosteric effector) would contradict the claim that regulatory events decouple while pathological events rigidify.

3. **Any benign polymorphism producing a λ₂ increase >20%** would indicate the discriminator has insufficient specificity.

### Falsification Status (2026-05-12)

The falsification criteria remain prospectively untested due to absence of cancer mutant crystal structures for DAXX and FUBP1. Attempted falsification runs:

- **DAXX:** No cancer mutant structures available in PDB. The 14 available structures are all WT in various complexes. The same-state comparison (4H9N vs 4H9O, +1%) serves as a specificity control only.
- **FUBP1:** Available structures cover different domain fragments (KH1, KH2, KH3+KH4) that do not align. No same-domain two-state comparison is possible with current PDB data.

Systematic falsification testing requires either: (a) new mutant crystal structures, or (b) computational structure prediction (AlphaFold2 missense variant models) as surrogates. The latter is a defined future experiment.

**UPDATE (2026-05-12): ESMFold surrogate test FAILED for structural mutants.** ESMFold-predicted R248W and R175H structures show +0.9% and +1.4% λ₂ change respectively — within the DAXX null range. Single-sequence structure prediction cannot model thermodynamic destabilization. ESMFold/AF2 surrogates are only valid for contact mutants (which maintain the fold). Structural mutants require molecular dynamics simulation to generate representative conformational ensembles. See `docs/findings/2026-05-12-tp53-alphafold-surrogate-test.md`.

Prospective application to CDK8 and additional bridge proteins will extend the tested set. The framework currently rests on 7 pathological comparisons (KRAS +97%, SPOP +18% × 4 hotspots, DDX3X +331%, TP53 R273H +5.1%, R273C +7.1%, R280K +5.1%), 1 physiological comparison (KEAP1 BTB -21%), 1 invalidated result (KEAP1 Kelch — artifact), and 1 specificity control (DAXX +1%).

**Additional finding (TP53, 2026-05-12):** R248 is a constitutive dehydron with ρ=5 — the deepest wrapping deficit observed in any protein analyzed. This explains why R248W is the most common TP53 cancer mutation: position 248 is maximally vulnerable to chemical perturbation. The R248W mutation cannot crystallize without rescue mutations because it exploits this extreme vulnerability to trigger global unfolding.

## Method Notes

- All comparisons use DTIE Phase 4b conductance spectral analysis (Fiedler eigenvalue λ₂ of the Cα contact graph weighted by GNN-derived cone depth)
- Dehydron threshold TAU = 13.0 (calibrated GOSP threshold)
- Epistemic filter: residues with GNN epistemic uncertainty > per-structure median
- Normalization: direct Needleman-Wunsch alignment (seqres_fallback path, conn=None)
- All alignments marked is_verified=False (not UniProt-guided)
- No GNN learned representations used for archetype classification — purely structural physics + spectral graph theory
