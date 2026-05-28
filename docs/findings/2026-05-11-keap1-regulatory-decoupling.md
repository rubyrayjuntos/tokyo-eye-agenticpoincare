# KEAP1 Regulatory Decoupling Finding

**Date:** 2026-05-11
**Method:** DTIE structural physics — dehydron analysis + Phase 2 + Phase 4b conductance
**GNN status:** NOT APPLIED to interpretation. GNN provided epistemic filtering only.

## Purpose: Physiological Baseline for Cancer Mutation Framework

KEAP1 serves as the regulatory baseline against which SPOP and DDX3X cancer mutations are calibrated. The KEAP1 Kelch + NRF2 comparison shows what NORMAL allosteric behavior looks like (substrate binding → regulatory decoupling), establishing that cancer mutations (SPOP, DDX3X) produce the OPPOSITE pattern (aberrant rigidification).

## Kelch Domain: Apo vs NRF2-Bound

**Structures:** 1ZGK (apo, 1.35Å, chain A) vs 2FLU (NRF2-bound, 1.50Å, chain X + NRF2 peptide chain P)

| Metric | Value |
|--------|-------|
| Normalization | RMSD=0.47Å, 99.64% identity, 278 common core — **excellent** |
| Doorways | 19 state-selective (apo-exposed, NRF2-wrapped) |
| Constitutive | 113 dehydrons |
| λ₂ | 0.848 → 0.446 (**-47%**) |
| Hub shift | ALA 510 → GLY 462 |

**CAVEAT:** λ₂ result includes NRF2 peptide (chain P, 16 residues) in the GTP graph. The -47% decrease may be partly artifactual from adding loosely-connected NRF2 nodes. Phase 2 doorway results are clean (normalization excluded chain P).

### NRF2 Contact Analysis

Of 19 state-selective doorways:
- **2 in direct NRF2 contact** (<4.5Å): residues 364, 414
- **17 are allosteric responders** (>4.5Å from NRF2): residues 342, 391, 410, 411, 412, 439, 463, 498, 500, 506, 507, 532, 549, 554, 562, 578, 579

The β-propeller transmits the NRF2 binding event across all 6 blades — wrapping changes propagate far beyond the direct contact site. This is the normal allosteric response of a β-propeller to substrate engagement.

### 410-414 Cluster

4 consecutive residues (410, 411, 412, 414) form a doorway cluster. Only residue 414 is in direct NRF2 contact — the other three are allosteric responders to the adjacent contact. This demonstrates backbone wrapping propagation through local secondary structure.

### Constitutive Dehydron Landscape

113 constitutive dehydrons (40% of all residues) — the β-propeller is massively underwrapped. Most exposed: residues 381, 544 (ρ=0.0). Cluster at 541-544 (4 residues, ρ=0-1) is a permanent PPI landing pad.

This is consistent with the Kelch domain's biological role as a multi-partner PPI hub — it needs a large underwrapped surface to accommodate diverse binding partners (NRF2, p62, PGAM5, etc.).

## BTB Domain: WT vs C151W (Electrophile Sensor)

**Structures:** 7EXI (WT apo, 1.82Å, chain A) vs 4CXJ (C151W, 2.80Å, chain A)

| Metric | Value |
|--------|-------|
| Normalization | RMSD=0.45Å, 99.23% identity, 130 common core — **excellent** |
| Doorways | 3 state-selective |
| Constitutive | 53 |
| λ₂ | 0.173 → 0.137 (**-21%**) |
| Hub | TYR 141 (unchanged) |

### BTB Doorways

| Residue | Identity | ρ (WT) | Distance from C151 | Structural Role |
|---------|----------|--------|-------------------|-----------------|
| GLN 75 | Gln | 12.0 | 30.1Å | N-terminal, dimerization region |
| SER 124 | Ser | 11.0 | 22.4Å | Near C151 |
| ILE 125 | Ile | 11.0 | 18.8Å | Near C151 |

### GLN 75: Allosteric Propagation to Dimerization Interface

GLN 75 is 30Å from C151 and in the N-terminal region that mediates KEAP1 homodimerization. Both crystal structures (7EXI, 4CXJ) are monomers, but the biological BTB domain is a homodimer.

**Novel structural prediction:** C151W (electrophile mimic) propagates allosteric changes 30Å to the dimerization interface. This suggests electrophile sensing at C151 may alter KEAP1 dimer geometry as a secondary mechanism beyond NRF2 release. Not previously described in the C151W literature.

### C151W Behaves as a Regulatory Event, Not Cancer Disruption

- λ₂ decreases (-21%) — same direction as NRF2 binding to Kelch (-47%)
- Hub is stable (TYR 141 in both states) — no hub migration
- Pattern matches physiological regulation, not pathological rigidification

C151 is the electrophile sensor — its modification is a NORMAL regulatory signal. C151W mimics the electrophile-modified state. The DTIE signature correctly classifies it as regulatory rather than pathological.

## Three-Archetype Framework (Complete)

| Archetype | Protein | Comparison | λ₂ Δ | Mechanism | Classification |
|-----------|---------|-----------|------|-----------|----------------|
| **Convergent occlusion** | SPOP | 4 hotspots vs WT | +18% | Multi-mutation → same distal wrapping | PATHOLOGICAL |
| **Domain locking** | DDX3X | R326H vs WT | +331% | Single mutation → global rigidification | PATHOLOGICAL |
| **Regulatory decoupling** | KEAP1 Kelch | NRF2-bound vs apo | -47%* | Substrate binding → modular relaxation | PHYSIOLOGICAL |
| **Regulatory decoupling** | KEAP1 BTB | C151W vs WT | -21% | Sensor modification → local relaxation | PHYSIOLOGICAL |

*Kelch λ₂ includes NRF2 peptide in graph — true Kelch-only value may differ.

### Discriminatory Claim

**λ₂ direction predicts whether a perturbation is cancer-pathological (increase) or physiologically regulatory (decrease).** This is a testable, falsifiable claim applicable prospectively to any protein with structural data for two states.

### Falsification Criteria

The framework would be falsified by:
1. A known cancer mutation that decreases λ₂
2. A known physiological regulatory event that increases λ₂
3. A benign polymorphism that increases λ₂ by >20%

## Pipeline Parameters

All runs: pipeline_mode=source_leak_v4, n_landmarks=200, TAU=13.0, epistemic filter > median, GNN checkpoint: robust_experts.pt


## Pipeline Run IDs

- Kelch apo vs NRF2 (1ZGK vs 2FLU): 390092fc-9b20-478d-b396-b20d6c8c2717 (final run)
- BTB WT vs C151W (7EXI vs 4CXJ): e8b115b8-1826-46f6-955a-309a91733b45 (final run)
