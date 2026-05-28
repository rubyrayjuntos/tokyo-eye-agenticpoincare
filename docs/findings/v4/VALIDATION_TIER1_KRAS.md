# Tier 1 Validation: KRAS WT (4OBE) vs G12D (4DSO)

**Date:** 2026-05-20
**Pipeline version:** DTIE v3.0 with v4 GNN bridge
**Checkpoints tested:** Both (pipeline epoch 75, scientific epoch 35)

---

## Checkpoint-Independent Findings (Primary Scientific Claims)

| Finding | Source | Checkpoint 2 (pipeline) | Checkpoint 1 (scientific) | Status |
|---------|--------|------------------------|--------------------------|--------|
| λ₂ GDP < λ₂ GTP | Phase 4b | 0.056 | 0.054 | ✓ Identical |
| Top GDP hub: α5 C-terminal | Phase 4b | B:152 | B:151 | ✓ Same region |
| Doorway at res 57 (Switch-II start) | Phase 2 | ✓ | ✓ | ✓ Both |
| Doorway at res 101 (α3-helix relay) | Phase 2 | ✓ | ✓ | ✓ Both |
| Doorway at res 46 | Phase 2 | ✓ | ✓ | ✓ Both |
| Doorway at res 169 (C-terminal) | Phase 2 | ✓ | ✓ | ✓ Both |
| Doorway at res 4 (N-terminal) | Phase 2 | ✓ | ✓ | ✓ Both |
| Phase 3: 0 terminal leaks | Phase 3 | 0 | 0 | ✓ Expected |

## Checkpoint-Dependent Findings (Hypothesis-Generating)

| Finding | Source | Checkpoint 2 | Checkpoint 1 | Note |
|---------|--------|-------------|-------------|------|
| Doorway count | Phase 2 | 22 state-selective | 14 state-selective | Uncertainty-gated |
| Constitutive count | Phase 2 | 12 | 20 | Inverse of above |
| Top GTP hub | Phase 4b | A:6 | A:78 | Angular reorganization effect |

---

## Interpretation

### λ₂ Relationship (checkpoint-independent)

GDP λ₂=0.054–0.056, GTP λ₂=0.094–0.100. GTP state is more strongly connected.
This is correct: GTP-bound KRAS is in the active conformation with Switch-I and
Switch-II ordered and engaged, creating stronger allosteric conductance. GDP-bound
is the inactive state with disordered switches. Both checkpoints produce nearly
identical values — this comes from Phase 4b's structural physics (conductance
graph from Cα distances + cone_depth weighting), not the GNN representation.

### Top GDP Hub: α5 C-terminal (checkpoint-independent)

Residue 151-152 is the α5-helix C-terminal region — the membrane-proximal end
of KRAS that coordinates with the G-domain for nanoclustering. Independently
validated by the Probe 3 graph analysis finding that C-terminal 168-169 gained
30 hyperbolic neighbors in G12D. Two different pipeline phases pointing at the
same region.

### Consistent Doorways (checkpoint-independent)

Residues 4, 46, 57, 101, 169 are doorways in both checkpoints. They survived
architectural changes, different training curricula, and different geometric
representations.

- **Residue 57:** Start of Switch-II — the GAP interaction region
- **Residue 101:** α3-helix — known allosteric communication relay between
  Switch-II and the hydrophobic core
- **Residue 169:** C-terminal HVR — membrane-proximal, nanoclustering

### Phase 3: 0 Terminal Leaks (expected)

The arc geometry means H1 topology requires closed loops that don't form in a
single KRAS structure at this filtration scale. This is correct for a well-folded
globular protein. Document as expected, not as failure.

### Doorway Count Divergence (checkpoint-dependent)

The doorway classification is uncertainty-gated. Checkpoint 1 has higher epistemic
uncertainty overall (trained less aggressively), so more sites exceed the epistemic
threshold and get flagged as constitutive. Checkpoint 2's domain separation training
created more confident predictions on specific regions, promoting sites from
constitutive to state-selective.

The underlying ρ values are identical (same PDB structures). The physics is
checkpoint-independent. The doorway set is checkpoint-conditional.

---

## Summary Table for Write-Up

| Finding | Source | Checkpoint-independent? |
|---------|--------|------------------------|
| λ₂ GDP < λ₂ GTP | Phase 4b | Yes — identical both |
| Top GDP hub: α5 C-terminal | Phase 4b | Yes — 151/152 both |
| Doorways at 46, 57, 101, 169 | Phase 2 | Yes — both checkpoints |
| Doorway count (22 vs 14) | Phase 2 | No — uncertainty-gated |
| Top GTP hub (6 vs 78) | Phase 4b | No — checkpoint-dependent |

Checkpoint-independent findings are primary scientific claims.
Checkpoint-dependent findings are labeled hypothesis-generating pending further validation.

---

## Run Details

```
Run 1 (pipeline checkpoint):
  run_id: 0f554fc5-4d79-4191-825a-6947a794dc17
  checkpoint: checkpoints_v4_retrain/checkpoint_stage_2b.pt (epoch 75)
  runtime: 124.1s

Run 2 (scientific checkpoint):
  run_id: 12faa26d-a6c1-4490-8369-7bff31801c58
  checkpoint: checkpoints_v4_cone_fix2/checkpoint_stage_1.pt (epoch 35)
  runtime: 145.2s