# Disc projection must never stand alone for relational inference

**Date:** 2026-07-17 (escalated same day after flow-influence pilot)
**Status:** Standing rule — three independent experiments; disc alone is forbidden for
pairwise / relational claims

## The rule (one line)

The disc-projection layer (`hyp_projections_2d`) must **never** be used alone for
pairwise or relational inference in this codebase. Every relational claim requires a
**paired trunk** measurement (`encoder_h`). A disc-only result is not evidence; a
disc result that disagrees with or sign-flips the trunk is an indictment of the disc
readout, not a finding about the biology or the edge type under test.

## Why this escalated past "caution"

Three unrelated perturbations now show the same failure mode. The third adds
**sign-flipped** classical correlations — not merely weaker agreement:

| # | Experiment | What the disc did wrong | Artifact |
|---|-----------|-------------------------|----------|
| 1 | Dehydron barcode OOD (KRAS coupled lock) | Disc separation with **flat trunk** (`projection_only`) | `kras_coupled_lock_ood/report.json`; barcode `ablation.md` §"Layer pin" |
| 2 | Chem-MVP disulfide pairs | Trunk **compacts** (1LYZ 4/4) while disc **expands** (3/4) — `trunk_disc_disagree` | `chem_mvp_stage_a12/disulf_pair_probe.json` |
| 3 | Jacobian flow-influence (4OBE pilot) | Trunk↔betweenness **+0.69**; disc↔betweenness **−0.56** (sign flip); disc asymmetry ≈ 0 | `learned_flow_influence/pilot_4obe.json` |

Failing the same way three times, under three different interventions, with the third
actively **inverting** an externally validated hub ranking, is no longer a soft caveat.
Treat disc-alone relational numbers as invalid by default.

## Suspected cause

Projection-step fragility is sufficient: rim-push vs filled-disc recipe effects,
τ-coupled radial targets, and past z-norm/render bugs all act at or after the disc
projection. The map from high-dimensional trunk to 2D is not distance- or
influence-preserving for arbitrary pairs.

## Required practice

1. Measure **trunk** and **disc** separately from the start; never pool.
2. Promotion / scientific claims about **communication / representation change** ride on the **trunk** (`encoder_h`) — never disc alone.
3. **Pathway / hub / leak biology** (Tokyo Eye inference) must be graded **post-lift** (ball / geodesics / cone artifacts), not disc-alone and not Euclidean contact betweenness as the headline — see [`EUCLIDEAN_CONSTRUCTION_HYPERBOLIC_INFERENCE.md`](EUCLIDEAN_CONSTRUCTION_HYPERBOLIC_INFERENCE.md).
4. Disc may be reported as a diagnostic of projection fidelity; it must not be the sole support for a hub, pair-compaction, or coupling claim.
5. If disc and trunk disagree or sign-flip, record `trunk_disc_disagree` / `projection_only`, treat disc as broken for that claim, and do **not** promote the disc-side story.

## Reusable instruments

- `experiments/diagnostics/kras_coupled_lock_ood.py`
- `experiments/diagnostics/chem_mvp_disulf_pair_probe.py`
- `experiments/diagnostics/jacobian_flow_influence.py`

All three hook trunk via `radial_head` / `angular_head` (T1a path) and keep disc as a
paired, non-deciding layer.
