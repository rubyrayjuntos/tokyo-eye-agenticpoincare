# Hyp biology MP — three-arm ablation (pre-registration)

**Status:** LOCKED before grade  
**Date:** 2026-07-21  
**Stamp:** [`data/gates/tokyo_eye_v7_hyp_biology_mp_three_arm_prereg.json`](../../../data/gates/tokyo_eye_v7_hyp_biology_mp_three_arm_prereg.json)  
**Make:** `make grade-v7-hyp-biology-mp-three-arm`

## Claim

On a **fixed** biology graph (hbond / dehydron / π / salt), does Hyp MP add non-tautological value vs classical graph scores — and does 8-ep train beat sealed weights on that same graph?

## Arms (same PDBs, same edge ontology)

| Arm | Model | Graph |
|-----|--------|--------|
| `classical` | No GNN — betweenness hubs; ρ/τ vector on \(R_\star\) for basin; betweenness ratio for 163 migration | biology |
| `sealed_biology` | `HEALTHY_V7_CKPT` + `hyp_biology_mp` | biology |
| `hyp_mp_trained` | `v7_hyp_biology_mp_best.pt` + `hyp_biology_mp` | biology |

## Pass form

| Check | Pass |
|-------|------|
| Contract | all GNN arms `ca_in_mp=false` |
| Basin bar | gap > 0.10 (per arm, report) |
| Migration bar | \(R_{4DSO}>R_{4OBE}\) (per arm) |
| Non-tautology | best GNN basin gap > classical **or** best GNN ΔR > classical ΔR |
| Train vs wiring | report `hyp_mp_trained` vs `sealed_biology` |

## Non-goals

Cα MP arms; Option B/C connectivity; claiming product promote.
