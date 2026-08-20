# Tokyo Eye v7 — Hyp biology MP (child lineage design)

**Status:** LOCKED — v1 implement landed (audit + extract + fail-closed + smoke); train deferred  
**Date:** 2026-07-21  
**Prereg:** [`hyp-biology-mp-prereg.md`](hyp-biology-mp-prereg.md) · [`data/gates/tokyo_eye_v7_hyp_biology_mp_prereg.json`](../../../data/gates/tokyo_eye_v7_hyp_biology_mp_prereg.json)

## Problem

Sealed Hyp MP can still talk on Cα contact topology (`resolve_hyp_mp_edges` fallback). Biology and structure then merge without a hard contract.

## Solution

Child lineage `tokyo_eye_v7_hyp_biology_mp_v1`:

1. Build **biology-only** `hyperbolic_edge_index` from H-bond, dehydron, π-stack, salt bridge.
2. Fail if Hyp MP would use Cα / packing Euc / SE(3) rel edges.
3. Degree-0 nodes stay self-only (Option A) for clean v1 ablation.
4. Never write sealed `HEALTHY_V7_CKPT`.

## Components (landed)

| Module | Responsibility |
|--------|----------------|
| `science/tokyo_eye/biology_graph.py` | Extract salt / π / hbond / dehydron; `attach_biology_mp_graph` |
| `science/tokyo_eye/biology_mp_audit.py` | Ontology counters; `ca_in_mp=false` assert |
| `resolve_hyp_mp_edges` | Fail-closed when `hyp_biology_mp` / `allow_ca_fallback=false` |
| `TokyoEye.hyp_biology_mp` | Audit trail + resolve kwargs |
| `make grade-v7-hyp-biology-mp-smoke` | Sibling RUN_ID smoke only |

## Isolation

- Flag `hyp_biology_mp` off by default on sealed path.
- Checkpoints only under `checkpoints/v7/runs/tokyo_eye_v7_hyp_biology_mp_v1/`.
- v2 may add Option C (`hyp_proximity`) without reopening Cα MP.

## Out of scope (v1)

Virtual CoM anchor, solvation-shell edges, biology Pass bars beyond audit smoke.
