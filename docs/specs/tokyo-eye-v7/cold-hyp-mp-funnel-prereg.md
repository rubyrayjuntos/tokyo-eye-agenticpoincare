# Tokyo Eye v7 — Cold Hyp MP + simple graph + funnel curriculum

**Status:** PARKED (2026-07-22) — child angfill also parked; not trunk  
**Date:** 2026-07-21  
**Lineage / RUN_ID:** `tokyo_eye_v7_cold_hyp_mp_funnel_v1`  
**Machine stamp:** [`data/gates/tokyo_eye_v7_cold_hyp_mp_funnel_prereg.json`](../../../data/gates/tokyo_eye_v7_cold_hyp_mp_funnel_prereg.json)  
**Train:** `make train-v7-cold-hyp-mp-funnel`  
**Does not overwrite:** `HEALTHY_V7_CKPT`, Fix-1 champions, `hyp_biology_mp_v1`

> Compare-only archaeology under `checkpoints/v7/runs/tokyo_eye_v7_cold_hyp_mp_funnel_*/`. Restore SSOT remains sealed healthy.

## Claim

A **from-scratch** Hyp MP trunk, on the **simplest connected graph** (Cα contact), trained primarily on **funnel / disc geometry** (early v7 face), can learn hierarchical hub structure in the ball **without** Euc-MP warmstart and **without** stuffing a rich biology edge ontology.

This is the fair test of the hyperbolic-hubs thesis after the biology-graph child showed Euc-pretrained weights fail on sparse chem edges.

## Locks

| Field | Value |
|-------|--------|
| Init | **Cold** — random by default (`TokyoEye` sealed-era flags, new weights). Optional cutover scaffold via `--use-cutover-scaffold`. **No** Fix-1 / `HEALTHY_V7` weight load |
| Graph | **Simple** — default Cα contact `edge_index` from loader (cutoff as today). No π/salt/dehydron MP ontology |
| Transport | Hyp MP only (`se3_aux=false`) |
| Curriculum | Funnel / disc: cone consistency, disc occupancy / depth-scale, light physics ρ — **B′-style disc coeffs**, not allele/Jacobian |
| `hyp_biology_mp` | **false** |
| Disc hold | `disc_r_mean ≥ 0.25` for best saver; abort after 2 consecutive misses **after** warmup (`--disc-hold-warmup-epochs`, default 6) |

## Non-goals (v1)

- Biology typed MP edges (deferred; separate lineage already probed)
- Warmstart from Fix-1 or sealed healthy
- Claiming biology Pass on migration/basin in v1 — first bar is **disc/funnel health from cold**
- Overwriting sealed Θ

## Pass form (v1)

| Check | Bar |
|-------|-----|
| Train completes without Cα-forbidden path | n/a (Cα graph intentional) |
| `hyp_mp_primary` / `se3_aux=false` | hard |
| Cold init (no Fix-1 keys in resume) | hard |
| Disc hold ≥ 0.25 | hard for best saver |
| Hub migration / basin vs sealed | **report-only** after disc Pass |

## Isolation

| Artifact | Path |
|----------|------|
| Run dir | `checkpoints/v7/runs/tokyo_eye_v7_cold_hyp_mp_funnel_v1/` |
| Best | `v7_cold_hyp_mp_funnel_best.pt` |
| Closeout (later) | `data/gates/tokyo_eye_v7_cold_hyp_mp_funnel_closeout.json` |
