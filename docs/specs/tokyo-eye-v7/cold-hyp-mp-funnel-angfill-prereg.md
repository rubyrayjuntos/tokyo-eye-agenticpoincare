# Tokyo Eye v7 — Cold Hyp MP funnel angular-fill + specialization continue

**Status:** PARKED (2026-07-22) — not usable as trunk; disc ≠ sealed dehydron-rim ontology; soft MoE still under-committed  
**Parent:** [`cold-hyp-mp-funnel-prereg.md`](cold-hyp-mp-funnel-prereg.md)  
**Lineage / RUN_ID:** `tokyo_eye_v7_cold_hyp_mp_funnel_angfill_v1`  
**Resume:** `checkpoints/v7/runs/tokyo_eye_v7_cold_hyp_mp_funnel_v1/v7_cold_hyp_mp_funnel_best.pt`  
**Does not overwrite:** `HEALTHY_V7_CKPT`, cold v1 best (sibling run dir only)

> **Park note:** Engineering probes useful (cold disc Pass; angfill ER>1.5 + hard monopoly cracked; scorecard beats sealed on basin/migration). **Do not promote** — Poincaré layout does not recover sealed dehydron=rim / wrapped=core semantics; soft routing remains mushy. Sealed `HEALTHY_V7_CKPT` remains restore SSOT.

## Claim

Continuing cold Hyp MP (Cα graph) with **sealed feeler parity** (`rim_fanout_forward` + `geometric_angular_prior`) and **light in-structure MoE anti-monopoly** can break soft-mush / E1 hard monopoly and lift `disc_effective_rank_mean > 1.5` without Euc warmstart.

## Graph vs physics (clarification)

| Layer | What cold uses |
|-------|----------------|
| **MP edges** | Simple **Cα contact** adjacency (Euclidean neighbor definition → Hyp MP transport). Not typed π/salt/dehydron edges. |
| **Node / gate physics** | **ρ, τ, SS (H/E/C), degree, clustering, cone depth, SASA** still attached and fed to the gate (`topology_only_gate`). |
| **Losses** | Funnel still supervises ρ / cone / disc; plus new ER + specialization terms. |

So: **no ρ/τ as edge ontology**, not “no ρ/τ at all.” Disc J is consistent with radial funnel pressure + collapsed routing, not missing ρ/τ channels.

## Locks

| Field | Value |
|-------|--------|
| Init | Continue cold best only (forbid Fix-1 / `HEALTHY_V7`) |
| Graph | Cα Hyp MP; `hyp_biology_mp=false` |
| Arch parity | `rim_fanout_forward=True` (0.14 / min_r 0.20), `geometric_angular_prior=True` |
| Specialization | **C:** `majority_committed_share` + `routing_load_floor` (light) + small `balance_coeff` |
| Fill | `disc_eff_rank_coeff` → Pass `disc_effective_rank_mean > 1.5` |
| Disc hold | `disc_r_mean ≥ 0.25` |
| MLflow | **On** — experiment `tokyo-eyes-v7`, run name = RUN_ID (parent cold v1 did **not** log; that is why UI was empty) |

## Light coeff defaults (v1)

| Coeff | Default |
|-------|---------|
| `majority_committed_share_coeff` | 0.08 |
| `routing_load_floor_coeff` | 3.0 |
| `routing_load_floor_min` | 0.05 |
| `balance_coeff` | 0.01 |
| `disc_eff_rank_coeff` | 1.0 |
| funnel disc / cone / proto | carry cold v1 |

## Pass form

| Check | Bar |
|-------|-----|
| `disc_effective_rank_mean` | **> 1.5** (hard save) |
| `disc_r_mean` | ≥ 0.25 hold |
| Hard expert monopoly | report: max hard share ≪ 1.0 |
| Soft commit (`mean max soft`) | report: ↑ vs cold ~0.29 |
| Basin / migration vs cold-best & sealed | report-only |

## Isolation

| Artifact | Path |
|----------|------|
| Run dir | `checkpoints/v7/runs/tokyo_eye_v7_cold_hyp_mp_funnel_angfill_v1/` |
| Best | `v7_cold_hyp_mp_funnel_angfill_best.pt` |
| Train | `make train-v7-cold-hyp-mp-funnel-angfill` |
| Gate | `data/gates/tokyo_eye_v7_cold_hyp_mp_funnel_angfill_prereg.json` |
