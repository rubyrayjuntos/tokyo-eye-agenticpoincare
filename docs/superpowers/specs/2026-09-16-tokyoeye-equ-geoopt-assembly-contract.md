# Tokyo Eye EQU — Geoopt Assembly Contract (restore)

**Status:** APPROVED_LOCKED — Ray: start over on 1–4 the right way (2026-09-16)  
**Date:** 2026-09-16  
**Source:** Ray assembly blueprint (EquiformerV3 + geoopt Poincaré + R0–R5 + MoE + evidential + telemetry)  
**Does not authorize train until APPROVED_LOCKED + preflight Pass**

## 1. Intended stitch (three domains)

1. **SE(3) front-end:** `mirror-physics/equiformer_v3` / `atomicarchitects/equiformer_v3` weights **through final pooling** (OMat24/OC20/MPtrj family).  
2. **Hyperbolic mechanics:** `geoopt.manifolds.PoincareBall` for lift + geodesic ops; RiemannianAdam where manifold params require it.  
3. **Sparse MoE + telemetry:** relation-aware hyp attention on frozen R0–R5; E0–E3 routing; radius-aware evidential; Poincaré flow telemetry.

## 2. Live v8 vs contract (honest gap)

| Piece | Blueprint | Live v8 EQU (lift_radius lineage) |
|-------|-----------|-----------------------------------|
| Equiformer bind | Full forward to pooling | **SE(3)-lite** over weight bank (block-0 slices) |
| R0–R5 graph | Freeze typed sparse graph | **Present** (`r0_r5_graph.py`) |
| Lift | `geoopt.PoincareBall.expmap0` | **Hand-rolled** `exp_map_zero` + `RadialAngularProjector` |
| Hyp attention | geoopt `manifold.dist` sparse | **On-manifold gyro + Einstein/Klein** (hand-rolled; pure-hyp compliant) |
| MoE | Top-1 physical-scale experts | Soft Gumbel MoE + CV/quota (live) |
| Evidential | Radius→Dirichlet toy | **EvidentialHead (γ,ν,α,β)** on post-MoE features |
| Telemetry | PoincareFlowTelemetryLogger | PoincareDiagnosticsEngine + MLflow diags |
| Optimizer | RiemannianAdam | AdamW on spine (no geoopt manifold params) |

## 3. What we restore (next card scope)

**Must restore (real drift):**
- A) Equiformer structural forward **to pooling cut** (or sealed proof that lite ≡ that cut — default is restore, not “prove lite”).  
- B) Lift via **`geoopt.manifolds.PoincareBall`** (`expmap0` + `proj`) wrapping projector output as `ManifoldTensor` where needed.  
- C) Preflight: geoopt path used in forward; no silent handroll fallback under `pure_hyp_strict`.

**Keep (already better / sealed law):**
- Pure-hyp after lift: **no** dumping attention into Euclidean for MoE as the geometry path (blueprint sketch that returns Euclidean from attention is **vetoed**). GyroOrthogonal + Einstein aggregation stays unless a sealed card replaces it with an **on-manifold** geoopt equivalent.  
- Soft MoE + load diagnostics (Top-1-only is diagnostic, not the training default).  
- Existing EvidentialHead NIG-style 4-tuple (blueprint radius-only Dirichlet is weaker — may *augment* with radius channel, not replace).  
- Existing MLflow / PoincareDiagnostics (blueprint logger maps onto these metrics).  
- MPtrj pin sha `59c6c235…`; wrap_max=1; Stage-A panels; no Pearson until ligand inference returns.

## 4. Explicit non-adoptions from LLM sketch

- `missing_stubs` / broken mock edge_index examples — not code.  
- Attention → Euclidean value path as the sealed hyp spine.  
- Hard Top-1 MoE as the only router.  
- Immediate theme_biology relaunch on unrestored stack.  
- Blind TorchScript of geoopt graphs.

## 5. Exit of restore card (proposed)

Gate id (provisional): `tokyo_eye_equ_geoopt_restore`  
Pass = preflight proves (A)+(B)+(C) + pure_hyp + geometry HOLD on Stage-A probe (sat/spread/equiv lift+spine).  
Biology theme AUPRC is **out of scope** for restore — separate card after restore QUALIFIED.

## 6. Target product question (Ray)

Blueprint asks macro binding cascades vs local ligand pockets. **Current freeze:** Stage-A fold themes / dehydron mechanism first; ligand/Pearson parked. Macro cascade is later corpus-expand, not this restore.


## Framing note (2026-09-16 MoE eval collapse)

E0–E3 “physical-scale tiers” (wrapped core / dehydron rim / interface / coil) are **design intent**.
On `tokyo_eye_equ_geoopt_restore` QUALIFIED best (ep19), **eval argmax** uses only E1+E2; E0=E3 dead.
Do not claim four-tier physical routing as observed behavior until `tokyo_eye_equ_moe_eval_util` Pass.
Train-step `moe_load_*` under Gumbel STE is **not** evidence of eval utilization.
