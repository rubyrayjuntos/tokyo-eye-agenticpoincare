# Curvature consumer map (structural SSOT + v6 encoder)

Single reference for **which numeric curvature** feeds which subsystem. Prevents
checkpoint-vs-pin split-brain when reading occupancy audits or planning retrains.

See also: [`PYG_ATTACH_MUTATION.md`](PYG_ATTACH_MUTATION.md) — false-identical disc
coordinates across checkpoints from in-place PyG reuse.

## What “structural SSOT” means (and what it excludes)

| In SSOT (frozen, not GNN-learned) | Outside SSOT (varies today) |
|-----------------------------------|-----------------------------|
| Layout **algorithm**: ρ, τ, Cα → macro hubs, PCA offsets, Möbius compose | Numeric **curvature c** passed into `expmap₀` |
| Tier-1 governed inputs (ρ, `tau_flag`, ss, Cα) | Checkpoint `model.curvature` (learned `softplus(log_c)`) |
| Passthrough of composed `structural_z_disc` when `structural_disc_frozen=True` | Ball lift, soft clamp `k = -model.curvature` |
| Hyperbolic k-NN edges built from composed disc coords | Encoder routing, expert weights, learned heads |

**“Structural SSOT frozen” means the GNN does not re-learn disc placement** — not
that curvature is pinned to a single governed constant.

Today, compose + hyperbolic edges consume **`model.curvature` at attach time**
(e.g. cold_v1 ≈ 0.769, route_v1 ≈ 0.686). That is **inside the compose pipeline**
but **outside Tier-1 physics inputs**. Two checkpoints therefore produce **different**
`structural_z_disc` (verified: 1MBN max diff ≈ 0.060 at those c values).

### Why `CANONICAL_V6_CURVATURE` (0.7026) exists separately

| Role | Purpose |
|------|---------|
| Governed ingest / onboard contract pin | Cross-run comparability, DB `embedding_space.curvature` reference |
| Lever-A disc checkpoint lineage | Historical SSOT for production v5/v6 ball geometry |
| **Not used** on slim MoE training attach path today | Compose does not default to this pin unless code is changed |

The pin is the **recommended target for retrain A/B placement** so cold/route/audit
runs share one placement c while `model.curvature` may still learn for ball lift.

## Constants and sources

| Symbol / name | Value (approx) | Source |
|---------------|----------------|--------|
| `CANONICAL_V6_CURVATURE` | 0.7026273608207703 | Pin in `science/dtie/common/curvature_loader.py` (lever_a disc checkpoint) |
| `CANONICAL_TS002_CURVATURE` | 0.6054342985153198 | Deprecated TS-002 reference only |
| `model.curvature` | Per-checkpoint learned | `softplus(log_c)` in v6 GNN; e.g. cold_v1 ≈ 0.769, route_v1 ≈ 0.686 |

`require_learned_curvature()` rejects missing/non-positive **explicit** arguments; it
does not inject a default numeric c.

## Consumer matrix (current slim MoE + structural SSOT path)

| Stage | Curvature used today | Notes |
|-------|----------------------|-------|
| **Structural disc compose** (`compose_structural_disc`) | `model.curvature` at attach time | ρ/τ → `expmap₀` placement, `hyperbolic_r`, cone depth on artifact |
| **Hyperbolic k-NN graph** (`build_hyperbolic_disc_graph`) | Same c as compose (artifact `curvature_c`) | Edge geodesic distances + `log_p` tangents |
| **Viewer export** (`disc_payload_from_nodes`) | `model.curvature` for `hyperbolic_r` tooltip | Coordinates from frozen `structural_z_disc` |
| **v6 forward ball lift** (`structural_ball_lift_from_disc`) | `model.curvature` (`self.curvature`) | Maps frozen disc → high-d ball for MoE |
| **v6 forward disc output** (`hyp_projections_2d` when frozen) | Coords from artifact; soft clamp uses `k = -model.curvature` | **Not** re-learned; `structural_z` passthrough + `project_disc_2d` |
| **GNN message passing** | N/A (Euclidean node_emb → hyperbolic edges) | Hyperbolic edge **distances** use compose c |
| **DB / ingest governed rows** | `embedding_space.curvature` from inference run | Separate from training checkpoint unless synced |

**Today there is one c per forward for structural paths:** compose, hyperbolic edges,
and attach all use the **same** `model.curvature` passed into
`attach_structural_disc_for_forward`. They are **not** pinned to `CANONICAL_V6_CURVATURE`
on the slim MoE training path.

## σ₂/σ₁ occupancy convention (do not invert)

Implemented in `science/training/disc_occupancy.py` and
`experiments/diagnostics/embedding_occupancy_audit.py`:

```
σ₂/σ₁ = s[1] / s[0]   # centered SVD on disc xy, s[0] ≥ s[1]
```

| Value | Interpretation |
|-------|----------------|
| **→ 1.0** | Healthy 2D spread (both axes carry variance) |
| **→ 0.0** | Rank-1 streak / collapse (PC2 ≪ PC1) |
| **Promote floor** | 0.35 default (`DISC_SIGMA2_SIGMA1_PROMOTE_MIN`) |
| **Crescent block** | < 0.22 with low eff_rank + thin streak |

**High σ₂/σ₁ is good.** Example: 1MBN disc_2d σ₂/σ₁ ≈ 0.95 is **strong** occupancy on
structural SSOT coordinates, not a streak flag.

### σ₂/σ₁ invariance under different c — assumption vs test

Uniform scaling of all coordinates preserves σ₂/σ₁, but `expmap₀` applies a
**per-point** radial warp: `z = v · (1/√c) · tanh(√c·‖v‖/2) / ‖v‖`. Invariance
across c is **not guaranteed a priori**; it is an empirical claim.

Property tests in `tests/test_pyg_structural_attach_mutation.py`:

- Synthetic tangent clouds → `expmap₀` at two c values
- Full `compose_structural_disc` on synthetic ρ/τ residues at two c values

If tests pass broadly, similar σ₂/σ₁ across checkpoints with different c is
plausible **in addition to** checking coordinates are not mutation-contaminated.

When `structural_disc_frozen=True`, disc_2d occupancy audits measure **physics SSOT
placement**, not encoder-learned layout.

## Expert-colored disc — how to read (and misread)

Split viewer disc coloring (`interactive_viewer.py`):

- Metric `expert`: `t = expert_index / 3` on a red→blue continuous scale (E0 red-orange, E3 blue)
- **No discrete legend** on canvas; confirm expert index from tooltip (`E0`–`E3`) or audit JSON
- Dominant expert (e.g. route 1MBN: E2 = 109/153 ≈ 71%) **covers most of the disc by construction** — spatial spread is collapse rendered, not specialization
- Minority experts (E1 = 8, E3 = 8 on route 1MBN) can look “localized”; that is the only pattern worth treating as a specialization signal without corroboration
- Always note **which checkpoint** generated the viewer; cold vs route dominant profiles differ sharply

## Recommended retrain gate (encoder / routing — not disc SSOT)

| Metric | Layer | Pass hint |
|--------|-------|-----------|
| σ₂/σ₁ | `x_routed_hyp` or pre-routing learned disc (frozen **off**) | > 0.35 floor; watch trend |
| eff_rank | same | > 1.6 promote target |
| Routing H | expert_weights | < ~1.0 (not blocked at ln 4) |
| Expert load | dominant fraction | No single expert > ~60% on corpus |

Disc SSOT viewer looking “good” validates Tier-1 physics placement only.

## Retrain policy options (pick one before greenlight)

1. **Pin placement c** — `CANONICAL_V6_CURVATURE` for compose + hyperbolic edges; `model.curvature` learns for ball lift only
2. **Single c** — keep today’s wiring; log `structural_disc_curvature_c` on every audit row

Either is valid; the doc must match the code path chosen.
