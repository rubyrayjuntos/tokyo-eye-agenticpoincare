# Tokyo Eye v7 — B′ disc-health continue (prereg)

**Status:** PREREG  
**Date:** 2026-07-21  
**Decision:** Option B — mild disc occupancy / depth-scale bump + sparsity-style save eligibility (no legacy `H(f̄)≤1.21`)

## Resume

| Field | Value |
|-------|--------|
| Checkpoint | `checkpoints/v7/runs/tokyo_eye_v7_bprime_health_continue_v1/v7_best_disc.pt` |
| Prior | continue_v1 (health FAIL; peak `disc_r_mean`≈0.21) |
| Locks | `hyp_mp_primary=True`, `se3_aux=False`, `hyperbolic_mp_graph=False` |

## Stack delta (vs `V7_BPRIME_STACK`)

| Knob | Prior (phase 12 geom angular) | This run |
|------|-------------------------------|----------|
| `disc_occupancy_coeff` | ≈0.49 (`0.65×0.75`) | **0.70** |
| `disc_depth_scale_coeff` | 1.25 | **1.60** |
| `disc_depth_scale_target` | 0.55 | **0.60** |
| `routing_entropy_sparsity_coeff` | 0 (off) | **0** (no further sharpen) |
| Save eligibility | legacy `H(f̄)≤1.21` | **sparsity-style**: mean_residue ∈ **[0.25, 0.90]**, max_share &lt; 0.45; `H(f̄)` monitor/abort only |

### Why widen mean_residue floor to 0.25

Continue_v1 sits at `mean_i H(p_i)≈0.31` with balanced `H(f̄)≈1.33` — healthy local commit + global balance. Champion band `[0.50, 0.90]` would reject that niche. This prereg keeps sparsity *style* (drop legacy ceiling) without forcing λ-sparse.

### Why not raise 1.21 toward ln(4)

`H=1.21` ≈ 3.35 effective experts (~84% of 4-way uniform), **not** 2-expert routing. Raising the ceiling alone does not Pass `disc_r_mean≥0.25`.

## Health Pass bars

1. Forward audit: `hyp_mp_primary=true`, `se3_aux=false`
2. Final `disc_r_mean` ≥ **0.25**
3. Learned curvature finite and &gt; 0
4. Train completes without NaN loss
5. Prefer eligible `v7_best` under sparsity-style gates (telemetry; not a biology Pass)

**Non-claims:** biology vs Fix-1; KRAS hub migration; promote / sparsity champion status.

## Commands

```bash
make train-v7-bprime-disc-continue DEVICE=cuda EPOCHS=24
```

## Artifacts

| Path | Role |
|------|------|
| `data/gates/tokyo_eye_v7_bprime_disc_health_continue_prereg.json` | This lock |
| `checkpoints/v7/runs/tokyo_eye_v7_bprime_disc_continue_v1/` | Run dir |
