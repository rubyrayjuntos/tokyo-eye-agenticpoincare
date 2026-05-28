# GNN Poincaré Ball Embedding: Training-Level Bug Report

**Date:** 2026-05-13
**Severity:** Critical — all Phase 1 Poincaré embeddings are invalid
**Impact:** Visualization layer only. Structural physics results (Phase 2, 4b) are unaffected.

## The Bug

All 200 landmark points in the KRAS 4OBE Poincaré embedding have |p| = 1.192, outside the unit ball. The embedding is mathematically invalid — points must satisfy |p| < 1 for the Poincaré ball model.

## Root Cause

Two compounding errors in `data_science/sub_agents/dtie/phases/phase1_witness_embedding.py`:

### Error 1: Wrong input to exponential map

```python
landmarks_euclidean = kmeans.cluster_centers_  # [K, 3] — raw Cartesian Å coordinates
landmarks_poincare = map_to_poincare(landmarks_euclidean, c=curvature_c)
```

The exponential map receives **raw 3D protein coordinates** (N-O midpoints in Ångströms, norms ~10–30Å), not GNN-learned tangent vectors. The GNN only provides the curvature parameter `c`; it does not produce per-residue tangent-space embeddings for Phase 1.

### Error 2: Missing /2 in exponential map formula

The implementation:
```python
scale = tanh(√c · ‖v‖) / (√c · ‖v‖)
|p| = scale × ‖v‖ = tanh(√c · ‖v‖) / √c
```

With c=0.704 and ‖v‖=15Å: `|p| = tanh(0.839 × 15) / 0.839 = 1.0 / 0.839 = 1.192`

The correct Poincaré ball exp map at origin with curvature -c:
```
exp_0(v) = tanh(√c · ‖v‖ / 2) · v / (√c · ‖v‖)
|p| = tanh(√c · ‖v‖ / 2) / √c
```

Even with the /2 fix, feeding raw Cartesian coordinates (‖v‖ >> 1) will still saturate tanh and push |p| → 1/√c = 1.19 for c=0.704.

### The fundamental problem

Phase 1 does not use GNN embeddings at all. It:
1. Takes N-O midpoint coordinates (physical 3D space)
2. Runs k-means to select landmarks
3. Applies the exponential map to the k-means centroids using GNN curvature

This is geometrically meaningless. The exponential map expects tangent vectors at the origin (small displacements in the tangent plane), not absolute Cartesian coordinates with norms of 10–30Å.

## What should happen

For a valid Poincaré ball embedding, Phase 1 should either:

**Option A:** Use the GNN's actual per-residue embeddings (the 64-dim projections from `fact_gnn_node_output.projections`) and project those into the ball. These are learned representations in a latent space where the exp map is meaningful.

**Option B:** Normalize the Cartesian coordinates to unit scale before applying the exp map:
```python
centered = landmarks_euclidean - landmarks_euclidean.mean(axis=0)
max_norm = np.linalg.norm(centered, axis=1).max()
normalized = centered / (max_norm + 1e-8)  # now ‖v‖ < 1
landmarks_poincare = map_to_poincare(normalized, c=curvature_c)
```

**Option C:** Use a learned hyperbolic embedding (e.g., train a Poincaré embedding of the residue contact graph using the Nickel & Kiela 2017 algorithm).

## Impact Assessment

| Pipeline component | Uses Phase 1 Poincaré? | Affected? |
|-------------------|----------------------|-----------|
| Phase 2 (dehydron scan) | No — uses raw ensembles | ✗ Not affected |
| Phase 3 (witness persistence) | Yes — uses witness distances | ⚠️ Distances computed in Euclidean, not hyperbolic |
| Phase 3.5 (topological lift) | Depends on Phase 3 | ⚠️ Indirectly affected |
| Phase 4 (resistance mapping) | Uses Phase 3.5 output | ⚠️ Indirectly affected |
| Phase 4b (conductance/λ₂) | No — uses Cα contact graph directly | ✗ Not affected |
| Phase 5 (pharmacophore) | Uses Phase 4 output | ⚠️ Indirectly affected |
| Visualization (Poincaré disc) | Yes — directly | ✗ BROKEN |

**Critical structural physics results (ρ values, λ₂, doorways) are NOT affected.** They come from Phase 2 and Phase 4b which operate on raw coordinates and contact graphs, not Poincaré embeddings.

## Fix Priority

1. **Immediate:** Do not use Phase 1 Poincaré coordinates for visualization. Use PCA of dehydron coordinates or the GNN's 64-dim projections instead.
2. **Short-term:** Implement Option B (normalize before exp map) as a quick fix to get valid ball coordinates.
3. **Medium-term:** Implement Option A (use GNN projections) for a learned hierarchy.
4. **Long-term:** Train a proper hyperbolic embedding (Option C) with boundary penalty in the loss function.

## Verification

After fix, confirm:
- All |p| < 0.95
- Radial distribution shows meaningful variance (not all points at same radius)
- Key residues (Q61, G12) have interpretable radial positions
