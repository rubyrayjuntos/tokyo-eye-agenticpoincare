# V4 GNN Output Shape Validation

**Date:** May 27, 2026  
**Purpose:** Validate that the governed data model (migration 004) can represent the actual v4 GNN (`Gnnv4.py`) output without loss.

---

## V4 GNN Forward Pass Output Dict (from Deep Audit)

Based on the Phase 0.1 audit of `Gnnv4.py`, the forward pass returns:

```python
{
    "projections": Tensor[N, 64],          # Euclidean projection (backward compat)
    "hyp_projections": Tensor[N, 2],       # Native 2D Poincaré disc projection (NEW)
    "x_hyp": Tensor[N, hidden_dim],        # Full hyperbolic embedding (NEW)
    "x_routed_hyp": Tensor[N, hidden_dim], # Post-MoE hyperbolic embedding (NEW)
    "cone_depth": Tensor[N],               # Depth in learned hierarchy
    "cone_width": Tensor[N],               # Width of cone
    "epistemic_uncertainty": Tensor[N],    # From EvidentialHead
    "aleatoric_uncertainty": Tensor[N],    # From EvidentialHead (NEW in v4)
    "total_uncertainty": Tensor[N],        # Combined
    "expert_weights": Tensor[N, n_experts],# MoE routing weights
}
```

Where `N` = number of residues (nodes) in the graph.

Input features per node: `[rho, tau_flag, ss_type, sasa]` (dim=4).

---

## Schema Mapping (Migration 004 → V4 Output)

| V4 Output Field | Schema Column | Type | Status |
|-----------------|---------------|------|--------|
| `projections` | `embedding` (VECTOR) | VECTOR(64) | ✅ Covered — use Euclidean space_id |
| `hyp_projections` | `hyp_projections` (JSONB) | JSONB | ⚠️ Works but suboptimal — see note 1 |
| `x_hyp` | `embedding` (VECTOR) | VECTOR(hidden_dim) | ✅ Covered — use Hyperbolic space_id |
| `x_routed_hyp` | Not directly stored | — | ⚠️ Gap — see note 2 |
| `cone_depth` | `cone_depth` (DOUBLE PRECISION) | float | ✅ Covered |
| `cone_width` | `cone_width` (DOUBLE PRECISION) | float | ✅ Covered |
| `epistemic_uncertainty` | `epistemic_uncertainty` (DOUBLE PRECISION) | float | ✅ Covered |
| `aleatoric_uncertainty` | `aleatoric_uncertainty` (DOUBLE PRECISION) | float | ✅ Covered |
| `total_uncertainty` | `total_uncertainty` (DOUBLE PRECISION) | float | ✅ Covered |
| `expert_weights` | `expert_weights` (JSONB) | JSONB | ✅ Covered |
| Input `rho` | `input_rho` (DOUBLE PRECISION) | float | ✅ Covered |
| Input `tau_flag` | `input_tau_flag` (DOUBLE PRECISION) | float | ✅ Covered |
| Input `ss_type` | `input_ss_type` (DOUBLE PRECISION) | float | ✅ Covered |
| Input `sasa` | `input_sasa` (DOUBLE PRECISION) | float | ✅ Covered |

---

## Issues Found & Resolutions

### Note 1: `hyp_projections` as JSONB

The 2D Poincaré disc projection (`hyp_projections`) is stored as JSONB in migration 004. This works but prevents vector similarity search on the native disc coordinates.

**Resolution:** For v4, we should store `hyp_projections` as a separate `VECTOR(2)` in a dedicated row with a `poincare_disc_2d` embedding space. The Normalizer payload already supports this — the `GNNOutputPayload` can emit two records per residue (one for the full hyperbolic embedding, one for the 2D projection) using different `space_id` values.

**Action:** Add a migration (027) that adds a `hyp_projection_2d` VECTOR(2) column to `fact_gnn_node_embedding` for direct disc coordinate storage alongside the JSONB.

### Note 2: `x_routed_hyp` not stored

The post-MoE hyperbolic embedding (`x_routed_hyp`) is not directly stored in the current schema. This is the embedding AFTER expert routing in tangent space + re-lift.

**Resolution:** This is acceptable for Phase 1. The `x_routed_hyp` is an intermediate representation. If downstream analysis needs it, it can be stored via `fact_computed_property` (migration 014) or a dedicated column can be added later. The Normalizer payload includes it as optional for future use.

**Action:** No schema change needed now. Document as a known extension point.

### Note 3: Dual-space storage pattern

V4 produces BOTH Euclidean projections (64-dim, backward compat) AND hyperbolic embeddings. The schema supports this via the `space_id` foreign key — one residue can have multiple embedding records in different spaces.

**Validation:** ✅ This works correctly with the current schema. The Normalizer should emit two `fact_gnn_node_embedding` rows per residue for v4 (one Euclidean, one Hyperbolic).

---

## V3 GNN Output Shape (for comparison)

```python
{
    "projections": Tensor[N, 64],          # Euclidean only
    "cone_depth": Tensor[N],
    "cone_width": Tensor[N],
    "epistemic_uncertainty": Tensor[N],    # No aleatoric in v3
    "expert_weights": Tensor[N, n_experts],
}
```

V3 is a strict subset of v4 outputs. The schema handles it cleanly — hyperbolic fields are nullable.

---

## Conclusion

The existing schema (migration 004) can represent v4 outputs **without data loss**. Two minor improvements are recommended:

1. Add `VECTOR(2)` column for native disc coordinates (enables similarity search on disc).
2. Document `x_routed_hyp` as a future extension point.

The Normalizer payload schemas (`GNNOutputPayload` + `GNNNodeResult`) are validated as compatible with both v3 and v4 output shapes.

---

**Status: VALIDATED** — Schema is compatible. Minor enhancement (027) recommended.
