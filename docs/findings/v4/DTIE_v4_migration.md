# DTIE v4 Migration Guide
## Tokyo Eyes GNN: Full Hyperbolic Pipeline

**Date:** 2026-05-13  
**Status:** Files written, retraining required before production use.

---

## What was built

Three files implementing the original design intent: TDA operating on
the GNN's hyperbolic conformational hierarchy, with uncertainty gating
the landmark selection.

| File | Change |
|------|--------|
| `Gnnv4.py` | Steps 4–6 in hyperbolic/tangent space. Exposes `x_hyp`, `x_routed_hyp`, `hyp_projections`. |
| `phase1_witness_embedding_v4.py` | Landmarks selected from `x_routed_hyp` by uncertainty gate, not k-means on Cartesian. Distances are hyperbolic. |
| `phase3_witness_persistence_v4.py` | Witness complex uses hyperbolic distances. max_alpha in hyperbolic units. |

---

## Hard constraint: retraining required

`robust_experts.pt` cannot load into Gnnv4 because three weight shapes changed:

| Component | v3 input shape | v4 input shape |
|-----------|---------------|----------------|
| Gate | `x` [N, hidden] (Euclidean) | `x_tangent` [N, hidden] (logmap0(x_hyp)) |
| Experts | `x` [N, hidden] (Euclidean) | `x_tangent` [N, hidden] (logmap0(x_hyp)) |
| EvidentialHead | `x_routed` [N, hidden] | `cat([logmap0(x_routed_hyp), depth, cone_width])` [N, hidden+2] |

The expert weights and uncertainty head weights are shape-compatible in
`hidden` but semantically incompatible — loading them would produce
undefined behavior without raising an error.

**Do not use `strict=False` to load the old checkpoint into v4.**

---

## What to retrain on

The training loop is unchanged. `gosp_loss()` works with v4 output.
The input contract is unchanged: node_dim=4, (rho, tau_flag, ss_type, sasa).

```python
from Gnnv4 import GOSPConeMapper, gosp_loss, build_optimizer, precompute_clustering

model = GOSPConeMapper(
    node_dim=4,
    hidden=128,
    num_layers=6,
    num_experts=4,
    projection_dim=64,
    hyp_proj_dim=2,       # NEW: 2D disc output
    depth_conditioning=False,
)
optimizer = build_optimizer(model, lr=1e-3)

# Training loop unchanged — gosp_loss takes same arguments
loss_dict = gosp_loss(output, target_rho, target_dehydron)
```

Run `python Gnnv4.py --internaltest` before training to verify 7 property checks pass.

---

## gnn_runner.py changes needed

Two changes to make gnn_runner.py feed v4:

**1. Restore 4-dim input contract:**

```python
# REMOVE this (the refactor that broke things):
NODE_FEATURE_DIM: int = 28

# RESTORE:
NODE_FEATURE_DIM: int = 4

# build_node_features() must produce [N, 4]: [rho, tau_flag, ss_type, sasa]
# The 28-dim expansion is incompatible with the checkpoint and v4 architecture.
```

**2. Pass x_hyp outputs through to gnn_output.npz:**

```python
# In _run_single_condition(), add to return dict:
return {
    ...existing keys...,
    "x_routed_hyp": _np(output["x_routed_hyp"]),   # [N, 64] in ball
    "x_hyp": _np(output["x_hyp"]),                  # [N, 64] in ball
    "hyp_projections": _np(output["hyp_projections"]),  # [N, 2] disc coords
}

# In run_gnn(), add to result dict:
result[f"{prefix}x_routed_hyp"] = gnn_out["x_routed_hyp"]
result[f"{prefix}x_hyp"] = gnn_out["x_hyp"]
result[f"{prefix}hyp_projections"] = gnn_out["hyp_projections"]
```

---

## Phase 1 v4 interface change

Phase1Output field `witnesses` now contains `[N, hidden]` Poincaré ball
positions, NOT `[N, 3]` NO_Midpoints.

Phase 3.5 (topological lift) is unaffected because it uses
`landmark_to_residue_map` to find physical coordinates — it never
directly uses the witnesses array geometry.

Phase 3 v4 includes a guard that raises if `witnesses.shape[1] == 3`,
catching any accidental v3/v4 mixing.

---

## max_alpha calibration for Phase 3

v3 used max_alpha=20.0 Å (Ångström-scale dehydron distances).  
v4 uses max_alpha=4.0 hyperbolic units.

To calibrate for your specific KRAS training run:
1. After training, compute `pmath.dist0(x_routed_hyp, k=-c)` for all residues.
2. Find the median inter-residue hyperbolic distance within each known domain.
3. Set max_alpha to 1.5× the cross-domain distance (coupling but not full funnel).

The default 4.0 is a reasonable starting point for c ≈ 0.7.

---

## What v4 gives you that v3 didn't

| Signal | v3 | v4 |
|--------|----|----|
| Disc coordinates | Post-hoc PCA of broken Cartesian embedding | Native MobiusLinear from ball |
| TDA geometry | Physical Ångström space | Learned conformational hierarchy |
| Uncertainty grounding | Euclidean latent space | Hyperbolic position-aware |
| Expert routing | Euclidean features | Tangent-space features from ball |
| Allosteric channel detection | Geometric loops in 3D | Hierarchical loops in wrapping landscape |

---

## Verification checklist before production

- [ ] `python Gnnv4.py --internaltest` passes (7 checks)
- [ ] Retrained checkpoint: all `|x_hyp|² < 1/c` for all residues
- [ ] `hyp_projections` norms < 1.0 (true disc coords)
- [ ] Phase 1 v4: landmark uncertainty stats logged and reasonable
- [ ] Phase 3 v4: at least 1 terminal leak found on KRAS GDP
- [ ] WT vs G12D: hyperbolic distances between domain barycenters differ
- [ ] SBIR language: "structural physics results derive from Phase 2 and
      Phase 4b, both of which are unaffected by the v4 GNN rewrite"
