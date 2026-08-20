# Hierarchical Containment Edges — Depth-Collision Addendum

**Status:** Paper decision locked 2026-07-17. **Still binding** under containment design v2. Chem-MVP prerequisite is satisfied (`partial`); hierarchy materialization is **Path B** (train-side stopgap) per [`design.md`](design.md) §9. Implementation may proceed under that lock **without reopening this addendum**.  
**Related:** [containment design v2](design.md) · [containment ablation](ablation.md) · [`../struct-conn-typed-edges/design.md`](../struct-conn-typed-edges/design.md) · [`../learned-flow-influence/ablation.md`](../learned-flow-influence/ablation.md) · [barcode edge design §9](../dehydron-barcode-input-channel/design.md#9-deferred-work-post-p1)

## 1. Why this addendum exists

The hierarchical containment proposal introduces a **centripetal loss**: macroscopic nodes (domains / chains / assemblies) toward the origin, microscopic residues toward the rim.

The residue **disc / cone radial coordinate is already a trained target** in this codebase — typically via `cone_target_mode=tau_dehydron_rim` (dehydron underwrap → rim depth). Asking that **same** radial scalar to also encode hierarchical containment level is the same failure shape as the historical **SASA–depth collapse** (`shell_corr_depth_sasa_weight` pulling depth toward SASA instead of the physics quantity it was meant to represent).

This must be resolved **on paper before any containment training**, not discovered after a rim-enrichment / cone–τ regression.

## 2. Locked decision

**Physics radial authority on residue leaves remains exclusive.**

| Quantity | Owner | Hierarchical containment may… |
|----------|-------|--------------------------------|
| Residue `cone_depth` / disc radius / `tau_dehydron_rim` target | **Physics only** | **Not** receive a centripetal / hierarchy-level penalty |
| Parent-node embeddings (domain / chain / assembly — **new node types**) | Hierarchy | Carry their own radial / origin-proximity targets |
| Vertical edge message passing | Hierarchy | Use distinct relation IDs (extend multi-rel stack), without rewriting residue physics depth |

**Rejected without a new, separate registration:**

- Dual-load residue radius (physics + hierarchy on one scalar)
- “Small auxiliary weight” hierarchy centripetal on residue disc radius as a soft compromise — still the SASA-depth pattern unless physics non-regression gates are absolute and hierarchy weight is zero on residue radial heads

## 3. Options considered

### A. Separate hierarchy radial channel / subspace (residue)

Add a second residue radial head for hierarchy.  
**Rejected for v1 containment:** doubles the geometry story the investigation has spent months stabilizing; still couples into disc projection unless carefully isolated.

### B. Auxiliary hierarchy loss on the same residue radius with small weight

**Rejected as default:** structurally identical to SASA–depth dual-load; “small weight” is how that failure mode entered training.

### C. Hierarchy targets only on parent nodes; residue leaves keep physics depth (adopted)

Heterogeneous graph: parent nodes get centripetal / tier-margin losses; residue leaves keep `tau_dehydron_rim` (or whatever active physics cone target). Vertical edges communicate through message passing; **shortcut effects** are scored by geodesic / pair-distance probes (reuse coupled-lock / Chem-MVP methodology: trunk **and** disc), not by forcing residue radius to mean “hierarchy level.”

## 4. Implications for the original containment validation plan

| Original metric | Status under this lock |
|-----------------|------------------------|
| “Mean hyperbolic radius by hierarchy level 0…3 with residue leaves at rim” | **Invalid as a residue-leaf criterion** if it requires residue radius to encode level. Rephrase: parent-node radii stratify by level; residue radii remain physics-owned |
| Horizontal edge preservation / interaction benchmarks | Keep — but translate to **pipeline gates** (physics rim enrichment, cone/τ, MoE) before training |
| Dilution / same-domain geodesic shrink | **Keep** — reuse Chem-MVP / `kras_coupled_lock_ood` trunk+disc distance methodology with domain-member pairs as the positive set |

## 5. Pre-registered non-negotiables (when containment is eventually built)

1. Residue `cone_target_mode` / cone–τ / physics rim gates **must not regress** vs the containment parent under an explicit refusal threshold.
2. Any centripetal term applies to **parent node types only** in v1.
3. Parent-node feature init — **resolved in design v2**: mean-pool (Option A) for first experiment; learned embeddings only if A fails acceptance.
4. Sequencing: Chem-MVP closed (`partial`) → **containment next** ([`design.md`](design.md) v2, flow-motivated) → Path 2 directionality reward on diam ≤9 → optional barcode typed shared-bar. Chem-Full remains independent.

## 6. One-line SSOT

> Hierarchy may own **parent-node** radial geometry and **vertical** relation types; it does **not** own residue disc / cone depth. That stays physics (`tau_dehydron_rim` or successor). Dual-loading residue radius is a known failure mode in this repo — do not reintroduce it.
