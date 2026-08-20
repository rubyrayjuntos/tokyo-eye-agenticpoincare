# Tokyo Eye v7 — design

**Status:** LOCKED for implementation  
**Date:** 2026-07-21  
**Companion:** [`README.md`](README.md)

## Goals

1. Hyperbolic **neighbor** message passing is primary (Möbius/gyro aggregation on a hyp graph).
2. Euc 3D coords/chemistry remain the reference for building features and (optional) SE(3) aux — not the primary transport engine.
3. Gate/MoE stay hyperbolic (as in v6).
4. Curvature and cone depth are **learned**; no hardcoded curvature on v7 codepaths.
5. Production module is always `science/tokyo_eye/TokyoEye.py`.
6. Zero GNN versioning under `science/dtie/`; that tree is structural biology (rename later).

## Forward sketch

```
structure/chem features (Euc ref)
  → hyp graph edges (Poincaré / typed)
  → Hyp MP stack (primary)  →  x_hyp
  → optional SE(3) aux (secondary; not S4)
  → Hyp Prototype Gate + MoE
  → cone_depth = dist0(...); uncertainty; disc views
```

## Non-goals (this open)

- Editing `science/dtie/**`
- Renaming dtie → structural-biology
- Softening KRAS allele Child-1 bars
- Reopening chem-MVP as trunk
- Relocating production model into `science/dtie/v7/...` on promotion

## SE(3) vs “S4”

Historical training flags named “S4 hyperbolic MP graph” attached Poincaré k-NN edges into **Euclidean** `EquivariantConv`. That is **not** Hyp MP and must not bump platform version. v7 default: `hyp_mp_primary=True` with real ball aggregation.

## Isolation

| Import | Allowed on v7 path? |
|--------|---------------------|
| `science.tokyo_eye.*` | yes |
| `experiments.training.v7.*` | yes |
| `science.dtie.common.*` / platform common | yes (structural builders) |
| `science.dtie.v66.*` / `experiments.training.v66.*` | **no** for TokyoEye forward (compare baselines via checkpoint load only) |

## First smoke (later pre-reg)

Compare non-local graft/depth response of Hyp MP TokyoEye vs frozen v66 euc-trunk champion as external baseline. New bars; no chem-MVP theater.
