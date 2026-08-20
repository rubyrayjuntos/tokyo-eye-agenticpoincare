# TokyoEye-v8 Sprint 7 — Live SE(3)-Lite Backbone Bind

**Date:** 2026-07-22  
**Status:** APPROVED (Option A Bind-Thin)  
**Depends on:** Sprint 5 weight map + Sprint 6 curated loader

---

## Goal

Connect remapped `mptrj_gradient.pt` bank tensors into a **trainable** forward that emits `(s, v)` for the hyperbolic spine, so dehydron loss gradients update backbone weights under `lr_backbone=1e-5`.

## Non-goals

- Full native EquiformerV3 / spherical-harmonic stack (Option C — later)
- Flag-only “unfreeze” without forward bind (Option B)

## Architecture

```
CA coords + R0–R5 edges
  → seed_proj + atom_embed (bank, live)
  → radial RBF(edge dist) + block0 attn source/target/proj (bank)
  → block0 FFN scalar_in (bank) → s [N,128], v [N,3]
  → RadialAngularProjector → hyp trunk
```

**Constraint:** Bank has no `radial_fourier` keys in MPtrj ckpt — Sprint 7 uses a **small trainable RBF MLP** on Euclidean edge distance (not a fake rename).

## Config

| Knob | Value |
|------|-------|
| `lr_backbone` | `1e-5` |
| `lr_hyperbolic` | `3e-4` |
| `--freeze-backbone` | stub trunks only (Sprint 5/6 behavior) |
| Default | `live_backbone=True` when weight map loads |

## Acceptance

1. With live bind: `backbone.atom_embed.weight.grad` is non-`None` after one train step  
2. `--freeze-backbone`: stub path; bank grads absent  
3. 4OBE run logs `backbone_mode=live_se3_lite` and finite `val_dehydron_auprc`
