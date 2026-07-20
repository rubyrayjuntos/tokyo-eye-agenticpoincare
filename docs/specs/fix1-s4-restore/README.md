# Fix-1 + S4 healthy lineage restore

**Date:** 2026-07-19  
**Status:** Active SSOT for v6.6 biology / routing trunk  
**Replaces:** chem-MVP as default substrate ([`../chem-mvp-reengage/README.md`](../chem-mvp-reengage/README.md) → PARKED)

---

## Active trunk

| Item | Value |
|------|--------|
| Run | `fix1_s4_stack_initseed_controlled_3d_seed2_v1` |
| **Sealed checkpoint** | `checkpoints/v66/runs/fix1_s4_stack_initseed_controlled_3d_seed2_v1/v66_healthy_sealed.pt` |
| Python constant | `experiments.training.v66.healthy_fix1.HEALTHY_FIX1_CKPT` |
| Lineage | v6.6 cold Fix-1 + S4 (not a v6 weight alias) |
| Epoch | 30 |

**Do not use as default load:**

- Bare `phase_12.pt` — weights only; mis-reconstructs geom prior / rim `min_r` / `hyperbolic_mp_graph`
- `v66_best_disc.pt` — ep7 origin-collapsed saver

Re-seal if needed:

```bash
PYTHONPATH=. python -m experiments.training.v66.seal_healthy_fix1
```

---

## Independence (locked)

- Entrypoint: `experiments.training.v66.launch_training` (no import of v6/v65 packages)
- Cold: `--no-warm-start`, `resume=None`
- Model: `GOSPConeMapperV66`, `architecture.version=v6.6`
- Shared `science/training/` and `v6_corpus_*` manifest names are platform/data — not weight tethers

---

## Train recipe (recommended)

```bash
make train-v66-fix1-healthy-restore
```

Wraps `train-v66-fix1-s4-proto-repulsion-scale-l2-stage-a12` with:

- `SEED=2`, `EPOCHS=30`, `GNN_INPUT_MODE=topology_three_vector`
- Default `RUN_ID=fix1_s4_stack_initseed_controlled_3d_seed2_restore_v1` (never overwrite the sealed run dir)
- Flags: feeler lineage + rim-fanout model + geom angular prior + hyp-MP + T1a z-norm + gate SASA + prototype repulsion + softplus floor ≈6.612

Intentional override vs chem-MVP-era “no hyp-MP next bet”: this restores the known-good Fix-1 trunk, not a new hyp-MP experiment.

---

## Parked (compare-only)

Chem-MVP and successors (containment, HA, euc-reach, z-norm/τ-swap arms) remain on disk for historical compare. Makefile targets STOP unless `ALLOW_PARKED=1`.

Known limitation on this healthy run: dehydron partition purity was `DEHYDRON_PARTITION_BLURRED` — restore is for disc fill + biology-routing behavior, not a purity win.

---

## Next: corpus expansion + sparsity bet

Staged ladder + MLflow lineage `tokyo-eyes-v66-fix1-expand`:  
[`corpus-expansion.md`](corpus-expansion.md)

Mean-residue routing entropy sparsity (sealed continue only):

```bash
make train-v66-fix1-sparsity-sealed-continue
# default RUN_ID=fix1_s4_sparsity_sealed_continue_v1
```

Design: [`../routing-entropy-sparsity/design.md`](../routing-entropy-sparsity/design.md)
