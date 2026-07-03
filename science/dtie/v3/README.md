# V3 teacher (v2-bridge)

Frozen **GOSPConeMapper** scaffold used as a distillation teacher for v6 training.
Not used in production ingest or agent inference — v6 is the production model.

## Layout

| Path | Purpose |
|------|---------|
| `gnn/model.py` | Vendored Tokyo Eyes v3 architecture (`hyper_lift`, `mobius_proj`, evidential head) |
| `checkpoints/v2_bridge_epoch_014.pt` | v2-bridge warmstart diversity run, epoch 14 (shell teacher) |

## Usage

Loaded automatically by `experiments/training/v6/v2_teacher.py` when `--v2-teacher-checkpoint` is omitted.

```bash
make train-v6-benchmark RUN_ID=my_run EPOCHS=5 MAX_PROTEINS=18 DEVICE=cuda
```

## Provenance

Checkpoint from `v2_bridge_warmstart_diversity` training (physical router, v2 warm-start).
Do **not** use `phase1l_blackout_111_120` for shell teacher — that run blocks expert routing.
