# V6.5 GNN lineage — lifecycle-managed architecture fork

**Status:** Active development line (forked from v6 baseline)  
**Production v6:** Unchanged — `science/dtie/v6/gnn/` remains the frozen inference line until promotion.

## Why v6.5 exists

Editing `science/dtie/v6/gnn/model.py` in place erases rollback capability: every training run implicitly depends on whatever is currently in that file. v6.5 is a **parallel package** so architecture experiments can diverge without touching the production v6 tree.

| Concern | v6 (frozen baseline) | v6.5 (active fork) |
|---------|----------------------|---------------------|
| Model code | `science/dtie/v6/gnn/model.py` | `science/dtie/v65/gnn/model.py` |
| Checkpoints | `checkpoints/v6/runs/<RUN_ID>/v6_*.pt` | `checkpoints/v65/runs/<RUN_ID>/v65_*.pt` |
| MLflow experiment | `tokyo-eyes-v6` | `tokyo-eyes-v65` |
| Inference runner | `V6GNNRunner` (production) | Not wired to ingest until promotion |

## Lifecycle rules

1. **Do not edit v6 model code** for new architecture work — edit `science/dtie/v65/gnn/model.py` instead.
2. **Cold starts** on v6.5 use `--no-warm-start` and a fresh `RUN_ID` under `checkpoints/v65/runs/`.
3. **Promotion** to production is explicit: copy `v65_best.pt` → registry path only after assess gates pass (same as v6 promote flow).
4. **Rollback:** `git checkout` the v65 package or resume from a prior `checkpoints/v65/runs/<RUN_ID>/` snapshot.

## Training commands

```bash
# Cold start — slim MoE + frozen structural disc (recommended first v6.5 run)
make train-v65-cold-start RUN_ID=cold_v1 DEVICE=cuda

# Generic v6.5 launcher (inherits all v6 flags)
make train-v65 RUN_ID=my_run STAGE=1 DEVICE=cuda
```

MLflow UI (`make mlflow-ui` or :5000) lists `tokyo-eyes-v65` separately from historical `tokyo-eyes-v6` runs.

## Forking additional modules

Today only `model.py` is forked. Shared modules (`hyperbolic_moe.py`, `loss.py`) still import from v6 until v6.5 diverges. When you need to change MoE gating:

```bash
cp science/dtie/v6/gnn/hyperbolic_moe.py science/dtie/v65/gnn/hyperbolic_moe.py
# update imports in v65/gnn/model.py
```

## Registry

Lineage metadata lives in `science/training/gnn_lineage.py` (`LINEAGE_REGISTRY`).
