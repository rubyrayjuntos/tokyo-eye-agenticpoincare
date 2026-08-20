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
2. **Cold starts** on v6.5 use `--no-warm-start` + **`--master-cold-lineage`** (learned MP→geometry→MoE) and a fresh `RUN_ID` under `checkpoints/v65/runs/`.
3. **Slim MoE + structural SSOT** is legacy only (`make train-v65-slim-cold-start`) — it freezes the backbone and is **not** representation learning. See [`docs/audit/LEARNED_GNN_VS_SLIM_SSOT.md`](../../docs/audit/LEARNED_GNN_VS_SLIM_SSOT.md).
4. **Promotion** to production is explicit after assess gates pass.
5. **Rollback:** `git checkout` the v65 package or resume from a prior run snapshot.

## Training commands

```bash
# Recommended first v6.5 run — learned GNN cold start
make train-v65-master-cold RUN_ID=master_cold_v1 DEVICE=cuda
# Alias:
make train-v65-cold-start RUN_ID=master_cold_v1 DEVICE=cuda

# LEGACY slim MoE (frozen disc / frozen backbone) — opt-in only
make train-v65-slim-cold-start RUN_ID=slim_legacy_v1 DEVICE=cuda

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
Note: contract `v65_champion` must be added to the checkpoint catalog before promotion — do not promote slim `cold_start_v8_*` as the learning champion.
