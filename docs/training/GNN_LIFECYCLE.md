# GNN Model Lifecycle — operator guide

Primary surface: **localhost:3000** → activity bar **Model Lifecycle**.

Backend: MLflow 3.x server (`http://localhost:5000`) with Postgres `mlflow` DB + disk artifacts under `mlflow-artifacts/` and `checkpoints/`.

## Quick status

```bash
make lifecycle-status
# or
curl -s http://localhost:8000/api/lifecycle/status | jq .
```

## Promote (CI / power user)

```bash
make promote-challenger LINEAGE=v6.5 CHECKPOINT=checkpoints/v65/runs/cold_v1/v65_best.pt
make promote-champion LINEAGE=v6.5 CHECKPOINT=checkpoints/v65/runs/cold_v1/v65_best.pt
```

Cockpit panel calls the same helpers via `/api/lifecycle/register` and `/api/lifecycle/promote`.

## Train

Prefer the Lifecycle panel **Queue cold start**, or:

```bash
make train-v65-cold-start RUN_ID=cold_v1 DEVICE=cuda
```

## RCSB → corpus

1. Structure picker → RCSB → Load & run (ingest).
2. Lifecycle → **Add current structure to corpus**.

## MLflow in the workbench

Vite proxies `/mlflow` → `:5000`. Open from the Lifecycle panel or spawn Dockview `mlflow-panel`.
