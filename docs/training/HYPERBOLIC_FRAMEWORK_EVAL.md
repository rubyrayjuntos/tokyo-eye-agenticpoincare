# Hyperbolic Framework Evaluation — V6 GNN Lifecycle

This document records the parallel spike comparing **geoopt-native** training (Track A) with **selective HyperCore adoption** (Track B) for the Tokyo Eyes v6 GOSPConeMapper lifecycle.

## Context

| Framework | Role in this repo |
|-----------|-------------------|
| **PyTorch Geometric + geoopt** | Production stack — `GOSPConeMapperV6`, `RiemannianAdam`, Poincaré `expmap0` / `MobiusLinear` |
| **MLflow** | Experiment tracking (`science/training/tracking.py`) |
| **HyperCore** (`pip install hypcore`) | Optional spike — hyperbolic optimizers / layers |
| **HazyResearch/hgcn** | Reference only — citation-graph HGCN; not used for v6 architecture |

## Track A — Wrap (implemented)

- Launcher: `experiments/training/v6/launch_training.py`
- Config: `science/training/config.py` — 3-phase MoE schedule
- Tracking: MLflow via `science/training/tracking.py` — **metric definitions:** [`MLFLOW_METRICS.md`](MLFLOW_METRICS.md)
- Assessment: `experiments/training/v6/assess_checkpoint.py`
- Promotion: `python -m science.training.promote`

### Success criteria

- Phase schedule runs with `routing_entropy`, `expert_load`, `capacity_loss` logged
- Checkpoint registrable as `candidate` in `onboard_contract.yaml`
- Promotion gate enforces routing entropy ≤ 1.2, no expert starvation, `proj_frac` ≤ 0.95

## Track B — HyperCore spike (implemented)

Script: `experiments/training/v6/spike_hypercore_optimizer.py`

Compares on the same 5-protein subset:

1. `geoopt.optim.RiemannianAdam` (via `build_optimizer` in v5 model)
2. HyperCore optimizer if `hypcore` installed and `hypercore.optim.RiemannianAdam` exists

Results written to `docs/training/spike_hypercore_results.json` after each run.

### Decision gate

| Outcome | Action |
|---------|--------|
| HyperCore stable + faster convergence or lower routing entropy | Adopt HyperCore optimizer incrementally in `build_optimizer` wrapper |
| No measurable benefit or API instability | Stay geoopt-native; use HyperCore as reference for new layer designs |
| Warm-start breakage | Do not adopt — v5 backbone transfer is mandatory |

## API mapping notes

| geoopt (current) | HyperCore (spike) |
|------------------|-------------------|
| `geoopt.optim.RiemannianAdam` | `hypercore.optim.RiemannianAdam` (if present) |
| `geoopt.manifolds.stereographic.math` | HyperCore manifold modules (not swapped in v6 forward pass) |
| `MobiusLinear` in v5/v6 model | HyperCore linear layers — **not** drop-in (different weight layout) |

## hgcn — not adopted

[HazyResearch/hgcn](https://github.com/HazyResearch/hgcn) targets citation-graph node classification. Our v6 model uses SE(3)-equivariant convolutions, dehydron density ρ, and topological MoE — incompatible with HGCN layers without full rewrite.

## Running the lifecycle

All commands run **inside the science container** (CUDA, PyG, geoopt, MLflow):

```bash
# Rebuild science image after dependency changes
make build-science

# Smoke test: Phase 1, 5 proteins
make train-v6 STAGE=1 MAX_PROTEINS=5

# Assess checkpoint (container path)
make assess-v6 CHECKPOINT=/app/checkpoints/v6/runs/<run_id>/v6_best.pt MAX_PROTEINS=5

# Promote (contract registration)
make promote-v6 CHECKPOINT=checkpoints/v6/runs/<run_id>/v6_best.pt

# Full theory test with MLflow (seed baseline for promote-v6-from-run)
make train-v6-theory-test-mlflow RUN_ID=mlflow_theory_baseline DEVICE=cuda

# Promote from MLflow run (after training with MLflow ON)
make promote-v6-from-run MLFLOW_RUN_ID=<run_id> STATUS=production CHECKPOINT_ID=tokyo_eyes_v6

# MLflow UI (see MLFLOW_METRICS.md for chart guide)
make mlflow-ui
```

Or directly:

```bash
docker compose run --rm science python -m experiments.training.v6.launch_training \
  --corpus /app/manifests/v6_corpus_120.json \
  --output-dir /app/checkpoints/v6/runs/smoke \
  --phase 1 --max-proteins 5 --device cuda
```

## References

- [HyperCore GitHub](https://github.com/Graph-and-Geometric-Learning/HyperCore) — MIT, active 2026
- [hgcn GitHub](https://github.com/HazyResearch/hgcn) — NeurIPS 2019 reference
- Internal spec: `docs/specs/v6-moe-specialization/requirements.md`
