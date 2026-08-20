# Learned GNN vs slim MoE / structural SSOT

**Date:** 2026-07-09  
**Status:** Course correction — slim is not the representation-learning path

## Summary

Slim MoE + frozen structural disc SSOT was the default v65 cold start and ingest always attached structural SSOT before forward. That recipe **freezes** `node_emb` / `convs` / radial / angular and **bypasses** message-passing for disc/depth lift. It is appropriate only as an explicit legacy / inference-alignment experiment — **not** for learning hyperbolic representations or dehydron barcode channels.

## What went wrong

1. `make train-v65-cold-start` hard-wired `--slim-moe-structural-ssot`.
2. Lifecycle UI/API defaulted the same preset.
3. Barcode ablation resumed slim parents → barcode columns in `data.x` never entered a trainable, output-connected path (`‖W_barcode‖ = 0` for the whole run).
4. Ingest `PipelineConfig.structural_disc_frozen=True` forced SSOT even for learned checkpoints (e.g. lever_a).

## Correct contract

| Concern | Learned path (default) | Slim / SSOT (legacy) |
|---------|------------------------|----------------------|
| Training flag | `--master-cold-lineage` | `--slim-moe-structural-ssot` |
| Makefile | `train-v65-master-cold` / `train-v65-cold-start` | `train-v65-slim-cold-start` |
| Backbone | trainable | frozen |
| Disc | `gnn_learned` | `structural_ssot_frozen` |
| Barcode | allowed + liveness probe | **refused** (unless `--allow-dead-feature-channel`) |
| Ingest SSOT | follow checkpoint `training_config` | attach SSOT |

## Guardrails

- Launcher refuses `--use-dehydron-barcode` + slim.
- `--feature-liveness-probe` compares barcode vs zeroed barcode and MP input noise.
- Governance emits `structural_disc_frozen`, `disc_layout_source`, `backbone_trainable`.
- `resolve_structural_disc_frozen()` in `science/dtie/common/structural_disc_policy.py` + `job_params` override.
- `make validate-learned-ssot-gate CHECKPOINT=...` before production behavior flips.

## Production flip

Do **not** silently change ingest for unknown checkpoints (legacy default remains SSOT-on). After a learned v65 champion exists, validate Stage A / 4OBE with `job_params.structural_disc_frozen` true vs false, then promote.
