# Hyperbolic graph vs Cα `edge_index` — consumer checklist

Structural SSOT attaches a **hyperbolic k-NN graph** on explicit PyG fields. The Cα
contact graph remains on `edge_index` / `edge_attr` unchanged.

| Field | Semantics |
|-------|-----------|
| `edge_index`, `edge_attr` | Cα contact graph (8Å), Euclidean rel pos + distance |
| `hyperbolic_edge_index`, `hyperbolic_edge_attr` | k-NN in Poincaré distance; tangent `log_p` + geodesic dist |
| `hyperbolic_graph` | `True` when hyperbolic edges are attached |
| `hyperbolic_degree`, `degree` | k-NN degree when `hyperbolic_graph` (gate `log_degree_norm`) |
| `clustering` | Still from Cα `edge_index` via `precompute_clustering` (known mixed topology) |

## Consumers audited (2026-07)

| Location | Uses | Action |
|----------|------|--------|
| `science/dtie/v6/gnn/model.py` `resolve_message_passing_edges` | MP edges | **Uses hyperbolic when `hyperbolic_graph`** |
| `science/dtie/v5/gnn/model.py` `precompute_clustering` | `edge_index` | Cα — intentional |
| `science/dtie/v6/gnn/runner.py` `_ensure_v6_features` | `edge_index` for degree | Skipped when `degree` set by attach |
| `science/dtie/common/graph_builder.py` | builds Cα | SSOT builder — unchanged |
| `science/dtie/common/load_graph_from_db.py` | Cα from DB | unchanged |
| `science/dtie/v5/resistance/profiler.py` | `edge_index` cache | Cα — correct for resistance |
| `science/dtie/v5/resistance/operator.py` | mutates `edge_attr` | Cα — correct |
| `science/dtie/v5/resistance/analysis/graph_comparer.py` | contact graph | Cα — correct |
| `science/dtie/common/compare_graphs.py` | `edge_index` | Cα — correct |
| v3/v4 GNN models | `edge_index` | legacy paths — no structural SSOT |
| `agent/pipeline/*` | legacy payloads | not on structural SSOT path |

**Rule:** Anything that needs **spatial Cα contacts** reads `edge_index`. The v6 encoder
message-passing path reads `hyperbolic_edge_*` via `resolve_message_passing_edges`.

## Curvature provenance

See **`docs/audit/CURVATURE_CONSUMERS.md`** (authoritative). Summary:

- **Structural compose + hyperbolic k-NN edges** today both use **`model.curvature`**
  passed at attach (same c per forward; not the 0.7026 pin on slim MoE path).
- **`CANONICAL_V6_CURVATURE`** (0.7026…) is the DB/disc pin; checkpoint c may differ.
- **`hyperbolic_utils.hyperbolic_dist0(..., c=1.0)`** default is legacy-only; structural
  paths pass explicit c via `require_learned_curvature`.

## σ₂/σ₁ convention

`sigma_ratio[1] = σ₂/σ₁` from centered SVD. **Higher is healthier** (→1 good, →0 streak).
Do not label high values (e.g. 0.95 on 1MBN) as rank-1 collapse. See
[`CURVATURE_CONSUMERS.md`](CURVATURE_CONSUMERS.md).

PyG attach must clone before multi-checkpoint compares — see
[`PYG_ATTACH_MUTATION.md`](PYG_ATTACH_MUTATION.md).
