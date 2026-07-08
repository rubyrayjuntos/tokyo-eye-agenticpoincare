# GNN Topology Input — Design

**Spec:** [`requirements.md`](requirements.md)

---

## Code map

| Component | Change |
|-----------|--------|
| `science/dtie/common/residue_features.py` | `GnnInputMode`, `stack_gnn_node_features()`, `resolve_gnn_input_mode()` |
| `science/dtie/common/load_graph_from_db.py` | Strip SASA from `x`; attach `data.sasa` |
| `science/dtie/common/graph_builder.py` | `to_pyg(..., gnn_input_mode=)` |
| `experiments/training/v6/_data.py` | PDB-local path uses `stack_gnn_node_features` |
| `experiments/training/v6/train_loop.py` | Read SASA from `data.sasa` (fallback `x[:,3]`) |
| `science/dtie/v6/gnn/model.py` | `node_dim=gnn_input_dim()` on new runs |
| `science/training/mlflow_governance.py` | `feature_set` from `gnn_feature_set_id()` |

---

## `stack_gnn_node_features`

```python
# topology (target)
x = stack_gnn_node_features(rho, tau, ss, sasa=sasa, mode=TOPOLOGY_THREE_VECTOR)
# → [N, 3]

data.sasa = torch.tensor(sasa)  # always side-channel for scan / legacy losses
```

Legacy production path unchanged when `GNN_INPUT_MODE=legacy_four_vector` (default).

---

## Checkpoint compatibility

| Checkpoint | `node_emb` in_features | Env |
|------------|------------------------|-----|
| `tokyo_eyes_v6.pt` | 4 | `legacy_four_vector` (default) |
| cold v6.1+ | 3 | `topology_three_vector` |

`infer_v6_model_kwargs` should read `node_emb.weight.shape[1]` from state dict when present.

---

## Ingest unchanged

`persist_master_features()` still writes all four MASTER fields. P_FEATURE_01 still compares ρ, τ, ss, sasa. Only the **graph assembly** strips column 4 for GNN forward.

---

## Binding scan hook (phase 2)

```sql
-- mean SASA of site residues at rank time
SELECT AVG(r.sasa) FROM dim_residue r WHERE r.residue_id = ANY(:residue_ids)
```

Use in `site_merger` or `scan_phase` druggability heuristic — not in GNN forward.
