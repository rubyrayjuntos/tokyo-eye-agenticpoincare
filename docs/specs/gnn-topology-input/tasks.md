# GNN Topology Input — Tasks

**Spec:** [`requirements.md`](requirements.md) · **Design:** [`design.md`](design.md)

---

## Phase 1 — Strip SASA from GNN forward (this PR)

- [x] `GnnInputMode` + `stack_gnn_node_features()` in `residue_features.py`
- [x] Graph builders: `load_graph_from_db`, `graph_builder`, `_data.py` → `data.x` 3-D + `data.sasa`
- [x] `train_loop.py` SASA probes via `residue_sasa_from_data()`
- [x] `launch_training.build_model()` → `node_dim=gnn_input_dim()`
- [x] `infer_v6_model_kwargs()` infers `node_dim` from checkpoint
- [x] MLflow `feature_set` from `gnn_feature_set_id()`
- [x] Cold-start Makefile targets export `GNN_INPUT_MODE=topology_three_vector`
- [x] Unit tests `tests/test_gnn_topology_input.py`
- [x] `.env.example` documents `GNN_INPUT_MODE`

## Phase 2 — Cold lineage restart

- [ ] Stop `master_cold_dehydron_v1` (4-D, SASA shell losses active in early epochs)
- [ ] Launch `RUN_ID=master_cold_topology_v1 make train-v6-stage-a-small-master-cold`
- [ ] Verify `node_emb.weight.shape[1] == 3` and `P_DEHYDRON_CONE_01` passes on τ alone
- [ ] `make audit-dehydron-topology` pass on topology lineage checkpoint

## Phase 3 — Binding-site accessibility (follow-on)

- [ ] Join `dim_residue.sasa` in cryptic scan rank/filter (not GNN `node_emb`)
- [ ] Document threshold in binding-site-scan spec

## Out of scope

- In-place migration of `tokyo_eyes_v6.pt` (stays `legacy_four_vector` / `node_dim=4`)
- Removing FreeSASA from ingest
