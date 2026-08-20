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

**Prerequisite (2026-07-16):** two silent overrides of `topology_three_vector` were found:

1. **Env forward:** Makefile prefix did not enter the container → fixed
   (`SCIENCE_RUN -e GNN_INPUT_MODE` + compose interpolation).
2. **Corpus cache:** `graphs_*.pt` built as 4-D; `launch_training` sized
   `node_emb` from `data.x.size(1)`, overriding env. Prior
   `master_cold_topology_v1` and the aborted
   `master_cold_topology_three_vector_v1` (first attempt) both got
   `node_emb (128,4)`. Fixed: cache key includes `gnn_feature_set_id()`;
   `_resolve_training_node_dim` fail-fast if loaded ≠ expected.

Do **not** treat `master_cold_topology_v1` as Phase 2 complete.

Confirm during train: log line `Training node_dim=3` and after first snapshot
`node_emb.weight.shape[1] == 3`.

Phase 2 expectation: **width 4→3** (+ residual effect of removing z-normed SASA
from `node_emb`). `shell_corr_*_sasa` already 0 on feeler/stack; not the main delta.

- [x] Document false start `master_cold_topology_v1` (4-D)
- [x] Fix env forward + cache-key / fail-fast dim check
- [x] Launch `RUN_ID=master_cold_topology_three_vector_v1 make train-v6-stage-a-small-master-cold` (clean rebuild; log `/tmp/master_cold_topology_three_vector_v1.log`)
- [x] Verify early: `Training node_dim=3`, cache `graphs_ad76f4be916bce40.pt`, `epoch_001` `node_emb.weight` **(128, 3)**
- [x] Train finished (exit 0): 200 ep P1→P2; `phase_2.pt` / `epoch_200` still `node_emb` **(128, 3)**; P3 blocked by routing_H stability (max consecutive below 1.21 = 3, need 5)
- [x] Verify `P_DEHYDRON_CONE_01` PASS on `phase_2.pt` (`r(cone_depth,τ)=1.000`; audit JSON under `checkpoints/v6/diagnostics/dehydron_topology/phase_2_alignment.json`)
      — read as **`CONE_GATE_SANITY_WIRING_OK`** only: master-cold `cone_target_mode=tau_dehydron_rim` directly optimizes Pearson(depth, τ); r≈1 is near-tautological, not emergent discovery
- [x] `make audit-dehydron-topology CHECKPOINT=.../phase_2.pt` PASS (`verdict=dehydron_rim_aligned`)
- [x] Trajectory logged (not endpoint-only): `σ₂/σ₁` trough ep35=0.010 → peak ep164=0.921 → end 0.893; `eff_rank` end ≈1.80 matches T1a z-norm ceiling (~1.7–1.8); P2 `corr(H,σ₂/σ₁)≈0.18`
- [x] Routing note: end `route_H≈1.30` in historic 1.28–1.37 band → **`CLEAN_LINEAGE_ROUTING_UNAFFECTED_BY_WIDTH`** (stack levers absent from master-cold)
- [x] **Early Gram + three-vector battery + controlled width×seed — closed into decision (a)**
      (2026-07-16). Cold SEED=1/2 under real `topology_three_vector`; controlled 3d/4d × seed1/2
      with matched `isolated_init`. Sensitivity + commitment verified; purity/Gram **closed as
      not-solved-at-this-scale** (not still open). Conditioned-init / eighth loss-term / #1/#2
      purity lock **retired**. SSOT: `docs/audit/GNNV7_SUCCESS_CRITERIA.md` (WHERE WE ARE).
      - [x] Extra gate closed: T1a `transform_node_features` hard-required width≥4 — fixed to accept 3
        (`science/dtie/common/input_feature_norm.py`); first launch aborted on that; relaunch OK
      - [x] SEED=1 done `fix1_s4_stack_three_vector_cold_seed1_v1`: `node_emb (128,3)` ep1+ep30
      - [x] SEED=2 done `fix1_s4_stack_three_vector_cold_seed2_v1`: `node_emb (128,3)` ep1+ep30
      - [x] Early Gram scored: ge1 cond **2.35 vs 1.91**; ge30 **both** bank-collapse + blur
        (`THREE_VECTOR_BOTH_SEEDS_BANK_COLLAPSE_AND_BLUR`)
      - [x] Controlled width ablation: `NO_CONSISTENT_FOUR_D_WIN`; Gram mid-30s/40s all four
      - [x] Decision (a): bank sens+commit; purity/Gram named limitation + reopen bar; uncertainty parked

**End-state snapshot (ep200):** disc recovered (`σ₂/σ₁≈0.89`, `eff_rank≈1.80`, `ang≈0.20`); expert loads balanced (`≈0.29/0.28/0.22/0.21`); `p2_dehydron_gate_passed=True`; late saves skipped on `routing_H≈1.30` (ceil 1.21).

## Phase 3 — Binding-site accessibility (follow-on)

- [ ] Join `dim_residue.sasa` in cryptic scan rank/filter (not GNN `node_emb`)
- [ ] Document threshold in binding-site-scan spec

## Out of scope

- In-place migration of `tokyo_eyes_v6.pt` (stays `legacy_four_vector` / `node_dim=4`)
- Removing FreeSASA from ingest
