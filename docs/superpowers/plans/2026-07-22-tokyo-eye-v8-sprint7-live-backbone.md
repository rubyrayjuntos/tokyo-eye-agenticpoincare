# TokyoEye-v8 Sprint 7 — Live SE(3)-Lite Bind

> **For agentic workers:** Execute task-by-task.

**Goal:** Make MPtrj weight-bank tensors participate in forward + backward under differential LR.

**Architecture:** `StubEquiformerFrontend.live_backbone` runs one SE(3)-lite pass over R0–R5 using bank `atom_embed` + `blocks[0]` attn/FFN; `--freeze-backbone` restores stub trunks.

**Tech Stack:** PyTorch, PyG scatter, existing v8 harness.

## Global Constraints

- Isolation under `science/tokyo_eye/v8/`
- `lr_backbone=1e-5`, `lr_hyperbolic=3e-4`
- No full EquiformerV3 vendor

---

### Task 1: SE(3)-lite forward on weight bank

**Files:** `science/tokyo_eye/v8/equiformer_frontend.py`, `tests/v8/test_live_backbone.py`

- [x] `live_backbone` flag + lite modules (`seed_proj`, `radial_mlp`, …)
- [x] Forward uses `backbone.atom_embed` and `blocks[0]` source/target/proj/FFN
- [x] Test: live step → bank grad; freeze → stub, no bank grad

### Task 2: Harness wiring

**Files:** `run_v8_experiment.py`, `equiformer_v3_weight_map.json`, `Makefile`

- [x] Pass `edge_index`/`edge_type` into frontend
- [x] `--freeze-backbone` CLI; log `backbone_mode`
- [x] Param groups unchanged (frontend @ 1e-5, spine @ 3e-4)

### Task 3: 4OBE live smoke

- [x] `RUN_NAME=tokyo_eye_v8_4obe_live_backbone make train-v8-experiment`
