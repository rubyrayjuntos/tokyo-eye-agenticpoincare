# B0 Topology Observation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Forward `@champion` (and Mode C S9 compare-only) on the locked 18-structure topology panel and stamp hygiene + theme-rhyme observations with no biology Pass and no training.

**Architecture:** A manifest plus a small observation library. Load graphs the same way as `export_viewers.py`, run `TokyoEyeV8WithFrontend` eval-only, record H1–H6 from DSSP HELIX/SHEET + Poincaré diagnostics + MoE loads. Theme rhyme is a narrative over within-theme vs across-theme ρ-by-SS, not a gate.

**Tech Stack:** Python, PyTorch, Bio.PDB loader, `science.tokyo_eye.v8`, MLflow optional, pytest.

**Spec:** `docs/superpowers/specs/2026-08-24-tokyoeye-b0-topology-observation-design.md`

## Global Constraints

- Weights frozen. No optimizer. No 25×150 train.
- Θ SSOT: `models:/TokyoEye@champion` via `resolve_alias_checkpoint`. Never hardcode a `.pt` as SSOT.
- Mode C S9 is compare-only; skip with `mode_c_s9_absent` if missing.
- Learned curvature from the loaded module via `require_learned_curvature`. Never hardcode `c`.
- Disc-alone hubs forbidden. Evidential Investigation not the H5 signal. H5 uses dehydron ρ vs deposited HELIX/SHEET.
- Skip failed loads; do not swap a different topology. Theme still grades if ≥2 of 3 loaded.
- NMR `1A5R`: model 1 only (Bio.PDB default). If no CA-complete, skip.
- Do not retarget aliases.

---

### Task 1: Manifest + panel parse

**Files:**
- Create: `manifests/v8_b0_topology_observation_v1.json`
- Create: `experiments/training/v8/b0_topology_observation.py` (`load_panel`)
- Test: `tests/v8/test_b0_topology_observation.py`

**Interfaces:**
- Produces: `load_panel(path) -> list[dict]` with keys `pdb_id`, `chain`, `theme`, `gene`, `flags` (`grasp_cousin`, `nmr_model1`, `enabled`).

- [ ] **Step 1:** Write `test_load_panel_has_six_themes_three_each` asserting 18 enabled rows, unique `(pdb_id, chain)`, themes `{ig_like, lysozyme_like, ubiquitin_grasp, tim_barrel, globin, ploop_ntpase}`.
- [ ] **Step 2:** Implement JSON + `load_panel`.
- [ ] **Step 3:** `pytest tests/v8/test_b0_topology_observation.py::test_load_panel_has_six_themes_three_each -v`

### Task 2: SS class + rhyme narrative (no GPU)

**Files:**
- Modify: `experiments/training/v8/b0_topology_observation.py`
- Test: `tests/v8/test_b0_topology_observation.py`

**Interfaces:**
- Consumes: `science.tokyo_eye.sse_hierarchy.parse_pdb_helix_sheet`
- Produces: `ss_class_for_residue(resseq, chain, ranges) -> "helix"|"sheet"|"coil"`; `rho_by_ss(rho, ss) -> dict`; `theme_rhyme_narratives(per_pdb_rows) -> dict[theme, str]`

- [ ] **Step 1:** Tests: helix range maps to helix; coil default; rhyme vs singleton vs collapse on toy ρ means.
- [ ] **Step 2:** Implement helpers. Globin rhyme uses helix means; other themes use sheet means.
- [ ] **Step 3:** pytest those tests.

### Task 3: Forward observation CLI

**Files:**
- Create: `experiments/training/v8/run_b0_topology_observation.py`
- Modify: `experiments/training/v8/b0_topology_observation.py` (`observe_structure`, `build_stamp`)

**Interfaces:**
- Consumes: `resolve_alias_checkpoint`, `load_structure_batch`, `StubEquiformerFrontend` + `TokyoEyesHyperbolicV8` + `TokyoEyeV8WithFrontend` (same construction as `export_viewers.py` main), `PoincareDiagnosticsEngine`, `binary_auprc`, `require_learned_curvature`
- Produces: stamp dict written to `data/gates/tokyo_eye_v8_b0_topology_observation.json`; optional MLflow run `b0_topology_observation_champion` with no alias move.

- [ ] **Step 1:** Unit-test `build_stamp` with fake rows (no model): hygiene_finite false if any NaN; `biology_pass` key absent or false; theme narratives present.
- [ ] **Step 2:** CLI `--device cuda|cpu --tracking-uri --pdb-dir --out-stamp --compare-mode-c PATH`.
- [ ] **Step 3:** Run pytest for stamp builder.

### Task 4: Execute on champion

- [ ] Fetch PDBs via `ensure_pdb_cached`.
- [ ] `docker compose run --rm --user 1000:1000 -e MLFLOW_TRACKING_URI=http://mlflow:5000 science python -m experiments.training.v8.run_b0_topology_observation ...`
- [ ] Confirm stamp exists, champion sha256 matches `507d54bd6d8fb6c2…`, no alias change (`resolve --alias champion` still v5).

---

After the plan: **inline execution** (user already approved moving forward). 25×150 train remains a later spec.
