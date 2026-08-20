# TokyoEye-v8 Sprint 6 — Curated Loader + Viewer Export

> **For agentic workers:** Execute task-by-task. Checkboxes track progress.

**Goal:** Replace synthetic 24-mer batches with real PDB R0–R5 graphs (default `4OBE:A`), expose `--manifest` for Stage A, and export Poincaré disc HTML from `z_hyp`.

**Architecture:** Isolated `science/tokyo_eye/v8/loader.py` builds harness batches from Bio.PDB + Sprint-2 graph; harness defaults to Mode A, iterates one graph/step; exporter maps ball→disc without v66 imports.

**Tech Stack:** PyTorch, Bio.PDB, urllib RCSB download, MLflow, Makefile/`SCIENCE_RUN`.

## Global Constraints

- v8-only paths; no `load_training_proteins` / sealed Fix-1 / Normalizer writes
- Manifest schema: `proteins[]` with `pdb_id`, `chain`, `enabled`
- `sdrp_coeff=0.1`; primary dehydron BCE + `val_dehydron_auprc`
- One graph per optimizer step
- Default PDB: `4OBE` / chain `A` / `pdb_cache`

---

### Task 1: Manifest + curated loader

**Files:**
- Create: `manifests/v8_stage_a_small_v1.json`
- Create: `science/tokyo_eye/v8/loader.py`
- Test: `tests/v8/test_curated_loader.py`

- [x] Write loader with `ensure_pdb_cached`, `parse_enabled_manifest`, `TokyoEyeCuratedDataset`, `batch_from_records`, R2 dehydron labels, 5-way SDRP heuristic, mechanism soft targets
- [x] Unit-test 4OBE batch: `dehydron_labels` in `{0,1}`, AUPRC labels not all-one, edges present
- [x] Manifest filter keeps only `enabled: true`

---

### Task 2: Harness CLI + loss weights

**Files:**
- Modify: `experiments/training/v8/run_v8_experiment.py`
- Modify: `Makefile` (`train-v8-experiment` passes PDB/MANIFEST)

- [x] Flags: `--pdb`, `--chain`, `--pdb-dir`, `--manifest`; `--smoke` keeps synthetic
- [x] Default (non-smoke): Mode A `4OBE:A`
- [x] Mode B: round-robin one graph/step over enabled proteins
- [x] `loss = dehydron_bce + 0.1 * sdrp + margin + cv`
- [x] Log `data_mode`, `pdb_id`, `n_nodes`, `dehydron_frac`

---

### Task 3: Viewer export

**Files:**
- Create: `experiments/training/v8/export_viewers.py`
- Modify: `Makefile` → `export-v8-viewers`

- [x] `ball_to_disc(z)` with \(x_d=2z_x/(1+\|z\|^2)\), \(y_d=2z_y/(1+\|z\|^2)\) (use full-norm; first two coords)
- [x] Write `data/local_objects/gnn_viewer/v8/<pdb>/` HTML colored by radius, dehydron, risk
- [x] Hook after train optional `--export-viewers`

---

### Task 4: Live 4OBE run

- [x] `PDB=4OBE CHAIN=A RUN_NAME=tokyo_eye_v8_4obe_a make train-v8-experiment`
- [x] Confirm `val_dehydron_auprc < 1.0` on epoch 0+
- [x] `make export-v8-viewers CKPT=... PDB=4OBE` (via `EXPORT_VIEWERS=1`)
