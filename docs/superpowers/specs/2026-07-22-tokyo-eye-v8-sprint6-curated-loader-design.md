# TokyoEye-v8 Sprint 6 — Curated PDB Loader + Viewer Export

**Date:** 2026-07-22  
**Status:** APPROVED 2026-07-22 — open points frozen below  
**Lineage:** `science/tokyo_eye/v8/` only (no v7/v66 sealed ckpt / ingest rewrite)  
**Depends on:** Sprint 5 GREEN (`equiformer_v3_weight_map.json` audited; MPtrj bind stable)

---

## Goal

Replace synthetic 24-mer training with **real PDB graphs** so `val_dehydron_auprc` becomes a non-trivial biophysical metric, then export **live Poincaré / structure viewers** from v8 embeddings.

**Progression (locked):** Option **A → B**

| Mode | Default | Purpose |
|------|---------|---------|
| **A** | `4OBE:A` from `pdb_cache/` | Deterministic single-structure smoke; shatter synthetic AUPRC=1.0 |
| **B** | `--manifest manifests/v8_stage_a_small_v1.json` | 12-structure Stage A diversity (enabled entries only) |

---

## Non-goals

- Binding a full EquiformerV3 SE(3) forward (weight bank remains; stub trunks still drive `s,v` until a later sprint).
- Calling `experiments.training.v66.load_training_proteins` or sealed Fix-1 checkpoints.
- Writing through Normalizer / ingest.
- PDB-bind affinity regression (can come later; Sprint 6 labels are **graph-native**).

---

## Contract: curated loader

**Module:** `science/tokyo_eye/v8/loader.py`  
**Harness:** `experiments/training/v8/run_v8_experiment.py` gains `--pdb`, `--chain`, `--manifest`, `--pdb-dir`.

### Initialization (agreed, with schema fixes)

```python
TokyoEyeCuratedDataset(
    pdb_code="4OBE",
    chain="A",
    manifest_path=None,          # if set + exists → Mode B
    pdb_dir="pdb_cache",         # local cache root (docker: /tmp/dtie_pdb_cache or /app/pdb_cache)
)
```

- **Mode A:** `manifest = [{"pdb_id": pdb_code, "chain": chain}]`
- **Mode B:** parse existing corpus JSON `proteins[]` with keys `pdb_id`, `chain`, `enabled`; keep only `enabled: true` (and optionally `stage0` first for smoke ordering).

Do **not** invent a separate flat `{"pdb","chain"}` schema; accept the Stage A manifest shape already in-repo and ship a thin `manifests/v8_stage_a_small_v1.json` that points at / copies the same protein list with a v8 description.

### `get(idx)` pipeline

1. Resolve `pdb_path = pdb_dir / f"{pdb_id}.pdb"`; if missing, download from RCSB into `pdb_dir` (same pattern as v66 `_download_pdb`, vendored lightly under v8 — no import from v66 trainers).
2. `records = parse_residue_records_from_pdb_chain(pdb_path, chain)`
3. `graph = build_r0_r5_graph(records)` → `R0R5GraphResult` (**not** a bare PyG `Data` today).
4. Build harness batch dict (compatible with `run_epoch`):

| Key | Source |
|-----|--------|
| `x` | Cα coords `[N,3]` |
| `edge_index`, `edge_type` | from `R0R5GraphResult` |
| `dehydron_labels` | **R2 incidence**: node touches any `edge_type==R2_DEHYDRON` |
| `sdrp_target` | optional soft target from rim/core heuristic (E0–E3 proxy from radius+R2), or held-out constant until MoE labels exist — **document as heuristic, not ground truth** |
| `mechanism_pos` / `mechanism_neg` | soft scores from dehydron vs wrapped-core neighbors (R1 vs R2), not random |
| `pdb_id`, `chain`, `num_nodes`, `graph_meta` | provenance |

**Label layer (Sprint 6 biophysics — concrete, not placeholders):**

| Proposed name (user draft) | Sprint 6 name | Definition |
|----------------------------|---------------|------------|
| `node_mechanism_labels` | `dehydron_labels` | Binary; R2 incidence (wrap ≤ 1 dehydron gate — addendum §2.7) |
| `matrix_state_labels` | `sdrp_target` (heuristic) | 5-way soft class from local edge mix (rim / core / salt / hydrophobe / neighborhood) — **explicitly experimental**; primary gate metric remains `val_dehydron_auprc` |

Primary success metric for A→B: **`val_dehydron_auprc` drops below 1.0** on 4OBE and stays finite / non-NaN.

### PyG `Dataset` vs plain list

Prefer a thin `TokyoEyeCuratedDataset` implementing `__len__` / `__getitem__` returning the batch dict (or a `Data` with the same fields). Full `torch_geometric.data.Dataset` inheritance is optional; do not require processed/raw folders.

---

## Contract: harness CLI

```bash
# Mode A (default)
make train-v8-experiment   # → --pdb 4OBE --chain A --pdb-dir …

# Mode B
MANIFEST=manifests/v8_stage_a_small_v1.json make train-v8-experiment
```

Flags on `run_v8_experiment.py`:

- `--pdb` / `--chain` / `--pdb-dir` / `--manifest`
- Keep `--smoke` for synthetic 24-mer (regression of Sprint 5 mechanical path)
- Epoch loop: sample one structure per step (A) or round-robin / random over manifest (B)

---

## Phase 2: viewer export

**Module:** `experiments/training/v8/export_viewers.py`  
**Makefile:** `export-v8-viewers`

After a run (or inline `--export-viewers`):

1. Forward `z_hyp` (Poincaré ball) → disc projection `(x_d, y_d)` via standard ball→disc map already used in platform viewers.
2. Color by `dehydron_labels` / predicted risk / radius.
3. Write under `data/local_objects/gnn_viewer/v8/<pdb_id>/`:
   - `*_poincare_disc.html`
   - `*_interactive.html` (NGL) if coords available
   - `viewer_manifest.json`

Reuse HTML templates from `science/dtie/v66/visualization/interactive_viewer.py` **only as copy-adapted helpers under v8** or call pure write functions with v8-built node tables — do **not** load Fix-1 / v7 checkpoints.

---

## Isolation rules

- All new code: `science/tokyo_eye/v8/`, `experiments/training/v8/`, `tests/v8/`, `manifests/v8_*.json`, `checkpoints/v8/`
- Read-only use of `pdb_cache/*.pdb`
- No edits to sealed `HEALTHY_*`, v66 train loops, or onboard contract for this sprint

---

## Acceptance

| Gate | Pass |
|------|------|
| A smoke | `4OBE:A` trains ≥1 epoch; `val_dehydron_auprc < 1.0`; entropy finite |
| B flag | `--manifest` loads ≥2 enabled Stage A structures without code change |
| Isolation | `tests/v8` green; no v66 import in v8 loader |
| Viewers | `make export-v8-viewers` writes disc HTML for last/best ckpt + 4OBE |

---

## Frozen open points (2026-07-22)

1. **SDRP:** Keep 5-way rim/core edge-mix heuristic for `sdrp_target`; **loss weight `sdrp_coeff=0.1`**. Primary train signal is binary dehydron supervision driving `val_dehydron_auprc`.
2. **Batching:** **One graph per step** always (Mode A and B). No disjoint-union collation.
3. **Manifest filter:**
   ```python
   proteins = manifest_data.get("proteins", [])
   enabled_proteins = [p for p in proteins if p.get("enabled", True)]
   ```

---

## Next

Implementation plan: `docs/superpowers/plans/2026-07-22-tokyo-eye-v8-sprint6-curated-loader.md`.
