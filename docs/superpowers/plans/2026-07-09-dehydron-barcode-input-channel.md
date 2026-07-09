# Dehydron Barcode Input Channel (P1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a cacheable Euclidean dehydron-midpoint witness-persistence channel to training graphs (scalars always-on; binned vector optional) and run a baseline vs scalars vs full ablation without changing the slim MoE recipe.

**Architecture:** Precompute per-chain barcode tensors from PDB geometry (GUDHI witness complex on underwrapped backbone H-bond midpoints). Store sidecars + bump corpus graph cache. Concatenate features onto `topology_three_vector` `data.x` behind config flags. Evaluate with the existing investigation audit metrics.

**Tech Stack:** Python 3.10+, NumPy, SciPy, GUDHI 3.10.x, PyTorch / PyG, existing `residue_features` + v6/v65 training launchers.

**Design spec:** [`docs/specs/dehydron-barcode-input-channel/design.md`](../../specs/dehydron-barcode-input-channel/design.md)

## Global Constraints

- Filtration SSOT = Euclidean dehydron-midpoint **witness** persistence (option A); no hyperbolic barcodes as training input.
- Payload = scalars always when enabled; binned only if `use_binned_dehydron=true` (option C).
- No GUDHI inside the training epoch loop — cache read-only.
- No MoE timeout/freeze, cone-loss, or structural-disc unfreeze changes in this plan.
- No Normalizer / onboard-contract promotion in P1.
- No dehydron-typed edges in P1.
- Feature version string: `dehydron_barcode_v1`.
- Locked defaults: `max_alpha_angstrom=20.0` → `max_alpha_square=400.0`; `min_persistence_angstrom=0.1`; `long_lived_persistence_angstrom=2.0`; `n_landmarks=30`; binned = H1 persistence histogram, 40 bins × 0.25 Å over `[0, 10)` Å.
- Warm-start when expanding `node_emb`: copy first 3 input columns from checkpoint; zero-init new columns (do not silently load mismatched `node_dim`).

---

## File structure

| File | Responsibility |
|------|----------------|
| `science/dtie/common/dehydron_barcode_features.py` | Midpoints, witness persistence, per-residue scalars/binned, missing mask |
| `science/dtie/common/residue_features.py` | Optional concat helper / feature-set id extension |
| `experiments/training/v6/precompute_dehydron_barcodes.py` | Corpus sidecar writer |
| `experiments/training/v6/corpus.py` | Cache key bump when barcodes enabled |
| `experiments/training/v6/_data.py` | Attach barcode tensors into protein dict / `data.x` |
| `experiments/training/v6/launch_training.py` (+ v65 if mirrored) | CLI flags |
| `science/training/config.py` / TrainingConfig | `use_dehydron_barcode`, `use_binned_dehydron` |
| `science/dtie/v6/gnn/model.py` (load path only if needed) | Respect `node_dim` from features |
| `Makefile` | `precompute-dehydron-barcodes` target |
| `tests/test_dehydron_barcode_features.py` | Unit + golden dim/mask tests |
| `docs/specs/dehydron-barcode-input-channel/design.md` | Already approved — link only |

---

### Task 1: Dehydron midpoint extraction (true H-bonds)

**Files:**
- Create: `science/dtie/common/dehydron_barcode_features.py`
- Test: `tests/test_dehydron_barcode_features.py`

**Interfaces:**
- Consumes: BioPython/PDB atoms via existing training PDB parse patterns in `experiments/training/v6/_data.py` / `residue_features.build_from_pdb_chain`
- Produces:
  - `BARCODE_FEATURE_VERSION: str = "dehydron_barcode_v1"`
  - `SCALAR_DIM: int = 11` (excludes missing mask)
  - `BINNED_DIM: int = 40`
  - `dataclass DehydronMidpoint` with fields `coord: np.ndarray`, `donor_idx: int`, `acceptor_idx: int`, `wrapping_count: float`
  - `extract_dehydron_midpoints(structure_atoms, residue_index_map, *, wrapping_radius=6.5, tau=13.0, max_no_dist=3.5) -> list[DehydronMidpoint]`

**Notes:** Do **not** use per-residue local N–O midpoints from `compute_dehydron_wrapping_count` as witnesses. Witnesses must be midpoints of **inter-residue backbone H-bonds** (donor N … acceptor O) with wrapping count `< tau` (Fernández dehydrons). Reuse wrapping-count logic around the midpoint (apolar C within 6.5 Å, same exclusions as `compute_dehydron_wrapping_count`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dehydron_barcode_features.py
import numpy as np
from science.dtie.common.dehydron_barcode_features import extract_dehydron_midpoints

def test_extract_midpoints_requires_inter_residue_hbond():
    # Minimal synthetic: two residues with N/O close enough; wrapping left high → empty
    # Implement using lightweight fake atom records matching extract_dehydron_midpoints' expected type.
    midpoints = extract_dehydron_midpoints(
        atoms=[],  # replace with fixture atoms in implementation
        residue_index_map={},
    )
    assert isinstance(midpoints, list)
```

Expand the fixture in implementation so at least one underwrapped H-bond yields a midpoint with `donor_idx != acceptor_idx`.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dehydron_barcode_features.py::test_extract_midpoints_requires_inter_residue_hbond -v`  
Expected: FAIL (module/import or empty implementation)

- [ ] **Step 3: Implement `extract_dehydron_midpoints`**

Implement in `science/dtie/common/dehydron_barcode_features.py`:
- Enumerate candidate donor residues (backbone N) and acceptor residues (backbone O) with `|i-j| >= 1`
- Keep pairs with N–O distance `< max_no_dist` (default 3.5 Å)
- Midpoint = `(N_coord + O_coord) / 2`
- Wrapping count = same carbon-shell rule as `compute_dehydron_wrapping_count`
- Keep only `wrapping_count < tau` (default 13.0)
- Map donor/acceptor to residue indices used by the training graph (Cα residue order)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dehydron_barcode_features.py::test_extract_midpoints_requires_inter_residue_hbond -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add science/dtie/common/dehydron_barcode_features.py tests/test_dehydron_barcode_features.py
git commit -m "$(cat <<'EOF'
feat(barcode): extract underwrapped backbone H-bond midpoints

EOF
)"
```

---

### Task 2: Witness persistence + barcode bars

**Files:**
- Modify: `science/dtie/common/dehydron_barcode_features.py`
- Test: `tests/test_dehydron_barcode_features.py`

**Interfaces:**
- Consumes: `list[DehydronMidpoint]`
- Produces:
  - `dataclass PersistenceBar` with `dim: int`, `birth: float`, `death: float`, `persistence: float`
  - `compute_witness_persistence(midpoints: list[DehydronMidpoint], *, max_alpha_angstrom=20.0, min_persistence_angstrom=0.1, n_landmarks=30, random_state=42) -> list[PersistenceBar]`
  - Empty midpoints → empty bar list (no throw)

**Notes:** Adapt v3 `WitnessComplex` + `compute_nearest` pattern. Landmarks = k-means on midpoint coords (`n_clusters=min(n_landmarks, n_witnesses)`). Pass `max_alpha_square = max_alpha_angstrom ** 2` (fix v3’s ambiguous `max_alpha_square=max_alpha` naming). Filter bars with `persistence < min_persistence_angstrom`. Finite death only for scalar stats; treat `inf` death as `max_alpha_angstrom` for persistence length.

- [ ] **Step 1: Write the failing test**

```python
def test_witness_persistence_empty_midpoints():
    from science.dtie.common.dehydron_barcode_features import compute_witness_persistence
    assert compute_witness_persistence([]) == []

def test_witness_persistence_returns_nonneg_persistence(simple_midpoints):
    from science.dtie.common.dehydron_barcode_features import compute_witness_persistence
    bars = compute_witness_persistence(simple_midpoints, max_alpha_angstrom=20.0)
    assert all(b.persistence >= 0.0 for b in bars)
    assert all(b.dim in (0, 1) for b in bars)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dehydron_barcode_features.py -k witness_persistence -v`  
Expected: FAIL

- [ ] **Step 3: Implement `compute_witness_persistence`**

Use GUDHI `WitnessComplex`, `limit_dimension=2`, `homology_coeff_field=2`. Skip landmark k-means when `len(midpoints) < 2` (return `[]`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dehydron_barcode_features.py -k witness_persistence -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add science/dtie/common/dehydron_barcode_features.py tests/test_dehydron_barcode_features.py
git commit -m "$(cat <<'EOF'
feat(barcode): Euclidean witness persistence on dehydron midpoints

EOF
)"
```

---

### Task 3: Per-residue scalars, binned histogram, missing mask

**Files:**
- Modify: `science/dtie/common/dehydron_barcode_features.py`
- Test: `tests/test_dehydron_barcode_features.py`

**Interfaces:**
- Produces:
  - `SCALAR_NAMES: list[str]` length 11 (document order in module docstring)
  - `aggregate_residue_barcode_features(n_residues, midpoints, bars, *, long_lived_persistence_angstrom=2.0, use_binned=False, bin_width=0.25, bin_max=10.0) -> dict`
    - returns `{ "scalars": Float32[N,11], "binned": Float32[N,40]|None, "missing": Float32[N,1] }`
  - Assignment rule: the structure-level barcode summary is assigned to residues touched by any midpoint (`donor_idx` and/or `acceptor_idx`)
  - Residue with no touching dehydrons: scalar row zeros, `missing=1`
  - Structure with zero midpoints / failed TDA: all rows zero, `missing=1` for all residues
  - Structure-level WitnessComplex does not produce true per-midpoint bar sets; scalars 0–9 and binned histograms are broadcast structure-global summaries, while `n_dehydrons_touching` remains local
  - Apply `log1p` to count and total-persistence scalars before return (document in metadata)

**Locked scalar order (11):**
0. `n_bars` (log1p)
1. `n_h1_bars` (log1p)
2. `total_persistence` (log1p)
3. `max_persistence`
4. `mean_persistence`
5. `std_persistence`
6. `frac_long_lived`
7. `mean_birth_h1`
8. `mean_death_h1`
9. `max_h1_persistence`
10. `n_dehydrons_touching` (log1p)

Binned: H1 persistence histogram, 40 bins on `[0, 10)` Å, L1-normalize per residue when any mass; else zeros.

- [ ] **Step 1: Write the failing tests**

```python
def test_aggregate_missing_mask_when_no_midpoints():
    from science.dtie.common.dehydron_barcode_features import aggregate_residue_barcode_features
    out = aggregate_residue_barcode_features(5, [], [])
    assert out["scalars"].shape == (5, 11)
    assert out["binned"] is None
    assert out["missing"].shape == (5, 1)
    assert np.allclose(out["missing"], 1.0)

def test_aggregate_binned_shape_when_enabled(simple_midpoints, simple_bars):
    from science.dtie.common.dehydron_barcode_features import aggregate_residue_barcode_features
    out = aggregate_residue_barcode_features(10, simple_midpoints, simple_bars, use_binned=True)
    assert out["binned"].shape == (10, 40)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dehydron_barcode_features.py -k aggregate -v`  
Expected: FAIL

- [ ] **Step 3: Implement aggregation**

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dehydron_barcode_features.py -k aggregate -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add science/dtie/common/dehydron_barcode_features.py tests/test_dehydron_barcode_features.py
git commit -m "$(cat <<'EOF'
feat(barcode): per-residue dehydron scalar and binned features

EOF
)"
```

---

### Task 4: End-to-end featurizer for one chain + golden dim test

**Files:**
- Modify: `science/dtie/common/dehydron_barcode_features.py`
- Test: `tests/test_dehydron_barcode_features.py`

**Interfaces:**
- Produces:
  - `featurize_chain_dehydron_barcode(pdb_path: Path, chain: str, *, use_binned=False, **kwargs) -> dict`
    - keys: `scalars`, `binned`, `missing`, `metadata` (dict with version, params, `n_midpoints`, `n_bars`)
  - `stack_node_features_with_barcode(base_x: np.ndarray, barcode: dict, *, use_binned=False) -> np.ndarray`
    - `base_x` is `[N,3]` topology-three-vector
    - output `[N, 3+11+1]` or `[N, 3+11+40+1]`

- [ ] **Step 1: Write failing tests for stack dims**

```python
def test_stack_dims_scalars_only():
    from science.dtie.common.dehydron_barcode_features import stack_node_features_with_barcode
    x = np.zeros((8, 3), np.float32)
    barcode = {
        "scalars": np.zeros((8, 11), np.float32),
        "binned": None,
        "missing": np.ones((8, 1), np.float32),
    }
    y = stack_node_features_with_barcode(x, barcode, use_binned=False)
    assert y.shape == (8, 15)

def test_stack_dims_full():
    from science.dtie.common.dehydron_barcode_features import stack_node_features_with_barcode
    x = np.zeros((8, 3), np.float32)
    barcode = {
        "scalars": np.zeros((8, 11), np.float32),
        "binned": np.zeros((8, 40), np.float32),
        "missing": np.zeros((8, 1), np.float32),
    }
    y = stack_node_features_with_barcode(x, barcode, use_binned=True)
    assert y.shape == (8, 55)
```

- [ ] **Step 2: Run to verify fail; implement; run to pass**

- [ ] **Step 3: Optional smoke on real PDB if cache present**

Run (skip if PDB missing):  
`pytest tests/test_dehydron_barcode_features.py -k 4obe -v`  
Use `pdb_cache/4obe.pdb` or corpus path if available; assert `n_midpoints >= 1` and finite features.

- [ ] **Step 4: Commit**

```bash
git add science/dtie/common/dehydron_barcode_features.py tests/test_dehydron_barcode_features.py
git commit -m "$(cat <<'EOF'
feat(barcode): chain featurizer and node-feature stacking

EOF
)"
```

---

### Task 5: Precompute sidecars + Makefile target

**Files:**
- Create: `experiments/training/v6/precompute_dehydron_barcodes.py`
- Modify: `Makefile`
- Modify: `experiments/training/v6/corpus.py` (cache key includes barcode version when enabled)

**Interfaces:**
- CLI: `python -m experiments.training.v6.precompute_dehydron_barcodes --manifest ... --pdb-dir ... --out-dir ... [--binned]`
- Writes: `{out_dir}/{pdb_id}_{chain}_dehydron_barcode_v1.pt` with tensors + metadata
- Makefile: `precompute-dehydron-barcodes`

**Corpus cache key:** append `|dbh_v1` or `|dbh_v1_binned` when training flag enabled so old `graphs_*.pt` are not reused with wrong `data.x` width.

- [ ] **Step 1: Implement precompute script**

Loop Stage A small manifest entries; call `featurize_chain_dehydron_barcode`; `torch.save`.

- [ ] **Step 2: Add Makefile target**

```makefile
precompute-dehydron-barcodes: ## Cache dehydron_barcode_v1 sidecars for Stage A small corpus
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v6.precompute_dehydron_barcodes \
		--manifest /app/manifests/v6_corpus_stage_a_small_v1.json \
		--pdb-dir /tmp/dtie_pdb_cache \
		--out-dir /app/checkpoints/v65/dehydron_barcode_v1
```

- [ ] **Step 3: Bump corpus cache key when barcodes enabled**

In `experiments/training/v6/corpus.py`, include a barcode tag from config/env in the sha256 input string.

- [ ] **Step 4: Smoke precompute on one structure (local or docker)**

Run: `make precompute-dehydron-barcodes` (or python module for a single PDB first)  
Expected: sidecar files written; log `n_midpoints` per chain.

- [ ] **Step 5: Commit**

```bash
git add experiments/training/v6/precompute_dehydron_barcodes.py experiments/training/v6/corpus.py Makefile
git commit -m "$(cat <<'EOF'
feat(barcode): precompute dehydron barcode sidecars for training corpus

EOF
)"
```

---

### Task 6: Wire into training graph assembly + config flags

**Files:**
- Modify: `science/dtie/common/residue_features.py` (`gnn_feature_set_id` variants)
- Modify: `experiments/training/v6/_data.py`
- Modify: `science/training/config.py` (`TrainingConfig` fields)
- Modify: `experiments/training/v6/launch_training.py` and `experiments/training/v65/launch_training.py` (CLI)
- Modify: model init path to pass `node_dim=data.x.size(1)` (already mostly true — verify)

**Interfaces:**
- `TrainingConfig.use_dehydron_barcode: bool = False`
- `TrainingConfig.use_binned_dehydron: bool = False`
- `TrainingConfig.dehydron_barcode_dir: Path | None = None`
- Feature-set ids:
  - baseline: `master_topology_three_vector`
  - scalars: `master_topology_three_vector_dbh_scalars_v1`
  - full: `master_topology_three_vector_dbh_full_v1`
- When loading protein dict, if flag on: load sidecar, `stack_node_features_with_barcode`, set `prot["data"].x`
- If sidecar missing: zeros + missing=1 (log warning once per structure)

**Warm-start rule:** If resume checkpoint `node_emb.weight.shape[1] != new node_dim`, resize: copy overlapping columns, zero-fill new input features; log clearly. Prefer documenting this in launch logs.

- [ ] **Step 1: Add config + CLI flags**

- [ ] **Step 2: Attach barcodes in `_data.py` / corpus load path**

- [ ] **Step 3: Unit test feature-set id strings**

```python
def test_feature_set_ids():
    from science.dtie.common.residue_features import gnn_feature_set_id_for_barcode
    assert "dbh_scalars_v1" in gnn_feature_set_id_for_barcode(use_barcode=True, use_binned=False)
    assert "dbh_full_v1" in gnn_feature_set_id_for_barcode(use_barcode=True, use_binned=True)
```

- [ ] **Step 4: Commit**

```bash
git add science/dtie/common/residue_features.py experiments/training/v6/_data.py science/training/config.py experiments/training/v6/launch_training.py experiments/training/v65/launch_training.py tests/test_dehydron_barcode_features.py
git commit -m "$(cat <<'EOF'
feat(barcode): wire dehydron barcode flags into training graph assembly

EOF
)"
```

---

### Task 7: Ablation launch recipe + evaluation checklist

**Files:**
- Modify: `Makefile` (three documented targets or one parameterized)
- Create: `docs/specs/dehydron-barcode-input-channel/ablation.md` (short runbook)
- Reuse: investigation audit script/approach from `checkpoints/v65/runs/cold_start_v8_p3e/investigation_audit_corpus12.json` workflow

**Ablation arms (matched epochs, fixed MoE recipe):**

| Arm | Flags | Resume / init |
|-----|-------|----------------|
| Baseline | no barcode | `cold_start_v8_p3e/v65_best.pt` or p2 best |
| Scalars | `--use-dehydron-barcode` | same + node_emb resize |
| Full | `--use-dehydron-barcode --use-binned-dehydron` | same + node_emb resize |

Suggested: **15–20 epoch** P3-style continue, slim MoE unchanged, `GNN_INPUT_MODE=topology_three_vector`.

**Pass criteria (from design):**
- 4OBE motifs retained/improved on investigation
- Corpus-12 rim enrichment ≥11/12
- Cone/τ probes not regressing
- No MoE starvation attributable to features

- [ ] **Step 1: Write ablation runbook** (`docs/specs/dehydron-barcode-input-channel/ablation.md`) with exact `make` commands and success checklist

- [ ] **Step 2: Add Makefile helpers** e.g. `train-v65-dbh-scalars`, `train-v65-dbh-full`

- [ ] **Step 3: Dry-run launch help**

Run: `python -m experiments.training.v65.launch_training --help | rg dehydron`  
Expected: flags listed

- [ ] **Step 4: Commit**

```bash
git add Makefile docs/specs/dehydron-barcode-input-channel/ablation.md
git commit -m "$(cat <<'EOF'
docs(barcode): add dehydron barcode ablation runbook and make targets

EOF
)"
```

---

## Spec coverage self-check

| Spec requirement | Task |
|------------------|------|
| Euclidean dehydron-midpoint witness (A) | Tasks 1–2 |
| Scalars always / binned optional (C) | Tasks 3–4, 6 |
| Missing mask | Task 3 |
| Cache / no GUDHI in epoch loop | Task 5 |
| Node concat only (no edges) | Task 6 |
| Feature-set id / node_dim | Task 6 |
| Ablation baseline/scalars/full | Task 7 |
| No MoE/loss/ingest promotion | Global constraints |
| Hyperbolic B deferred | Global constraints / non-goals |

## Placeholder scan

No TBD/TODO left in task steps; open numeric params locked in Global Constraints.

## Type consistency

- `SCALAR_DIM=11`, `BINNED_DIM=40`, missing `[N,1]`
- Stacked dims: 15 (scalars) or 55 (full)
- Version `dehydron_barcode_v1` used in sidecars, cache key, feature-set ids

---

**Plan complete and saved to `docs/superpowers/plans/2026-07-09-dehydron-barcode-input-channel.md`.**

Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks  
2. **Inline Execution** — execute tasks in this session with checkpoints  

Which approach?
