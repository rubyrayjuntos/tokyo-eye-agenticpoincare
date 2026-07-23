# TokyoEye-v8 Sprint 10.1.1 — Production MOL2/SDF Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Spec SSOT:** [`docs/superpowers/specs/2026-07-22-tokyo-eye-v8-sprint10-1-1-production-ligand.md`](../specs/2026-07-22-tokyo-eye-v8-sprint10-1-1-production-ligand.md)

**Goal:** Replace HETATM-bootstrap ligand features with native MOL2/SDF 12-channel chemistry so Core Pearson can clear the provisional **0.40** gate (from 10.1.0 Core \(R \approx 0.366\)).

**Architecture:** Option C resolver (`data/pdbbind/ligands/{id}.mol2` → `.sdf` → HETATM). Native string parsers only (no RDKit/Open Babel). R6 / JointPocketAffinityHead / MoE / `v8_biophys_s8` cache unchanged except `LIGAND_FEAT_DIM=12`.

**Tech Stack:** Python stdlib + numpy/torch; existing v8 affinity harness.

## Global Constraints

- Splits: frozen `manifests/v8_pdbbind_refined_cluster30_v1.json` (no-leak wall absolute).
- R6: on-the-fly only; never write protein graph cache.
- No heavy cheminformatics deps for 10.1.1.
- Telemetry: per-complex `ligand_source ∈ {mol2,sdf,hetatm}`; epoch/`run_summary` ratios.
- Train knobs unchanged: `lr_hyperbolic=3e-4`, `lr_head=1e-3`, `λ_aux=0.10`.
- HETATM fallback: pad channels 10–11 with 0; set `hetatm_feature_pad=1`.

### Frozen 12-channel layout

```text
0–6  element: C N O S P Halogen Other
7–9  charge:  Negative Neutral Positive
10   aromatic flag {0,1}
11   heavy-atom degree / 4  (clamp deg∈[0,4])
```

---

## File map

| File | Role |
|------|------|
| `science/tokyo_eye/v8/ligand_interface.py` | MOL2/SDF parse, 12-ch feats, resolver |
| `experiments/training/v8/stage_ligand_assets.py` | Stage/symlink ligands into `data/pdbbind/ligands/` |
| `experiments/training/v8/run_affinity_s10.py` | `ligand_source` telemetry → `run_summary` |
| `tests/v8/test_ligand_mol2_sprint1011.py` | Parser + resolver + pad telemetry |
| Spec | Mark APPROVED |

---

### Phase 1: Native 12-channel organic parser

**Files:**
- Modify: `science/tokyo_eye/v8/ligand_interface.py`
- Create: `tests/v8/test_ligand_mol2_sprint1011.py`
- Verify: `science/tokyo_eye/v8/affinity_head.py` uses `LIGAND_FEAT_DIM` for `lig_in`

**Interfaces:**
- `resolve_ligand_path(pdb_id, root=...) -> Path | None`
- `extract_ligand_mol2(path) -> LigandAtoms | None`
- `extract_ligand_sdf(path) -> LigandAtoms | None`
- `extract_ligand_auto(pdb_id, pdb_path, ...) -> (LigandAtoms, source, meta)`
- `ligand_feature_matrix(...) -> [N, 12]` with aromatic + degree
- `LigandAtoms` gains optional `aromatic: tuple[bool,...]`, `degrees: tuple[int,...]`

- [ ] **Step 1: Write failing tests** (toy MOL2/SDF fixtures: charge, aromatic `@`/`ar`/bond-type-4, degree, resolver order, HETATM pad ch10–11 + `hetatm_feature_pad`)

- [ ] **Step 2: Run — expect fail**

- [ ] **Step 3: Implement parsers + bump `LIGAND_FEAT_DIM=12`** (keep 10.1.0 HETATM path; pad new channels)

- [ ] **Step 4: Tests pass; commit** `feat(v8): Sprint 10.1.1 native MOL2/SDF 12-ch ligand parse`

---

### Phase 2: Staging & asset resolver script

**Files:**
- Create: `experiments/training/v8/stage_ligand_assets.py`
- Create dir: `data/pdbbind/ligands/` (+ `.gitkeep` or README stub)

**CLI:**

```bash
PYTHONPATH=. python experiments/training/v8/stage_ligand_assets.py \
  --manifest manifests/v8_pdbbind_refined_cluster30_v1.json \
  --src /path/to/PDBbind_v*/refined-set \   # or --src-ligands DIR
  --dst data/pdbbind/ligands \
  --mode symlink   # or copy
```

- Prefer `{pdb}/{pdb}_ligand.mol2` PDBBind layout; also accept flat `{pdb}.mol2`.
- Report coverage: n_mol2 / n_sdf / n_missing for train∪val∪core.
- Do **not** fail the whole stage if some IDs missing (coverage report only).

- [ ] **Step 1: Implement staging script + dry-run coverage**

- [ ] **Step 2: Run against available local dump (or document missing)**

- [ ] **Step 3: Commit** `feat(v8): stage_ligand_assets for Sprint 10.1.1`

---

### Phase 3: Production rematch

**Files:**
- Modify: `experiments/training/v8/run_affinity_s10.py` (source telemetry)

```bash
# smoke
PYTHONUNBUFFERED=1 PYTHONPATH=. python experiments/training/v8/run_affinity_s10.py \
  --mode head_only --joint-head --smoke --device cuda \
  --run-name tokyo_eye_v8_affinity_s1011_head_only_smoke

# full rematch
PYTHONUNBUFFERED=1 PYTHONPATH=. python experiments/training/v8/run_affinity_s10.py \
  --mode finetune_hyp --joint-head --device cuda --epochs 12 \
  --lr 1e-3 --lr-hyperbolic 3e-4 --aux-coeff 0.10 \
  --init-ckpt checkpoints/v8/runs/tokyo_eye_v8_mode_c_moe_rebalance_s9/v8_best.pt \
  --run-name tokyo_eye_v8_affinity_s1011_finetune_hyp
```

`run_summary.json` must include:

```json
"ligand_source_rates": {"mol2": 0.0, "sdf": 0.0, "hetatm": 0.0}
```

Gate: Core Pearson \(R \ge 0.40\). Compare to 10.1.0 Core \(R \approx 0.366\).

- [ ] **Step 1: Wire `ligand_source` into dataset + summary ratios**

- [ ] **Step 2: `head_only` smoke** (finite loss; rates logged)

- [ ] **Step 3: Full `finetune_hyp` + Core report**

- [ ] **Step 4: Commit harness telemetry + plan checklist update**

---

## Acceptance checklist

- [ ] `LIGAND_FEAT_DIM == 12`; aromatic + degree unit-locked
- [ ] Native MOL2/SDF only (no RDKit import in `ligand_interface.py`)
- [ ] Resolver: mol2 → sdf → HETATM with pad telemetry
- [ ] Staging script populates / reports `data/pdbbind/ligands/`
- [ ] `run_summary.ligand_source_rates` present
- [ ] Core Pearson vs 0.40 gate recorded
- [ ] No edits to R0–R5 / MoE / cluster30 manifest

---

## Ready for review / execute

Plan logged. Phase 1 coding may begin immediately after this commit.
