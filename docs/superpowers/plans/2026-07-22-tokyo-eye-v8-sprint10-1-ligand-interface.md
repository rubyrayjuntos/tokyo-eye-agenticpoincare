# TokyoEye-v8 Sprint 10.1 — Ligand Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Spec SSOT:** [`docs/superpowers/specs/2026-07-22-tokyo-eye-v8-sprint10-1-ligand-interface-design.md`](../specs/2026-07-22-tokyo-eye-v8-sprint10-1-ligand-interface-design.md)

**Goal:** Inject filtered-ligand chemistry via on-the-fly R6 contacts and asymmetric cross-attention so Core \(-\log K\) Pearson can escape the protein-only capacity floor (\(R \approx 0.007\)).

**Architecture:** Keep frozen R0–R5 protein graphs + MoE spine. Parse ligand atoms (10.1.0: filtered HETATM), build bidirectional R6 edges (\(d\le 4.5\,\text{Å}\)) off-cache, and route \(Q\leftarrow z_{\mathrm{hyp}}\), \(K,V\leftarrow\) ligand 10-channel feats in `JointPocketAffinityHead`.

**Tech Stack:** PyTorch, existing v8 loader/spine, PDB HETATM parse (no RDKit for 10.1.0).

## Global Constraints

- **Do not modify** `r0_r5_graph.py` relation IDs, DSSP/cone, or `v8_biophys_s8` cache hash/write path.
- **R6 on-the-fly only** — never write `edge_index_r6` into `pdb_cache/v8_graph_cache/`.
- **Empty R6:** soft-skip **or** `r6_empty=1` with \(m_i=\mathbf{0}\) (no softmax over empty neighbor sets).
- **Progression:** `head_only` smoke first → then `finetune_hyp` Core rematch.
- **Splits:** reuse `manifests/v8_pdbbind_refined_cluster30_v1.json`; `assert_no_core_leak` still absolute.
- **Ligand features:** exact **10-channel** table below (frozen).

### Frozen 10-channel ligand feature contract

```text
Channels 0–6 (element one-hot):
  0:C  1:N  2:O  3:S  4:P  5:Halogen(F/Cl/Br/I)  6:Other

Channels 7–9 (formal charge bins):
  7:Negative(<0)  8:Neutral(0)  9:Positive(>0)

HETATM bootstrap (10.1.0): if charge absent → Neutral [0,1,0] on channels 7–9.
Array signature: float32 [N_lig, 10]
```

Constants (plan names → code):

```python
LIGAND_FEAT_DIM = 10
ELEMENT_TO_IDX = {"C": 0, "N": 1, "O": 2, "S": 3, "P": 4}
HALOGENS = frozenset({"F", "CL", "BR", "I"})  # after upper()
# idx 5 = halogen, idx 6 = other
R6_DISTANCE_A = 4.5
R6_PROTEIN_LIGAND = 6  # affinity-path constant only; not MoE num_relations
```

---

## File map

| File | Role |
|------|------|
| `science/tokyo_eye/v8/ligand_interface.py` | HETATM sanitize, 10-ch features, R6 builder |
| `science/tokyo_eye/v8/pdbbind_loader.py` | Thin wrappers / batch helpers for ligand+R6 |
| `science/tokyo_eye/v8/affinity_head.py` | Add `JointPocketAffinityHead` (keep Sprint 10 head) |
| `experiments/training/v8/run_affinity_s10.py` | Wire ligand+R6 batches; `--joint-head` / default 10.1 |
| `tests/v8/test_ligand_interface_sprint101.py` | Sanitize, 10-ch shape, R6 math, empty path, cache isolation |
| Spec (status bump) | Mark APPROVED + point at 10-ch table in plan |

---

### Task 1: Ligand sanitize + 10-channel embed + R6 builder

**Files:**
- Create: `science/tokyo_eye/v8/ligand_interface.py`
- Modify: `science/tokyo_eye/v8/pdbbind_loader.py` (re-export helpers)
- Test: `tests/v8/test_ligand_interface_sprint101.py`

**Interfaces:**
- Consumes: PDB path string; residue CA/CB coords `[N_res, 3]`
- Produces:
  - `extract_ligand_hetatm(pdb_path) -> LigandAtoms | None`
  - `ligand_feature_matrix(atoms) -> np.ndarray [N_lig, 10]`
  - `build_r6_edges(res_proxy_xyz, lig_xyz, cutoff=4.5) -> (edge_index [2,E], meta)`
  - `LigandAtoms`: coords `[N,3]`, elements `list[str]`, charges `list[float|None]`, resname, n_atoms

- [ ] **Step 1: Write failing tests** for water/ion drop, largest multi-atom hetero selection, 10-ch shape/neutral default, R6 bidirectional ≤4.5Å, empty R6 meta flag, and assert builder never imports/calls graph-cache write.

```python
def test_ten_channel_neutral_default():
    # single carbon HETATM-like atom, charge=None
    feats = ligand_feature_matrix(...)
    assert feats.shape == (1, 10)
    assert feats[0, 0] == 1.0  # C
    assert feats[0, 8] == 1.0  # neutral

def test_r6_bidirectional_cutoff():
    # one residue CB within 4.5 of one lig atom → E==2
    ...
```

- [ ] **Step 2: Run tests — expect fail** (`pytest tests/v8/test_ligand_interface_sprint101.py -q`)

- [ ] **Step 3: Implement `ligand_interface.py`**
  - Drop resnames `{HOH,WAT,DOD,TIP}` and `{CL,NA,K,MG,ZN,CA,SO4,PO4}`
  - Select largest contiguous multi-atom hetero (\(N\ge 2\))
  - Element → channels 0–6; charge bins 7–9 (default neutral)
  - R6: residue proxy = CB else CA; both directions; inclusive ≤4.5
  - Empty edges → `meta={"r6_empty": 1, "n_edges": 0}` and `edge_index` shape `[2,0]`

- [ ] **Step 4: Re-export from `pdbbind_loader.py`**

- [ ] **Step 5: Run tests — expect pass**

- [ ] **Step 6: Commit** `feat(v8): Sprint 10.1.0 ligand HETATM sanitize + R6 off-cache`

---

### Task 2: `JointPocketAffinityHead` asymmetric R6 cross-attn

**Files:**
- Modify: `science/tokyo_eye/v8/affinity_head.py`
- Test: `tests/v8/test_ligand_interface_sprint101.py` (attn section)

**Interfaces:**
- Consumes: `z_hyp [N,d]`, `mechanism_score`, `dehydron_labels`, `lig_feat [L,10]`, `edge_index_r6 [2,E]` (res→lig oriented rows: row0=res, row1=lig; builder emits both dirs — head uses res→lig only or both consistently)
- Produces: `{"affinity_pred", "z_graph", "pocket_weights", "r6_empty"}`

- [ ] **Step 1: Failing tests** — variable N/L; empty R6 → finite pred + `r6_empty==1` + zero messages; softmax pocket weights sum≈1 on dim=0; no NaN

- [ ] **Step 2: Run — expect fail**

- [ ] **Step 3: Implement `JointPocketAffinityHead`**
  - `nn.Embedding` or `Linear(10 → d_k)` for ligand
  - \(Q=W_Q z\), \(K=W_K \ell\), \(V=W_V \ell\); sparse softmax over ligand neighbors per residue
  - Empty neighborhood → \(m_i=0\)
  - Fuse: \(z' = z + \mathrm{Linear}_m(m)\) then existing PocketGate + log₀/exp₀ pool + AffinityFFN
  - Keep Sprint 10 `PocketGatedAffinityHead` intact for regression tests

- [ ] **Step 4: Run tests — pass**

- [ ] **Step 5: Commit** `feat(v8): JointPocketAffinityHead R6 cross-attn`

---

### Task 3: Training harness wire-up + progression discipline

**Files:**
- Modify: `experiments/training/v8/run_affinity_s10.py`
- Optional: `science/tokyo_eye/v8/loader.py` only if batch helper needs residue CB/CA export (prefer compute CB/CA in ligand_interface from records already loaded — avoid cache changes)

**Interfaces:**
- Batch adds: `lig_feat`, `edge_index_r6`, `r6_empty`, `n_lig`
- Flags: `--joint-head` (default true for new run names), modes `head_only` | `finetune_hyp` unchanged freeze contracts
- Metrics: log `r6_edges_mean`, `lig_atoms_mean`, `n_lig_skip`, `r6_empty_count`

- [ ] **Step 1: Extend `AffinityEntryDataset._load`** to call ligand extract + R6 after protein batch; soft-skip if no ligand; if ligand but empty R6 set `r6_empty=1` and zero lig path (do **not** crash)

- [ ] **Step 2: `head_only` smoke** (shape/stability)

```bash
PYTHONPATH=. python experiments/training/v8/run_affinity_s10.py \
  --mode head_only --joint-head \
  --smoke --device cuda \
  --run-name tokyo_eye_v8_affinity_s101_head_only_smoke
```

- [ ] **Step 3: Confirm smoke** — finite loss, no NaN, `improve_ckpts` rules unchanged (`--ckpt-baseline 0.054`)

- [ ] **Step 4: Full `finetune_hyp` rematch**

```bash
PYTHONPATH=. python experiments/training/v8/run_affinity_s10.py \
  --mode finetune_hyp --joint-head \
  --device cuda --epochs 12 \
  --lr 1e-3 --lr-hyperbolic 3e-4 --aux-coeff 0.10 \
  --init-ckpt checkpoints/v8/runs/tokyo_eye_v8_mode_c_moe_rebalance_s9/v8_best.pt \
  --run-name tokyo_eye_v8_affinity_s101_finetune_hyp
```

- [ ] **Step 5: Report Core Pearson / Spearman / RMSE** vs Sprint 10 protein-only floor; gate \(R\ge 0.40\)

- [ ] **Step 6: Commit** harness + summary pointer in plan checklist

---

## Acceptance checklist

- [ ] `LIGAND_FEAT_DIM == 10` unit-locked; HETATM default charge = neutral bin
- [ ] R6 never written to `v8_graph_cache`; `CACHE_INTACT` / `R6_OFF_CACHE` tests green
- [ ] Empty R6 → soft-skip or zero-message + `r6_empty=1` (no runtime shape fault)
- [ ] `head_only` smoke green before `finetune_hyp`
- [ ] Core metrics logged under `checkpoints/v8/runs/tokyo_eye_v8_affinity_s101_*/run_summary.json`
- [ ] `pytest tests/v8/test_ligand_interface_sprint101.py` green
- [ ] No edits to `r0_r5_graph.py` biophysics / relation exclusivity

---

## Integrity rules (copy into PR / gate notes)

1. **On-the-fly R6** — build at load time; zero interaction with `v8_biophys_s8` cache writers.
2. **Clean zero-message** — empty R6 ⇒ \(m_i=0\) and/or soft-skip; never `softmax` over an empty neighbor list without a guard.
3. **Progression** — `head_only` verification → `finetune_hyp` rematch only.

---

## Ready for review

This plan is logged at:

`docs/superpowers/plans/2026-07-22-tokyo-eye-v8-sprint10-1-ligand-interface.md`

Approve to begin Task 1 (sanitize + R6 + tests).
