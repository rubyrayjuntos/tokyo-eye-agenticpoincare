# Hierarchical Containment Edges (Path B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train-side Path B containment: parse deposited HELIX/SHEET from local Stage A PDBs, append SSE parent nodes + directed `contain_down`/`contain_up` edges on the existing homogeneous PyG `Data`, grow multi-rel by +2, and grade with the pre-registered diameter-stratified flow-influence probe.

**Architecture:** Mirror Chem-MVP attach order (role → chem → **containment**), but unlike chem this **grows `N`** with mean-pooled parent rows. Residue losses / audits / flow probe always slice to the leaf prefix `0:n_residues`. No Normalizer / governed-table writes. Depth-collision lock unchanged (residue radius physics-only).

**Tech Stack:** Python 3.10+, NumPy, PyTorch / PyG, existing v66 `EquivariantConvMultiRel` + `train_loop` / `launch_training`, local files under `pdb_cache/*.pdb`.

**Design spec:** [`docs/specs/hierarchical-containment-edges/design.md`](../../specs/hierarchical-containment-edges/design.md) · Ablation: [`ablation.md`](../../specs/hierarchical-containment-edges/ablation.md) · Hard lock: [`depth-collision.md`](../../specs/hierarchical-containment-edges/depth-collision.md)

## Global Constraints

- **Path B only** — train-side HELIX/SHEET parse; no `fact_*` / Normalizer hierarchy writes.
- **Path A is out of scope** — ingest unification is a separate project that supersedes this stopgap.
- **Deposited vs biotite accepted inconsistency** — parents from deposited HELIX/SHEET; residue `ss`/`ss_type` stays biotite (design §9.1). Do not “fix” boundary disagreement in the acceptance test.
- **Relation typing:** `contain_down` and `contain_up` are **separate** IDs (+2 radial MLPs), not one bidirectional relation.
- **Parent init:** mean-pool of children’s `data.x` rows (Option A). No learned parent embedding in v1.
- **v1 parents:** Level-2 SSE only. No assembly/entity/domain parents required for the diameter gate.
- **Matched parent stack:** Stage A-12 chem-MVP feeler (`role_edge_mp` + `chem_edge_mp`) — same as flow-influence baseline `chem_mvp_stage_a12_cold_v1`.
- **Vocabulary lock (chem on):** IDs `0..6` unchanged; `contain_down=7`, `contain_up=8`; `num_relations=9`.
- **Depth-collision:** residue `cone_depth` / disc radius / `tau_dehydron_rim` remain physics-only — no hierarchy centripetal on residue radius.
- **Primary grade:** `jacobian_flow_influence.py` diameter-stratified asymmetry (ablation table) — no new metric.
- **Prerequisite sequence (Chem-MVP-shaped):** PDB→graph attach → isolated-seed (`max |Δ|=0`, Task 4) → sparsity/empty-mask guard (Task 4) → liveness → cold matched arms → **oversmoothing-at-root (Task 8, design §5.2)** → flow-influence score (Task 9).

> **Plan integrity (2026-07-17):** Task 3 (model `num_relations=9`) and Task 4 (isolated-seed prototype identity + empty-SSE sparsity) are **in this file and binding**. Task 4 copies Chem-MVP’s `torch.equal` / `max |Δ|=0` gate/prototype snapshot pattern — hard prerequisite before any cold run. Design §5.2 oversmoothing-at-root is **Task 8** (not deferred, not silently dropped).

---

## File structure

| File | Responsibility |
|------|----------------|
| `science/dtie/v66/sse_hierarchy.py` | Parse deposited HELIX/SHEET ranges from PDB text; map to residue node indices |
| `science/dtie/v66/containment_edge_graph.py` | Parent-node append + `contain_up`/`contain_down` edge rows; pad chem `edge_attr`; constants |
| `science/dtie/v66/gnn/model.py` | `containment_edge_mp` → `num_relations=9` + aux column shifts |
| `science/training/config.py` | `containment_edge_mp: bool = False` |
| `experiments/training/v6/launch_training.py` | `--containment-edge-mp` CLI + guard (requires role+chem for matched arm) |
| `experiments/training/v6/train_loop.py` | Attach after chem; **residue-slice** all targets/losses to `n_residues` |
| `science/training/feature_liveness.py` | `probe_containment_edge_liveness` |
| `science/training/checkpoint.py` | Infer containment from `radial_mlps.7/8` if metadata thin |
| `experiments/diagnostics/t1c_containment_parent_oversmooth.py` | §5.2 oversmoothing-at-root: T1c MP-relative-loss-null on parent-node slice |
| `Makefile` | Matched cold baseline / containment Stage A-12 targets |
| `tests/test_sse_hierarchy.py` | HELIX/SHEET parse + residue mapping |
| `tests/test_v66_containment_edges.py` | Attach, empty-mask, isolated-seed (`max \|Δ\|=0`), reload |
| `tests/test_containment_liveness_metrics.py` | Liveness probe contract |
| `tests/test_t1c_containment_parent_oversmooth.py` | Parent-slice T1c contract (synthetic fan-in control) |
| `docs/specs/hierarchical-containment-edges/ablation.md` | Status updates only after gates |

---

### Task 1: Deposited HELIX/SHEET parser (PDB text)

**Files:**
- Create: `science/dtie/v66/sse_hierarchy.py`
- Test: `tests/test_sse_hierarchy.py`

**Interfaces:**
- Consumes: PDB file path or text; `residue_ids: Sequence[str]` in `"CHAIN:RESSEQ:"` form (training convention)
- Produces:
  - `@dataclass SSERange`: `sse_type: str` (`"H"`|`"E"`), `chain: str`, `start_resseq: int`, `end_resseq: int`, `helix_id`/`sheet_id` optional
  - `parse_pdb_helix_sheet(pdb_text: str) -> list[SSERange]`
  - `map_sse_ranges_to_node_indices(ranges, residue_ids) -> list[tuple[SSERange, list[int]]]` (skip empty mappings; do not invent residues)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sse_hierarchy.py
from science.dtie.v66.sse_hierarchy import parse_pdb_helix_sheet, map_sse_ranges_to_node_indices

UBQ_HELIX = """\
HELIX    1   1 ILE A   23  GLY A   34  1                                  12
ATOM      1  CA  ILE A  23      0.000   0.000   0.000  1.00  0.00           C
ATOM      2  CA  GLY A  34      1.000   0.000   0.000  1.00  0.00           C
END
"""

def test_parse_helix_range_inclusive():
    ranges = parse_pdb_helix_sheet(UBQ_HELIX)
    assert len(ranges) == 1
    assert ranges[0].sse_type == "H"
    assert ranges[0].chain == "A"
    assert ranges[0].start_resseq == 23
    assert ranges[0].end_resseq == 34

def test_map_range_to_residue_ids():
    residue_ids = [f"A:{i}:" for i in range(23, 35)]
    mapped = map_sse_ranges_to_node_indices(parse_pdb_helix_sheet(UBQ_HELIX), residue_ids)
    assert len(mapped) == 1
    assert mapped[0][1] == list(range(12))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sse_hierarchy.py -v`  
Expected: FAIL (import / missing module)

- [ ] **Step 3: Implement parser**

Parse classic PDB `HELIX` (cols for init/end chain+resseq) and `SHEET` (init/end chain+resseq). Ignore ATOM for range geometry. Map via exact `f"{chain}:{resseq}:"` keys present in `residue_ids`. Drop ranges with zero hits (log count; do not crash).

Also add a smoke test that `pdb_cache/1UBQ.pdb` yields ≥1 HELIX and ≥1 SHEET with non-empty mapped children.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_sse_hierarchy.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add science/dtie/v66/sse_hierarchy.py tests/test_sse_hierarchy.py
git commit -m "$(cat <<'EOF'
feat(containment): parse deposited HELIX/SHEET ranges for Path B parents

EOF
)"
```

---

### Task 2: Containment attach — parent nodes + directed edges

**Files:**
- Create: `science/dtie/v66/containment_edge_graph.py`
- Test: `tests/test_v66_containment_edges.py`

**Interfaces:**
- Consumes: `Data` with `role_edge_graph` + `chem_edge_graph` (matched arm); `ca_coords [N,3]`; `residue_ids`; PDB path/text
- Produces (mutates `Data`):
  - Constants: `ROLE_CONTAIN_DOWN = 7`, `ROLE_CONTAIN_UP = 8`, `NUM_ROLE_RELATIONS_WITH_CONTAINMENT = 9`
  - `pad_chem_edge_attr_for_containment(edge_attr) -> Tensor` — insert 2 zero one-hot cols after chem’s 7 role slots (same pattern as `pad_role_edge_attr_for_chem`)
  - `attach_containment_edge_graph(data, ca_coords, pdb_text_or_path, *, residue_ids, force=False) -> Data`
  - Flags: `data.containment_edge_graph`, `data.n_residue_nodes` (= original N), `data.n_parent_nodes`, `data.containment_edge_counts`

**Behavior (locked):**
1. Require `chem_edge_graph` for the matched Stage A arm (raise if missing unless `force` for unit tests with synthetic chem-width attrs).
2. Parse SSE → mapped child index lists.
3. For each non-empty SSE parent `p` at new index `N+k`:
   - `x[p] = mean(x[children], dim=0)`
   - `ca[p] = mean(ca[children], dim=0)` (geometry for edge vectors only)
4. Append directed edges (not same-ID bidirectional):
   - parent→child: one-hot at `GEO_DIM+ROLE_CONTAIN_DOWN`
   - child→parent: one-hot at `GEO_DIM+ROLE_CONTAIN_UP`
   - geo = `[dx,dy,dz,dist]` from parent centroid ↔ child CA
5. Pad existing `edge_attr` first; do not rewrite role/chem rows’ relation IDs.
6. Zero SSE after map → no parent rows, `edge_index`/`x` unchanged width aside from pad; forward must still work (sparsity guard).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_v66_containment_edges.py
import torch
from torch_geometric.data import Data
from science.dtie.v66.containment_edge_graph import (
    ROLE_CONTAIN_DOWN,
    ROLE_CONTAIN_UP,
    attach_containment_edge_graph,
)

def _toy_chem_ready_data(n=6, feat_dim=3):
    # Minimal: N residues, empty edges with chem-width edge_attr
    from science.dtie.v66.chem_edge_graph import EDGE_ATTR_CHEM_DIM
    x = torch.randn(n, feat_dim)
    ei = torch.zeros(2, 0, dtype=torch.long)
    ea = torch.zeros(0, EDGE_ATTR_CHEM_DIM)
    data = Data(x=x, edge_index=ei, edge_attr=ea)
    data.role_edge_graph = True
    data.chem_edge_graph = True
    return data

def test_attach_appends_parent_and_directed_relations():
    data = _toy_chem_ready_data()
    ca = torch.randn(6, 3)
    residue_ids = [f"A:{i}:" for i in range(10, 16)]
    pdb = (
        "HELIX    1   1 ALA A   10  ALA A   12  1                                   3\n"
        "END\n"
    )
    out = attach_containment_edge_graph(
        data, ca, pdb, residue_ids=residue_ids, force=True
    )
    assert out.n_residue_nodes == 6
    assert out.n_parent_nodes == 1
    assert out.x.shape[0] == 7
    # mean-pool of children 0..2
    assert torch.allclose(out.x[6], data.x[:3].mean(0), atol=1e-5)
    # directed: down and up present as distinct one-hots
    oh = out.edge_attr[:, 4:]  # after GEO_DIM=4
    assert (oh[:, ROLE_CONTAIN_DOWN] > 0).any()
    assert (oh[:, ROLE_CONTAIN_UP] > 0).any()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_v66_containment_edges.py::test_attach_appends_parent_and_directed_relations -v`  
Expected: FAIL

- [ ] **Step 3: Implement `containment_edge_graph.py`**

Follow `science/dtie/v66/chem_edge_graph.py` pad/append style. Keep parent features in the same `node_dim` as leaves (no extra columns).

- [ ] **Step 4: Empty-SSE / pad-only guard test**

```python
def test_empty_sse_pads_attr_but_does_not_grow_n():
    data = _toy_chem_ready_data()
    ca = torch.randn(6, 3)
    residue_ids = [f"A:{i}:" for i in range(10, 16)]
    out = attach_containment_edge_graph(
        data, ca, "END\n", residue_ids=residue_ids, force=True
    )
    assert out.x.shape[0] == 6
    assert out.n_parent_nodes == 0
```

- [ ] **Step 5: Run tests — expect PASS**

Run: `pytest tests/test_v66_containment_edges.py -v`

- [ ] **Step 6: Commit**

```bash
git add science/dtie/v66/containment_edge_graph.py tests/test_v66_containment_edges.py
git commit -m "$(cat <<'EOF'
feat(containment): Path B parent-node attach with contain_up/down

EOF
)"
```

---

### Task 3: Model / config — grow `num_relations` to 9

**Files:**
- Modify: `science/dtie/v66/gnn/model.py` (role+chem branch ~772–816)
- Modify: `science/training/config.py` (`containment_edge_mp: bool = False`)
- Modify: `science/training/checkpoint.py` (`infer_v66_model_kwargs` sniff `radial_mlps.7` / `.8`)
- Test: extend `tests/test_v66_containment_edges.py`

**Interfaces:**
- `GOSPConeMapperV66(..., containment_edge_mp: bool = False)`
- When `containment_edge_mp`: require `chem_edge_mp`; set `num_relations = 9`; shift spoke/coupling/barcode aux columns by +2 vs chem (mirror chem’s shift vs role)

- [ ] **Step 1: Failing isolated-width test**

```python
def test_containment_radial_mlp_count_is_nine():
    from science.dtie.common.isolated_init import isolated_torch_seed
    from science.dtie.v66.gnn.model import GOSPConeMapperV66

    with isolated_torch_seed(123):
        m = GOSPConeMapperV66(
            node_dim=3, role_edge_mp=True, chem_edge_mp=True,
            containment_edge_mp=True, init_seed=123,
        )
    assert len(m.convs[0].radial_mlps) == 9
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Wire model + config + checkpoint sniff**

In `model.py`, after the chem column block:

```python
if self.containment_edge_mp:
    from science.dtie.v66.containment_edge_graph import (
        COUPLING_STRENGTH_COL_CONTAIN,
        DEHYDRON_BARCODE_COL_CONTAIN,
        NUM_ROLE_RELATIONS_WITH_CONTAINMENT,
        SPOKE_RHO_COL_CONTAIN,
    )
    if not self.chem_edge_mp:
        raise ValueError("containment_edge_mp requires chem_edge_mp in v1 matched arm")
    num_relations = NUM_ROLE_RELATIONS_WITH_CONTAINMENT  # 9
    spoke_rho_col = SPOKE_RHO_COL_CONTAIN
    coupling_strength_col = COUPLING_STRENGTH_COL_CONTAIN
    dehydron_barcode_col = DEHYDRON_BARCODE_COL_CONTAIN
```

Define the `*_CONTAIN` column constants in `containment_edge_graph.py` as `GEO_DIM + 9` layout (chem’s `EDGE_ATTR_CHEM_DIM` + 2 one-hot cols).

- [ ] **Step 4: Tests PASS + commit**

```bash
git add science/dtie/v66/gnn/model.py science/dtie/v66/containment_edge_graph.py \
  science/training/config.py science/training/checkpoint.py \
  tests/test_v66_containment_edges.py
git commit -m "$(cat <<'EOF'
feat(containment): expand multi-rel bank to contain_up/down (9 relations)

EOF
)"
```

---

### Task 4: Isolated-seed prototype identity + sparsity guard (hard prerequisite)

**Status in plan:** **Present and binding** — same Chem-MVP prerequisite #2. Do **not** launch cold training until this task’s tests are green.

**Files:**
- Test: `tests/test_v66_containment_edges.py` (add cases)
- Reference pattern: `tests/test_v66_chem_edges.py::test_chem_radial_mlp_expansion_preserves_gate_init_under_isolated_seed`
- Also: `tests/test_chem_liveness_metrics.py::test_prototype_bank_identical_chem_on_vs_off_with_init_seed` (`max |Δ|` messaging)

**Interfaces:**
- Consumes: `isolated_torch_seed` / `init_seed` from `science/dtie/common/isolated_init.py`
- Produces: **prototype / gate identity** containment-off vs containment-on under the same seed — `torch.equal` on every `gate.*` and `*prototype*` state_dict tensor ⇒ **`max |Δ| = 0`**. New `radial_mlps.7/8` slots may differ in count only; they are not compared for equality.
- Sparsity: empty SSE → forward with no NaNs; unused contain relations skipped via existing `if not mask.any(): continue`

- [ ] **Step 1: Write the failing isolated-seed test (verbatim chem pattern)**

```python
def test_containment_radial_mlp_expansion_preserves_gate_init_under_isolated_seed() -> None:
    """New contain MLP slots must not scramble gate/prototype under init_seed discipline.

    Hard prerequisite before any cold containment run — same as Chem-MVP.
    Required: max |Δ| = 0 on gate/prototype tensors (torch.equal).
    """
    from science.dtie.common.isolated_init import isolated_torch_seed
    from science.dtie.v66.containment_edge_graph import (
        NUM_ROLE_RELATIONS_WITH_CONTAINMENT,
    )
    from science.dtie.v66.gnn.model import GOSPConeMapperV66

    def _gate_proto_snapshot(model: GOSPConeMapperV66) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for name, tensor in model.state_dict().items():
            if name.startswith("gate.") or "prototype" in name.lower():
                out[name] = tensor.detach().clone()
        return out

    with isolated_torch_seed(123):
        baseline = GOSPConeMapperV66(
            node_dim=4,
            hidden=32,
            num_layers=1,
            num_experts=2,
            role_edge_mp=True,
            chem_edge_mp=True,
            containment_edge_mp=False,
            init_seed=123,
        )
        base_snap = _gate_proto_snapshot(baseline)

    with isolated_torch_seed(123):
        contain = GOSPConeMapperV66(
            node_dim=4,
            hidden=32,
            num_layers=1,
            num_experts=2,
            role_edge_mp=True,
            chem_edge_mp=True,
            containment_edge_mp=True,
            init_seed=123,
        )
        contain_snap = _gate_proto_snapshot(contain)

    assert base_snap.keys() == contain_snap.keys()
    for key in base_snap:
        assert torch.equal(base_snap[key], contain_snap[key]), (
            f"{key} diverged containment-off vs on "
            f"(max |Δ|={(base_snap[key] - contain_snap[key]).abs().max().item()})"
        )
    assert len(contain.convs[0].radial_mlps) == NUM_ROLE_RELATIONS_WITH_CONTAINMENT
    assert len(baseline.convs[0].radial_mlps) == NUM_ROLE_RELATIONS_WITH_CONTAINMENT - 2
```

- [ ] **Step 2: Empty-SSE / empty-relation forward guard**

Build a tiny model + chem-width batch with **no** containment edges (`attach` on `"END\\n"` or skip attach). Forward once — no NaNs; relations 7/8 have empty masks and are skipped.

```python
def test_empty_containment_forward_no_nan():
    # model containment_edge_mp=True; data chem-ready, n_parent_nodes=0
    # out = model(data); assert torch.isfinite(out["encoder_h"]).all()  # or live output key
    ...
```

- [ ] **Step 3: Run tests — PASS (hard gate)**

Run: `pytest tests/test_v66_containment_edges.py::test_containment_radial_mlp_expansion_preserves_gate_init_under_isolated_seed tests/test_v66_containment_edges.py::test_empty_containment_forward_no_nan tests/test_v66_chem_edges.py::test_chem_radial_mlp_expansion_preserves_gate_init_under_isolated_seed -v`  
Expected: all PASS; prototype identity reports `max |Δ| = 0` if any assert message fires.

- [ ] **Step 4: Commit**

```bash
git add tests/test_v66_containment_edges.py
git commit -m "$(cat <<'EOF'
test(containment): isolated-seed max|Δ|=0 and empty-SSE forward guards

EOF
)"
```

---

### Task 5: Train-loop attach + residue-only loss slice

**Files:**
- Modify: `experiments/training/v6/train_loop.py` (after chem attach ~105–120)
- Modify: `experiments/training/v6/launch_training.py` (`--containment-edge-mp`)
- Modify: `experiments/training/v6/_data.py` if PDB path must be threaded onto `prot` (e.g. `prot["pdb_path"]` from `pdb_cache/{pdb_id}.pdb`)
- Test: `tests/test_v66_containment_edges.py` or a focused train-loop unit if one exists for chem

**Critical difference from Chem-MVP:** after attach, `data.x.shape[0] = n_residues + n_parents`. Every residue target (`target_rho`, rim, uncertainty, MoE labels, etc.) and every metric that assumes `out.shape[0] == N` must use `n = prot["n_residues"]` / `data.n_residue_nodes` and slice `out[:n]`.

- [ ] **Step 1: Document slice sites**

Grep train_loop for uses of `data.x.shape[0]`, `target_rho`, `encoder_h`, and ensure each residue path slices. Parent rows may receive MP messages but **must not** enter residue physics losses (depth-collision).

- [ ] **Step 2: Wire attach**

```python
if bool(getattr(model, "containment_edge_mp", False)):
    from science.dtie.v66.containment_edge_graph import attach_containment_edge_graph
    data = attach_containment_edge_graph(
        data, ca, prot["pdb_path"],
        residue_ids=residue_ids,
    )
```

CLI: `--containment-edge-mp` sets `config.containment_edge_mp`; exit 2 if role/chem not enabled (matched arm).

- [ ] **Step 3: Failing test for slice invariant**

```python
def test_attach_sets_n_residue_nodes_for_loss_slice():
    # after attach with 1 parent, n_residue_nodes == original N
    ...
    assert out.n_residue_nodes == 6
    assert out.x.shape[0] == 7
```

- [ ] **Step 4: Manual dry-forward on 1UBQ** (optional in CI)

```bash
# smoke: one structure attach inside a tiny script or pytest using pdb_cache/1UBQ.pdb
pytest tests/test_v66_containment_edges.py -k ubq -v
```

- [ ] **Step 5: Commit**

```bash
git add experiments/training/v6/train_loop.py experiments/training/v6/launch_training.py \
  experiments/training/v6/_data.py science/training/config.py tests/test_v66_containment_edges.py
git commit -m "$(cat <<'EOF'
feat(containment): train-loop Path B attach with residue-only loss slice

EOF
)"
```

---

### Task 6: Per-relation liveness probe

**Files:**
- Modify: `science/training/feature_liveness.py`
- Create: `tests/test_containment_liveness_metrics.py`

**Interfaces:**
- `probe_containment_edge_liveness(model, proteins, device, ...) -> dict`
- Drop **containment edge rows** (IDs 7/8), not zero one-hots (same rel-0 fallback lesson as chem)
- Keys: `liveness_containment_alive`, `liveness_containment_skipped`, `radial_var_contain_down`, `radial_var_contain_up`
- Wire into `run_feature_liveness_probes` when `containment_edge_mp`

- [ ] **Step 1: Failing test** — mirror `tests/test_chem_liveness_metrics.py` structure with containment flags

- [ ] **Step 2: Implement probe + wire**

- [ ] **Step 3: PASS + commit**

```bash
git add science/training/feature_liveness.py tests/test_containment_liveness_metrics.py
git commit -m "$(cat <<'EOF'
feat(containment): per-relation liveness probe for contain_up/down

EOF
)"
```

---

### Task 7: Checkpoint reload + Makefile matched arms

**Files:**
- Modify: `science/training/checkpoint.py` (if not finished in Task 3)
- Modify: `Makefile` — two targets mirroring chem MVP cold
- Modify: `docs/specs/hierarchical-containment-edges/ablation.md` status rows only

**Matched arms (lock at launch):**

| Arm | Flags | Output dir |
|-----|-------|------------|
| Baseline | chem-MVP Stage A-12 stack, **no** `--containment-edge-mp` | `checkpoints/v66/runs/containment_baseline_chem_stage_a12_cold_v1` |
| Containment | same + `--containment-edge-mp --feature-liveness-probe` | `checkpoints/v66/runs/containment_pathb_stage_a12_cold_v1` |

Copy the exact chem-MVP Makefile recipe block (`train-v66-chem-mvp-*` around lines 990–1030) and only add/remove the containment flag + output dir names. Do not change MoE / cone / disc recipe.

- [ ] **Step 1: Reload test**

```python
def test_load_checkpoint_restores_containment_num_relations(tmp_path):
    # save tiny containment-on checkpoint; load via load_model_from_checkpoint;
    # assert len(model.convs[0].radial_mlps) == 9
    ...
```

- [ ] **Step 2: Makefile targets**

- [ ] **Step 3: Commit**

```bash
git add Makefile science/training/checkpoint.py \
  docs/specs/hierarchical-containment-edges/ablation.md \
  tests/test_v66_containment_edges.py
git commit -m "$(cat <<'EOF'
chore(containment): matched Stage A-12 cold Makefile arms + reload sniff

EOF
)"
```

---

### Task 8: Oversmoothing-at-root re-check (design §5.2 — real gate)

**Not deferred.** Design §5.2 named this explicitly; v1 SSE parents have lower fan-in than assembly hubs, but risk is non-zero. Reuse T1c methodology before interpreting diameter-asymmetry.

**Files:**
- Create: `experiments/diagnostics/t1c_containment_parent_oversmooth.py`
- Test: `tests/test_t1c_containment_parent_oversmooth.py`
- Reference: `experiments/diagnostics/t1c_mp_relative_loss_null.py` (`_rel_drop`, synth rank control, MP hook pattern)

**Interfaces:**
- Consumes: containment-on checkpoint + Stage A protein with `n_residue_nodes` / parent rows after attach
- Produces: JSON comparing relative rank/variance drop on **parent-node slice** `encoder_h[n_residue_nodes:]` vs a synthetic control cloud with the **same fan-in** (mean children per parent) pushed through the same MP stack
- Gate (pre-register before first score): parent relative drop must be **consistent with null** (not markedly worse than synth control at matched fan-in) — same qualitative read as residue T1c (`MP_LOSS_CONSISTENT_WITH_NULL`). Exact numeric floor: copy the residue T1c reporting style; refuse to interpret flow-influence if parents show collapse far beyond the synth control.

- [ ] **Step 1: Failing unit test** — synthetic Data with known parent indices; helper returns parent-slice embeddings only

```python
def test_parent_slice_excludes_residues():
    # n_residue_nodes=4, n_parent_nodes=2 → parent indices 4,5 only
    ...
```

- [ ] **Step 2: Implement diagnostic CLI** wrapping T1c helpers; restrict metrics to parent rows; write artifact under `checkpoints/v66/diagnostics/t1c_containment_parent_oversmooth/`

- [ ] **Step 3: Run on containment cold best (after Task 7 arms exist)** before Task 9 primary score

```bash
python -m experiments.diagnostics.t1c_containment_parent_oversmooth \
  --checkpoint checkpoints/v66/runs/containment_pathb_stage_a12_cold_v1/v66_best.pt \
  --out checkpoints/v66/diagnostics/t1c_containment_parent_oversmooth/
```

- [ ] **Step 4: File result in ablation.md gate order (before primary asymmetry)** + commit diagnostic + tests only

```bash
git add experiments/diagnostics/t1c_containment_parent_oversmooth.py \
  tests/test_t1c_containment_parent_oversmooth.py \
  docs/specs/hierarchical-containment-edges/ablation.md
git commit -m "$(cat <<'EOF'
feat(containment): T1c-style oversmoothing-at-root check on SSE parents

EOF
)"
```

---

### Task 9: Flow-influence grade (no new metric)

**Files:**
- Modify only if needed: `experiments/diagnostics/jacobian_flow_influence.py` to honor `n_residue_nodes` / skip parent indices when present
- Ablation status update after run

**Pre-registered table:** [`ablation.md`](../../specs/hierarchical-containment-edges/ablation.md) — fail-group ≥3/5 clear asym 0.05; pass-group within 10%; hub-tracking holds.

**Gate order reminder:** isolated-seed (Task 4) → physics non-regression → **oversmoothing-at-root (Task 8)** → **this** primary asymmetry score.

- [ ] **Step 1: Ensure probe uses residue prefix only**

If `getattr(data, "n_residue_nodes", None)` is set, restrict Jacobian nodes and classical GT to `0:n_residue_nodes`. Parents must not enter betweenness / asymmetry pools.

- [ ] **Step 2: Unit test** — synthetic Data with 1 parent; probe node count == `n_residue_nodes`

- [ ] **Step 3: After cold arms finish + Task 8 green, score both checkpoints**

```bash
# baseline (reuse existing stage_a12.json if identical recipe/weights; else re-run)
# containment:
python -m experiments.diagnostics.jacobian_flow_influence \
  --checkpoint checkpoints/v66/runs/containment_pathb_stage_a12_cold_v1/v66_best.pt \
  --out checkpoints/v66/diagnostics/learned_flow_influence/containment_pathb_stage_a12.json
```

- [ ] **Step 4: File win/partial/fail in ablation.md — per-structure, no pooling**

Remember §9.1: odd behavior confined to SSE boundaries is an accepted deposited↔biotite inconsistency, not an automatic fail explanation for diameter-asymmetry.

- [ ] **Step 5: Commit results filing only when user asks** (do not auto-commit large checkpoint artifacts)

---

## Spec coverage checklist

| Spec item | Task |
|-----------|------|
| Path B HELIX/SHEET parse | 1 |
| Parent mean-pool Option A | 2 |
| `contain_down` / `contain_up` separate IDs | 2–3 |
| No governed writes | Global + 2/5 |
| Chem-MVP sequence (attach → seed → sparsity → liveness) | 2,4,6 |
| Isolated-seed `max \|Δ\|=0` prototype identity | **4 (hard prerequisite)** |
| Residue radius physics-only | 5 (loss slice) + Global |
| Oversmoothing-at-root (§5.2 T1c on parents) | **8** |
| Diameter-stratified acceptance | 9 |
| Deposited vs biotite note | Global + ablation reminder in Task 9 |
| Path A deferred | Global |

## Out of scope (do not implement in this plan)

- Path A ingest tables / Normalizer hierarchy
- Assembly / entity / CDD domain parents
- Path 2 directionality reward
- Chem-Full
- Changing residue `ss_type` to deposited SSE
- Dual-loading residue radius (forbidden)
- Deferring §5.2 oversmoothing — **not** out of scope; Task 8 is required