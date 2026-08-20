# Design: struct_conn Typed Edges (Chem-MVP → Chem-Full)

**Status:** Chem-MVP ablation **CLOSED `partial`, not promoted** (2026-07-17). Chem-Full remains a separate later registration. Active path: [`../chem-mvp-reengage/README.md`](../chem-mvp-reengage/README.md). Containment diameter unlock **FAIL** — not automatic next. `ha_edges_v1` **PARKED**.
**Bridge:** Train-side `fact_covalent_bond` → graph attach landed 2026-07-17 (`--chem-edge-mp`); no Normalizer writes.
**Related:** [`ablation.md`](ablation.md) · [barcode edge design §9](../dehydron-barcode-input-channel/design.md#9-deferred-work-post-p1) · [`../structure-ingestion-normalization/design.md`](../structure-ingestion-normalization/design.md) · [containment design](../hierarchical-containment-edges/design.md) · [`depth-collision.md`](../hierarchical-containment-edges/depth-collision.md) · [graph-communication](../graph-communication/ablation.md)

## 1. Goal

Ask whether **chemically curated** residue–residue edges from RCSB `_struct_conn` improve representation quality or relational structure beyond the **already-exercised** proximity / role-edge multi-relational MP — not whether we need to invent relational message passing.

## 2. Architecture decision (locked)

**Do not build a new H-RGCN.** Extend `EquivariantConvMultiRel` (`science/dtie/v66/gnn/equivariant_conv_multirel.py`).

| Design-doc Option | Status in codebase |
|-------------------|--------------------|
| **A — distinct per-relation weights** | **Already implemented** as per-relation radial MLPs feeding a shared equivariant tensor product |
| **B — learned relation embeddings** | Not implemented; not required for Chem-MVP |

Proven training (not scaffolding): feeler multi-rel / role / coupling runs landmarked under `checkpoints/v66/runs/` and MLflow `tokyo-eyes-v66` (e.g. `feeler_expand_23_multirel_cold_v1`, `feeler_expand_23_role_cold_v1`, `feeler_expand_23_coupling_v1`).

### Current exercised vocabularies (do not confuse with chem types)

- **Role graph** (`role_edge_mp`): `packing`, `dehydron`, `spoke`, `ribbon`, optional `coupling`
- **Thermo multi-rel** (`multi_rel_edge_mp` without role): `generic`, `wrapped_hbond`, `dehydron`

Chem types (`disulf`, `covale`, later `hydrog` / `saltbr` / `metalc`) are **new relation IDs** that expand `num_relations` and add radial-MLP slots. They are **not** remaps onto packing/spoke.

Locked Chem-MVP numeric IDs (role one-hot bank):

| ID | Name |
|----|------|
| 0–3 | packing / dehydron / spoke / ribbon |
| 4 | coupling (may be empty mask when coupling disabled) |
| **5** | **`disulf`** |
| **6** | **`covale`** |

With `--chem-edge-mp`, `EquivariantConvMultiRel.num_relations=7` and edge_attr one-hot width expands by two columns (aux/spoke/barcode columns shift +2).

## 3. Data readiness

| Type | Ingest today | Training graph today |
|------|--------------|----------------------|
| `disulf`, `covale` | Parsed → `fact_covalent_bond` | **Not** loaded into role/contact MP |
| `hydrog`, `saltbr`, `metalc` | **Dropped** in parser filter | Absent |

Chem-MVP = bridge + expand MLP bank for `disulf`/`covale` only.  
Chem-Full = ingest extension (+ DSSP fallback for `hydrog` sparsity) as its **own** registration.

## 4. Graph integration (Chem-MVP)

1. Load governed bonds from `fact_covalent_bond` at training-batch time (alongside `attach_role_edge_graph`).
2. Emit additional directed edge rows with new one-hot relation IDs (same exclusive-row pattern as role edges, or documented multi-row overlap policy).
3. Expand `EquivariantConvMultiRel.num_relations` / `radial_mlps` accordingly.
4. Decide precedence when a chem pair also qualifies as dehydron/packing/ribbon (pre-register; default proposal: chem rows coexist as separate relation rows, like ribbon+dehydron overlap under exclusivity nuances).
5. Refuse or skip metal partners that are not residue nodes until Chem-Full scopes ligand hubs.

Training-only until proven — do not silently rewrite Normalizer `fact_graph_edge` in Chem-MVP.

## 5. Risks already named

- **Sparsity:** many Stage A structures (incl. KRAS G-domain) have **no** disulfides — missing chem types must degrade to baseline, not degenerate MLP inputs (missing-mask / empty-relation skip).
- **Ignore risk:** sparse edges may get near-zero gradient — catch with **per-relation liveness**, same discipline as barcode/aleatoric dead-channel probes.
- **Init order:** new MLP parameters must not scramble downstream RNG (gate/prototypes) — `isolated_torch_seed` / `derived_seed` check before cold training.
- **Checkpoint reload debt:** bare feeler phase files can mis-route through V6 lineage if metadata is thin; Chem-MVP reload paths must restore relation flags from `radial_mlps.*` tensors (existing `infer_v66_model_kwargs` pattern).

## 6. Sequencing vs related work

1. Chem-MVP (this spec / [`ablation.md`](ablation.md)) — **closed `partial`**
2. Hierarchical containment **next** ([design v2](../hierarchical-containment-edges/design.md),
   [`ablation.md`](../hierarchical-containment-edges/ablation.md)) — under the
   [`depth-collision.md`](../hierarchical-containment-edges/depth-collision.md) lock;
   primary acceptance = diameter-stratified flow asymmetry
3. Path 2 explicit directionality reward — diam ≤9 only, after containment closes
4. Optional barcode **typed shared-bar** ([dehydron design §9](../dehydron-barcode-input-channel/design.md#9-deferred-work-post-p1))

Chem-Full is a separate optional ingest-extension branch after Chem-MVP, not a
gate on containment.
