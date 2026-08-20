# struct_conn Typed Edges — Chem-MVP Ablation Runbook

**Related:** [`design.md`](design.md) · [barcode edge design §9](../dehydron-barcode-input-channel/design.md#9-deferred-work-post-p1) · [`../dehydron-barcode-input-channel/ablation.md`](../dehydron-barcode-input-channel/ablation.md) · [containment design](../hierarchical-containment-edges/design.md) · [`depth-collision.md`](../hierarchical-containment-edges/depth-collision.md)

**Purpose (Chem-MVP):** Matched ablation — **baseline** (existing proximity + role edges) vs **+ chemical `disulf` / `covale` relation paths** on the existing `EquivariantConvMultiRel` stack — answering whether curated chemical tethers improve representation / relational structure beyond heuristic role edges.

> **2026-07-17 architecture lock:** This is **not** “build H-RGCN.” Relation-aware MP already exists and has been trained (feeler multi-rel / role / coupling). Chem-MVP **extends the relation vocabulary** and wires governed `fact_covalent_bond` into the training graph.

> **Vocabulary lock:** Current role IDs are `packing` / `dehydron` / `spoke` / `ribbon` [/ `coupling`]. Chem types are **new IDs** (`disulf`, `covale`, …) that expand `num_relations` + per-relation radial MLPs — not remaps onto packing/spoke.

> **Chem-Full is not bundled:** `hydrog` / `saltbr` / `metalc` need ingest extension (+ DSSP fallback for `hydrog`). Register separately after Chem-MVP closes.

---

## Staging (data readiness)

| Arm | Relations | Data status |
|-----|-----------|-------------|
| **Baseline** | Existing proximity + role edges only | Ready |
| **Chem-MVP** | + `disulf`, `covale` | Governed in `fact_covalent_bond`; needs DB → training-graph bridge |
| **Chem-Full** | + `hydrog`, `saltbr`, `metalc` | Ingest extension + sparsity strategy — **later registration** |

Run **Chem-MVP first**. No auto-promotion to Chem-Full regardless of MVP outcome.

---

## Prerequisites (before any Chem-MVP training)

### 1. DB → training-graph bridge

- Load `fact_covalent_bond` (`disulf` / `covale`) at batch time alongside `attach_role_edge_graph`.
- Emit chem relation rows compatible with `EquivariantConvMultiRel` one-hot masks.
- Document overlap policy with role pairs (default: separate one-hot rows; chem may coexist with ribbon/dehydron).

### 2. Isolated-seed / construction-order check

Adding relation MLPs adds parameters. Before the first cold Chem-MVP run, confirm gate / prototype / trunk init tensors match a baseline build under the same `isolated_torch_seed` / `derived_seed` discipline used after the `node_emb` width confound — **except** the new chem radial-MLP slots.

### 3. Sparsity / missing-relation guard

Many structures (e.g. KRAS G-domain) have **zero** disulfides. Confirm:

- Empty chem-edge set → forward identical to baseline (no NaNs, no forced rel-0 flood from broken one-hots).
- Per-relation radial MLP with zero edges in a batch is skipped (existing `if not mask.any(): continue` path) without poisoning gradients.

Cheap synthetic check + one Stage A structure known to lack disulfides.

### 4. Checkpoint reload sanity

Reload a Chem-MVP checkpoint via `load_model_from_checkpoint` and assert restored `num_relations` / chem flags (infer from `convs.*.radial_mlps.*` if architecture metadata is bare).

---

## Metrics (reuse existing instrumentation)

Identical refusal discipline to barcode P1: **do not interpret investigation / pairwise results if chem-relation liveness is dead.**

### A. Physics path non-regression (barcode gates)

Same instruments and thresholds as [`../dehydron-barcode-input-channel/ablation.md`](../dehydron-barcode-input-channel/ablation.md):

- Corpus-12 **physics_investigation** rim enrichment (≥11/12)
- Cone / τ probe stability
- MoE health non-regression (eligibility / route_H / starvation not attributable to chem edges)

### B. Chemical-pair distance probe (adapted coupled-lock OOD)

Reuse `experiments.diagnostics.kras_coupled_lock_ood` methodology:

- Positive control: **known disulfide-bonded cysteine pairs** (or governed `covale` pairs), not G12D motif
- Measure **trunk** (`encoder_h` Euclidean, T1a hook) **and** **disc** (`hyp_projections_2d` Poincaré) separately
- Normalize lock/chem-pair mean by structure all-pair median
- Prefer structures in or addable to Stage A-12 that actually have ≥1 disulfide / covalent pair (KRAS alone is the wrong positive-control set)

#### Stage A-12 chem coverage (locked 2026-07-17 from live DB + local PDB)

All 12 enabled Stage A structures are ingested (`structure_ingestion` success). Chem rows:

| Signal | Structures | Notes |
|--------|------------|-------|
| `disulf` | `1LYZ` (4), `1F88` (2), `1IVO` (42) | **1IVO alone = 42/48 corpus disulfides** |
| `covale` | `1BG1` (2), `1F88` (11), `1IVO` (10) | Overlaps `1F88`/`1IVO` with disulf |
| Any chem | **5 distinct** structures | Thin N — treat as small-N evidence |
| Zero chem rows | 8/12 (incl. `4OBE`) | Intentional (`bond_count=0` in audit); sparsity guard target |

#### Per-structure reporting + dominance guard (locked before results)

- Report trunk/disc compaction **per structure**, not only a corpus pool.
- Pooled aggregate is allowed only as a secondary summary and must state each structure's pair-share.
- **Dominance guard:** refuse a pooled "win" if any single structure contributes **>40%** of positive-control pairs in that aggregate (1IVO at 42/48 ≈ 87.5% fails this today — so disulfide pooled verdicts are invalid unless 1IVO is down-weighted / excluded from the pool, or the pool is explicitly labeled 1IVO-dominated and not used for promotion).
- **Win requires per-structure consistency:** a Chem-MVP win needs sign-correct trunk compaction on **at least the 3 disulfide-bearing structures individually** (`1LYZ`, `1F88`, `1IVO`), not merely a promising pooled number. Same skepticism as other small-N scores this session (bootstrap CIs, distance-correlation noise).

### C. Per-relation liveness / utilization

Analogous to barcode `liveness_*`:

- Inference probe: drop disulf/covale edge rows and measure output deltas (`probe_chem_edge_liveness`)
- Per-relation radial-MLP **output variance** on chem edges present in the sampled structure
- Empty chem set in the corpus sample → `liveness_chem_skipped=1` (not a dead-channel fail)
- A chem relation present in the sample with near-zero inference delta → **dead channel** → refuse pairwise interpretation
- Persist into `metrics.json` as `liveness_chem_*` (not MLflow-only) — same class of gap as `liveness_barcode_alive`

---

## Pre-registered Chem-MVP outcomes

| Outcome | Criteria |
|---------|----------|
| Win | Rim enrichment ≥11/12 held or improved; chem-pair distances show **consistent, sign-correct compaction at trunk** on **≥3/3 disulfide-bearing Stage A structures individually** (`1LYZ`, `1F88`, `1IVO`), not disc-only / not pooled-only; per-relation liveness non-degenerate for relations present; MoE health non-regressing; pooled aggregate (if reported) passes the ≤40% pair-share dominance guard |
| **Partial** | Disc-only compaction / separation (barcode-class projection artifact), **or** trunk compaction only on a 1IVO-dominated pool without per-structure confirmation — record as real for this architecture, **do not promote** |
| **Fail** | Chem-relation liveness dead, **or** no measurable pairwise effect at trunk or disc |

**No auto-promotion to Chem-Full.** Chem-Full requires its own registration after ingest extension.

---

## Suggested parent / recipe (locked at launch 2026-07-17)

Matched Stage A-12 cold feeler arms (no barcode Full / Fix-1 / containment):

| Arm | `RUN_ID` | Flags |
|-----|----------|-------|
| **Baseline** | `chem_mvp_baseline_role_stage_a12_cold_v1` | `--v66-feeler-lineage` (role on), no `--chem-edge-mp` |
| **Chem-MVP** | `chem_mvp_stage_a12_cold_v1` | same + `--chem-edge-mp` |

- Corpus: `manifests/v6_corpus_stage_a_small_v1.json` (12 enabled; chem positives `1LYZ`/`1F88`/`1IVO`)
- Seed: `1` (`init_seed` for gate/prototype isolation)
- Epochs: **20** (both arms)
- Make: `make train-v66-chem-mvp` / `make train-v66-chem-mvp-baseline`
- Identity checklist: `node_dim=3` (topology_three_vector), `gnn_lineage=v6.6`, `role_edge_mp=True`, chem false on baseline / true on Chem-MVP
- **Apples-to-apples proof (post-hoc):** `prototype_tangent` at epoch 0 identical between arms (`max |Δ|=0`). Chem arm lacked `epoch_000.pt` (snapshots started at ep1); reconstructed ep0 under the same kwargs is at `checkpoints/v66/diagnostics/chem_mvp_stage_a12/`. Baseline saves real `epochs/epoch_000.pt` for the comparison.
- **Verdict order:** physics gates first (rim / cone·τ / MoE), then per-structure chem-pair probe (`1LYZ`/`1F88`/`1IVO`); do **not** use aggregate score as the decision.

---

## Result — CLOSED `partial`, not promoted (2026-07-17)

Chem-MVP is closed as **`partial`** on the specific hypothesis it tested —
*curated disulfide/covalent tethers act as reliable geometric shortcuts that pull
bonded residues together in the learned representation*. Chemical liveness is real
and physics gates are clean, but the pairwise geometry is **not consistent across
structures at the trunk**, which is the promotion-deciding layer. No coefficient
tweak, no second seed. Chem-Full (`hydrog`/`saltbr`/`metalc`) remains a separately
registered question and is unaffected.

### Physics gates (passed — probe was allowed to proceed)

| Gate | Baseline | Chem-MVP | Verdict |
|------|---------:|---------:|---------|
| Corpus-12 rim enrichment | 12/12 | 12/12 | hold |
| `probe_r_depth_tau` | 0.793 | 0.787 | non-regressing |
| `probe_r_proj_depth` | 0.949 | 0.904 | non-regressing (small dip) |
| `routing_entropy` / eff. experts | 1.382 / 3.98 | 1.378 / 3.96 | flat |
| `min_routing_fraction` | 0.197 | 0.187 | healthy |
| `liveness_chem_alive` | — | 1.0 | alive |

Score ignored as decision input (baseline 3.59 vs chem 3.62). Artifact:
`checkpoints/v66/diagnostics/chem_mvp_stage_a12/physics_gates_summary.json`.

### Chem-pair probe (mapped in-graph disulfides)

Dominance guard used **mapped** pair counts (what the model sees), not raw RCSB
counts. Mapped: 1LYZ 4/4, 1F88 1/2 (chain-B copy is cross-chain → skipped), 1IVO
18/42. Pair shares 17% / 4% / **78%** → any pooled disulfide verdict is
1IVO-dominated and invalid for promotion (guard fails, as pre-registered).

| Structure | Trunk norm Δ | Disc norm Δ | Layer verdict | Trunk compacts? |
|-----------|-------------:|------------:|---------------|:----------------|
| 1LYZ | **−0.046** (std 0.020, 4/4) | +0.041 (3/4 expand) | `trunk_disc_disagree` | yes, clean |
| 1F88 | +0.049 (n=1) | +0.766 (n=1) | `trunk_and_disc_same_sign` | no |
| 1IVO | −0.028 (std 0.069, 6/18 expand) | −0.080 (std 0.251) | `trunk_and_disc_same_sign` | yes, weak/noisy |

Negative Δ = closer under Chem. Verdict **`partial`**, reason
`trunk_compaction_incomplete_across_structures` (clean only on 1LYZ). Artifact:
`checkpoints/v66/diagnostics/chem_mvp_stage_a12/disulf_pair_probe.json`; probe
`experiments/diagnostics/chem_mvp_disulf_pair_probe.py`
(`tests/test_chem_mvp_disulf_pair_probe.py`).

### Two findings that refine the interpretation

**1. 1F88 is single-pair-unresolvable, not a structurally anomalous counterexample.**
Its one mapped disulfide is **Cys110–Cys187**, the canonical rhodopsin EL2↔TM3
bridge (CA–CA 5.46 Å) — geometrically identical to the 5.0–6.8 Å band of every
1LYZ and 1IVO pair, and 1F88 genuinely has only one disulfide per chain (the 12
"skipped" are 11 covale + the chain-B copy, not lost disulfides). With n=1 the
per-structure number is a single pair distance, not an average. Contextualized
against the within-structure spread: its trunk `+0.049` sits **inside 1IVO's own
per-pair trunk range** (min −0.236, max +0.124), and its disc `+0.766` is a single
extreme draw from a layer that is unreliable everywhere (see finding 2). This is a
correction to the launch-note reading of 1F88 as an "unambiguous counterexample" —
the honest characterization is *single-pair, high-variance, wrong-direction*, and
it does **not** upgrade the verdict: even setting 1F88 aside, clean trunk
compaction holds only on 1LYZ, while 1IVO (which dominates the pool) is weak and
noisy (a third of pairs expand). Still `partial`, for a cleaner reason.

**2. Trunk↔disc disagreement recurs on a new edge type → standing caveat.**
1LYZ compacts cleanly at trunk on all 4 pairs yet expands at disc on 3/4; 1IVO's
disc per-pair std (0.251) is ≈3.6× its trunk std (0.069). This is the same
"disc-projection does not reflect trunk relational structure" pattern established
in the barcode OOD work, now independently reproduced on chemical edges. Written up
as a standing, cross-cutting caveat:
[`docs/audit/DISC_PROJECTION_NOT_TRUNK_PROXY.md`](../../audit/DISC_PROJECTION_NOT_TRUNK_PROXY.md)
— disc distance is never a proxy for trunk relational structure without a paired
trunk measurement.

---

## Status

| Item | Status |
|------|--------|
| Architecture lock (extend `EquivariantConvMultiRel`) | **Adopted** 2026-07-17 |
| Chem-MVP vs Chem-Full staging | **Adopted** |
| Metrics / win-partial-fail table | **Pre-registered** |
| DB→graph bridge | **Implemented** 2026-07-17 (`science/dtie/v66/chem_edge_graph.py`, `--chem-edge-mp`) |
| Isolated-seed check | **Confirmed** 2026-07-17 — `prototype_tangent` + topo encoder identical chem-on/off under `init_seed` (`tests/test_chem_liveness_metrics.py`) |
| Sparsity guard | **Covered** by unit test + Stage A zeros confirmed intentional |
| Chem liveness → `metrics.json` | **Confirmed** 2026-07-17 — `probe_chem_edge_liveness` + stage_runner `liveness_chem_*` persistence path tested (not MLflow-only) |
| First Chem-MVP train | **Complete** 2026-07-17 — `chem_mvp_stage_a12_cold_v1` (20 ep, best_score≈3.62, `liveness_chem_alive=1`) |
| Matched baseline | **Complete** 2026-07-17 — `chem_mvp_baseline_role_stage_a12_cold_v1` (20 ep, best_score≈3.59; `epoch_000.pt` present) |
| Ep0 prototype identity | **MATCHED** — baseline `epoch_000` vs chem-on/off reconstructed ep0: `max \|Δ\|=0` (`checkpoints/v66/diagnostics/chem_mvp_stage_a12/ep0_prototype_identity_baseline_vs_recon.json`) |
| Physics gates | **PASSED** 2026-07-17 — rim 12/12 both arms; cone·τ / MoE non-regressing (`physics_gates_summary.json`) |
| Chem-pair probe | **`partial`** 2026-07-17 — trunk compaction clean only on 1LYZ; 1IVO-dominated pool; 1F88 single-pair (`disulf_pair_probe.json`) |
| **Verdict** | **CLOSED `partial`, not promoted** — honestly negative on the disulf/covale geometric-shortcut hypothesis; no coefficient tweak / seed re-fish |

After this ablation closes: hierarchical containment was listed as next buildable
([design v2](../hierarchical-containment-edges/design.md),
[`ablation.md`](../hierarchical-containment-edges/ablation.md),
[`depth-collision.md`](../hierarchical-containment-edges/depth-collision.md)).
**Update 2026-07-19:** containment diameter unlock **FAIL** — do **not** treat as automatic next.
Active path: chem-MVP re-engage ([`../chem-mvp-reengage/README.md`](../chem-mvp-reengage/README.md)) —
measurement contract + one registered lever family before any successor train.
Optional barcode typed shared-bar remains optional
([design §9](../dehydron-barcode-input-channel/design.md#9-deferred-work-post-p1)).
Chem-Full remains a separately registered optional ingest-extension branch.
`ha_edges_v1` is **PARKED/STOP** ([`../graph-communication/ablation.md`](../graph-communication/ablation.md)).
