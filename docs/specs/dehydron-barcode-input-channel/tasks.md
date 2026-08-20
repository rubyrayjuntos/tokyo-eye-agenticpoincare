# Dehydron Barcode Input Channel — Tasks

**Design:** [`design.md`](design.md) · **Ablation:** [`ablation.md`](ablation.md)  
**Context (2026-07-16):** Open parameters in design §11 are **resolved**. Scaffolding code exists (`dehydron_barcode_features.py`, Makefile targets, continue/cold launchers) but July defaults (11-dim scalars, `long_lived=2.0`, flat feature-set ids, no isolated cold init, unspecified investigation path) are **not** the ablation contract.

---

## Phase 0 — already present (scaffolding; do not treat as finished science)

- [x] Midpoint extraction + GUDHI witness persistence + scalar/binned aggregation module
- [x] Precompute Makefile target + training attach path + `use_dehydron_barcode` flags
- [x] Ablation Makefile targets (baseline / scalars / full / cold)
- [x] Feature-liveness probe + slim+SSOT refuse path

---

## Phase 1 — before any promotion training (cheap, no model training)

- [x] **Bar-length distribution → long-lived threshold** — **LOCKED 2026-07-16**
  - Script: `experiments/diagnostics/dehydron_bar_length_threshold.py`
  - Make: `make diagnose-dehydron-bar-length` (ran host-side; 12/12 Stage A)
  - **Locked:** `LONG_LIVED_PERSISTENCE_ANGSTROM = 3.11` (carried into
    `dehydron_barcode_v1_2`) via pooled H1 p75; dominance clear (max 12.9% @
    2SHP); H0/H1 similarity clear.
  - Justification SSOT: `checkpoints/v65/diagnostics/dehydron_bar_length_v1/threshold_lock.json`
  - Cache safety verified: writer + loader derive
    `{PDB}_{chain}_{BARCODE_FEATURE_VERSION}.pt`; digest glob and graph-cache
    suffix follow the version string. Stale `v1` / `v1_1` sidecars cannot satisfy
    the current path.

- [x] **Orthogonality diagnostic + first-arm scalar lock** — **LOCKED 2026-07-16**
  - Script: `experiments/diagnostics/dehydron_scalar_orthogonality.py`
  - Make: `make diagnose-dehydron-scalar-orthogonality`
  - SSOT: `checkpoints/v65/diagnostics/dehydron_scalar_orthogonality_v1/`
  - Gate scope: valid/touching payload + structure-level correlations. All-residue
    zero/missing support alignment is retained as telemetry, not mistaken for
    payload redundancy.
  - All five candidates clear `[ρ,τ,ss]` at `|r|<0.70`; peer screen rejects
    `max_persistence_h1` and `num_h1_bars`.
  - Within-SS inspection of borderline `n_dehydrons_touching` vs ρ: helix
    `|ρₛ|=0.697` (n=757) is an **isolated** worst case; sheet/coil ~0.46–0.51 —
    legitimate keep (see `scalar_subset_lock.json`).
  - **Locked shipping set (`dehydron_barcode_v1_2`, `SCALAR_DIM=3`):**
    `total_persistence_h1`, `fraction_long_lived_h1`, `n_dehydrons_touching`.

- [x] **Corpus z-score μ/σ at cache-build** — wired 2026-07-16
  - `compute_corpus_zscore_stats` / `apply_corpus_zscore` in
    `dehydron_barcode_features.py`; two-pass precompute writes
    `corpus_zscore_stats.json` and z-scores sidecars before save.

- [x] **Min-dehydron-count → missing-mask guard** — wired 2026-07-16
  - `MIN_DEHYDRON_MIDPOINTS_FOR_WITNESS = 5` (Stage A-12 min midpoints = 77;
    guard only catches degenerates).
  - `compute_witness_persistence` returns `[]` below threshold → missing-mask path.
  - Unit coverage: `tests/test_dehydron_first_arm_lock.py`.

- [x] Cold-start barcode arms → `isolated_init` — wired for **v66**
  - `GOSPConeMapperV66` → `HyperbolicPrototypeGate(init_seed=…)` →
    `isolated_torch_seed` / `derived_seed` (`tests/test_isolated_gate_init_seed.py`).
  - `gnn_lineage.build_model` forwards `config.init_seed`; launcher sets it from `--seed`.
  - `train-v66-feeler-dbh-scalars` (deprecated) now passes `--seed $(or $(SEED),1)`.
  - Edge arms do not widen `node_emb`; isolation is still verified on the model class.
  - Archive v65 cold targets retain the same `--seed` wiring.

- [x] **Investigation audit scoring path = physics** — enforced 2026-07-17
  - `experiments/diagnostics/investigation_audit_corpus12.py` ranks and
    partitions only by `physics_investigation`, records `scoring_path`, and
    hard-fails if a viewer exposes only legacy evidential `investigation`.
  - The old `cold_start_v8_p3e` viewers were explicitly refusal-tested and
    rejected for lacking the physics field.

---

## Phase 1.5 — architecture (cheap; do alongside Phase 1)

- [ ] **Composable `stack_gnn_node_features`**
  - Confirm or refactor so node features assemble from named, independently versioned blocks (baseline topology ⊕ dehydron_scalars ⊕ …), not a hardcoded concat order.
  - Goal: future `heavy_atom_scalars` is append-a-block, not rewrite.

- [ ] **Composable feature-set version key**
  - Replace / extend flat enums (`master_topology_three_vector_dbh_scalars_v1`, …) with a sorted tuple/hash of active block names + each block’s version.
  - Cache key and MLflow `feature_set` must consume the same composable id.

---

## Phase 2 — train only after Phase 1 green (**plain master-cold three-vector**)

> **Parent lock 2026-07-17 (relaunch):** P1 resumes from
> `checkpoints/v66/runs/master_cold_topology_three_vector_v1/phase_2.pt`
> after a clean cold under `experiments.training.v66.launch_training`
> (`node_emb` width 3, ge200). **Not** Fix-1 routing-stack controlled 3-D/4-D
> seeds; **not** v66 feeler `p3_geom` / edge-barcode; **not** abandoned v65
> `dbh_*` cold; **not** a silently relocated copy of the v6-path reference.
> See [`ablation.md`](ablation.md).

- [x] Re-precompute Stage A-12 sidecars under `dehydron_barcode_v1_2`
  (12/12; 3-scalar + corpus z-norm + long-lived 3.11 + min-dehydron=5).
  Z-score scope is the 2,380 MASTER training-graph rows; PDB-only 1TEN:802 is
  excluded from μ/σ because it has **no Cα** (ARG with only C/O atoms) and is
  dropped by `build_node_features` — 1TEN still trains as an 89-residue graph
  (corpus remains 12/12, not 11-and-a-partial). SSOT:
  `checkpoints/v65/diagnostics/dehydron_barcode_v1_2/zscore_row_exclusions.json`.
  Current sidecar digest: `23bce163b2d3db8a`.
- [x] Provenance decision: do **not** graft
  `checkpoints/v6/runs/master_cold_topology_three_vector_v1/phase_2.pt` into
  `checkpoints/v66/` — relaunch under v66 entrypoint instead (option 1).
  Reference artifact stays under `checkpoints/v6/` with the mislabel documented.
- [x] Makefile target: `train-v66-master-cold-topology-three-vector`.
- [~] Relaunch parent: `make train-v66-master-cold-topology-three-vector
  RUN_ID=master_cold_topology_three_vector_v1 SEED=1` — **running**
  (log `/tmp/v66_master_cold_topology_three_vector_v1.log`; verified
  `Training node_dim=3`, lineage v6.6, no feeler/routing-stack flags).
  Gate stamp refreshed 2026-07-17 after hash mismatch. Await `phase_2.pt`
  under `checkpoints/v66/runs/master_cold_topology_three_vector_v1/`.
- [ ] Rewire Makefile barcode continue/cold targets to the **v66** parent
  (refuse feeler / `fix1_s4_stack_*` / v6-path reference as defaults).
- [ ] Matched continues: **baseline vs Scalars** off locked v66 `phase_2.pt`
  (`--master-cold-lineage`, no routing-stack flags).
- [ ] Promotion decision per ablation checklist (**physics** scoring). Full /
  binned deferred. Routing-stack-on-top and feeler edge barcode are **out of
  P1 scope**.
- [~] Archive: v65 `dbh_*` cold, v66 `p3_geom-edges` shrink-run, controlled
  routing-stack seeds as P1 parents — discarded for this question.

---

## Deferred (explicit non-work)

- Full / 2D birth–death image / projector MLP until Scalars verdict.
- H0 family and birth/death means until second-pass orthogonality.
- Heavy-atom multi-pair complexes (size compute first; MVP would be C–N + C–O only).
- Onboard-contract / Normalizer promotion.
- Routing purity / Gram levers (closed as not-solved-at-this-scale).

---

## Diagnostic script order (recommended)

1. **Bar-length distribution** — done (`make diagnose-dehydron-bar-length`).
2. **Orthogonality + subset lock** — done
   (`make diagnose-dehydron-scalar-orthogonality` → `scalar_subset_lock.json`).
3. **Corpus z-score + min-dehydron guard** — wired in `dehydron_barcode_v1_2`.
4. **Stage A re-precompute + path/digest verification** — done (12/12).
5. **isolated_init + physics audit enforcement** — done.
6. Training is next on **plain master-cold three-vector** parent
   (`ablation.md`); requires Docker science + Stage A-12 `v1_2` sidecars.
