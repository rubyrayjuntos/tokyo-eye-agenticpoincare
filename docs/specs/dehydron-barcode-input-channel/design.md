# Dehydron Barcode Input Channel — Design (P1)

**Status:** Approved; open parameters **resolved 2026-07-16** (post routing-track lock / T1a lessons)
**Decisions locked:** Filtration **A** (Euclidean dehydron-midpoint witness); payload **C** (scalars always-on, binned vector deferred)
**Related:** [`../gnn-topology-input/design.md`](../gnn-topology-input/design.md), [`ablation.md`](ablation.md), [`tasks.md`](tasks.md), legacy `science/dtie/v3/phases/phase3_witness_persistence.py`
**Standing context:** Routing sensitivity + commitment banked; purity/Gram closed as not-solved-at-this-scale — see `docs/audit/GNNV7_SUCCESS_CRITERIA.md`. This channel is the next enrichment track, not another routing lever.

---

## 1. Goal

Promote **persistent dehydron barcodes** from a post-inference / legacy analysis artifact into a **cacheable training input channel**, so the GNN can learn representations that respect thermodynamic wrapping topology rather than only discovering dehydron-adjacent sites after the fact (via uncertainty / investigation).

P1 answers a single experimental question:

> Do residue-level dehydron persistence features improve representation quality over the current `[ρ, τ, ss]` topology-three-vector baseline?

---

## 2. Non-goals (P1)

| Deferred | Why |
|----------|-----|
| Hyperbolic witness/Rips on learned coords as **training input** (option B) | Couples to embedding quality; reserved for viewer/interpretability |
| S2F-style multimodal pre-training | Separate program |
| Onboard-contract ingest job + Normalizer promotion | After ablation wins |
| Multi-relational dehydron edges | After scalars win |
| Loss / MoE / geometry-freeze recipe changes | Keep representation ablation clean |
| Ripser / Alpha complexes | Not in repo; witness (GUDHI) is SSOT for P1 |

**Product shape (hybrid C):** A is the training SSOT; B remains a post-hoc viewer tool on final/intermediate model coordinates.

---

## 3. Background — what exists today

| Layer | Status |
|-------|--------|
| ρ wrapping count + τ dehydron flag | Already GNN node inputs (`topology_three_vector`) |
| Dehydron as loss/gate (`tau_dehydron_rim`, cone probes) | Active in slim MoE training |
| Euclidean dehydron-midpoint witness persistence (v3) | Code exists; not wired to training or current ingest DAG |
| `fact_phase3_persistence` + Normalizer | Schema ready; under-populated by production jobs |
| Training corpus cache (`graphs_*.pt`) | ρ/τ/ss/(sasa) only — **no barcodes** |

Investigation audits (corpus-12 + literature spot-check) show the model already flags dehydron-adjacent functional sites via **investigation**, not raw epistemic. P1 supplies the missing **mechanism language** (persistent underwrapping topology) as input.

---

## 4. Computation SSOT (locked A)

### 4.1 Points

- Dehydron **N–O midpoints** from existing wrapping detection (under-wrapped backbone H-bonds).
- Same physical objects as v3 Phase 3 witnesses (`NO_Midpoints`).

### 4.2 Filtration

- Euclidean **witness complex** (GUDHI), adapted from `science/dtie/v3/phases/phase3_witness_persistence.py`.
- Default `max_alpha ≈ 20.0` Å (config-overridable).
- Homology: **H0 + H1** computed; **first Scalars arm uses H1-focused summaries only** (H0 family deferred to a second pass).
- Noise filter: drop bars with persistence below a config threshold (default floor in `0.01–0.25` Å; exact value may stay at the July `0.1` Å until bar-length diagnostics say otherwise).
- **Long-lived threshold:** do **not** ship a round guessed Å. Compute barcodes for Stage A-12 first (pure TDA), plot the pooled bar-length distribution, and set the threshold at a natural break or a fixed percentile (e.g. 75th) of that distribution. Record the justification in cache metadata and `tasks.md`.

### 4.3 Landmark selection (witness subsample)

- **Default:** reuse v3 Phase 1 landmark selection (k-means on midpoint coords) — do not silently diverge from the v3 contract.
- **Guard (new, required):** if dehydron midpoint count is below a minimum threshold, **skip** landmark subsampling and route the structure to the missing-mask path (§5.3). Do not let a 2–3-dehydron structure produce a garbage/trivial barcode that looks like valid data downstream.

### 4.4 Independence

- Computed from **PDB geometry only**.
- Must not depend on GNN embeddings, MoE routing, or learned curvature.
- Safe to cache once per `(structure_id, chain, feature_version)` and reuse across experiments.

---

## 5. Payload (locked C)

### 5.1 Always-on scalars (first arm: small orthogonal subset)

P1 computes one structure-level WitnessComplex over all dehydron midpoints in the chain.
The resulting barcode summary scalars are broadcast to residues that **touch** at least
one dehydron (donor and/or acceptor); only `n_dehydrons_touching` is residue-local.

**Normalization (locked):** corpus z-score, computed **once at cache-build time**, stored in
versioned sidecar metadata — **not** per-structure. Per-structure normalization would flatten
genuine cross-structure topological richness and risks division-by-zero on sparse/empty
dehydron sets. This is directly downstream of the T1a lesson (raw SASA std≈46 vs τ≈0.5
silently dominated `node_emb`).

**First Scalars arm — ship only after orthogonality clears (start smaller than July’s 10–16):**

| Ship candidates | Role |
|-----------------|------|
| `total_persistence`, `max_persistence` (H1) | Persistence mass / peak |
| `num_H1_bars` | Count |
| `fraction_long_lived` | Lifetime (threshold from §4.2 diagnostic) |
| `n_dehydrons_touching` | Residue-local only |

**Defer to a second pass:** `mean_birth` / `mean_death`, H0 family, and any scalar that fails
the redundancy check against `[ρ, τ, ss]` or against peer candidates (marginal + within-SS-class,
same method as the SASA orthogonality check). Fewer verified-orthogonal dims beat a wider
partially-redundant set — trunk rank ceiling is still ~1.7–1.8.

> **Note:** First-arm shipping contract is `dehydron_barcode_v1_2`
> (`SCALAR_DIM=3`): `total_persistence_h1`, `fraction_long_lived_h1`,
> `n_dehydrons_touching`, with `LONG_LIVED_PERSISTENCE_ANGSTROM = 3.11` and
> corpus z-score at cache-build. Lock SSOT:
> `checkpoints/v65/diagnostics/dehydron_scalar_orthogonality_v1/scalar_subset_lock.json`.
> Do not treat the July scaffolding 11-dim `SCALAR_NAMES` list as the shipping set.

This intentionally does **not** produce true per-midpoint or per-dehydron bar sets:
structure-level witness persistence does not associate bars back to individual
midpoints. A future post-P1 local-association pass may add that mapping if the
ablation warrants it.

### 5.2 Optional binned vector — **deferred**

- Config: `use_binned_dehydron: bool` stays **`false`** until the Scalars arm has a verified result.
- If Full is attempted later: start with a **1D H1 persistence histogram**, not a 2D birth–death image.
- A 2D image (and any projector MLP) is only justified once scalars have shown they are not enough.

### 5.3 Missing / failed TDA

- Fill barcode features with **zeros**.
- Set a boolean / float mask channel `dehydron_barcode_missing` so “no dehydrons / TDA failed” is not silently identical to “dehydrons present but zero persistence.”

---

## 6. Graph integration

### 6.1 Node features

Current baseline (`GNN_INPUT_MODE=topology_three_vector`):

```text
data.x = [ρ, τ, ss]          # [N, 3]
data.sasa = …                # side-channel (unchanged)
```

P1 augmented:

```text
data.x = [ρ, τ, ss] ⊕ dehydron_scalars ⊕ [missing_mask]
# (+ optional binned later; node_dim must match node_emb)
```

**Composable assembly (required, cheap, do alongside P1):**

1. Keep `stack_gnn_node_features` (or its successor) as a list of **named, independently versioned feature blocks** — not a hardcoded concat order. Future `heavy_atom_scalars` then appends a block instead of rewriting paths / re-deriving every cache key.
2. Version the feature-set identifier as a **composable key** (sorted tuple/hash of active block names + each block’s version), not a flat enum (`master_topology_three_vector_dbh_scalars_v1`, …). Flat enums become combinatorial as soon as a second optional modality exists.

Today `gnn_feature_set_id_for_barcode` is still a flat string suffix — treat migrating that as Phase 1.5 in [`tasks.md`](tasks.md).

- **No** dehydron-typed edges in P1 node-feature ablation.

### 6.2 Checkpoint / warm-start / cold-start policy

| Path | Policy |
|------|--------|
| **Matched continues** (`--resume`) | Safe by construction: overlap-copy `node_emb` weights, **zero-fill** new columns, no fresh RNG draw for the bank. Edge barcode continues do not widen `node_emb`. |
| **Cold-start / any arm that widens `node_dim`** | **Must** pass `--seed` → `TrainingConfig.init_seed` → `GOSPConeMapperV66` / `HyperbolicPrototypeGate(init_seed=…)` which uses `isolated_torch_seed` / `derived_seed` (`science/dtie/common/isolated_init.py`). Verified for v66 in `tests/test_isolated_gate_init_seed.py`. Without `init_seed`, widening `node_emb` silently shifts gate/prototype RNG (same confound as the 3-vs-4 ablation). |

Existing `node_emb` in_features = 3 checkpoints remain valid for baseline arms only.

### 6.3 Slot-in for heavy-atom / side-chain enrichment (no redesign later)

Flag now, do not build yet:

- Heavy-atom point clouds imply **up to six** atom-pair witness/Rips complexes per structure (C–C, N–N, O–O, C–N, C–O, N–O) vs one dehydron-midpoint complex — real compute multiplier; size before committing.
- Reasonable MVP-within-MVP later: start with **C–N and C–O** only (H-bond geometry orthogonal to ρ), not all six.
- Shape target: additive module mirroring `dehydron_barcode_features.py`, not a redesign of assembly.

---

## 7. Caching and versioning

| Item | Spec |
|------|------|
| Location | Corpus sidecar and/or fields inside bumped `graphs_*.pt` cache key |
| Key | `(pdb_id, chain, barcode_feature_version)` — version must encode scalar subset + norm stats + long-lived threshold |
| Metadata | structure_id, chain, code version, `max_alpha`, noise/long-lived thresholds **with justification**, corpus z-score μ/σ, binning (if any), aggregation rule, GUDHI version, min-dehydron guard |
| Training loop | **Read-only** from cache — no GUDHI inside the epoch loop |
| Recompute | Explicit Makefile / script target when feature version bumps |

P1 does **not** require Normalizer / `fact_phase3_persistence` writes. Promote to ingest after ablation success.

---

## 8. Training / ablation plan

| Arm | Node features | Purpose |
|-----|---------------|---------|
| **Baseline** | `[ρ, τ, ss]` | Control |
| **Scalars** | + orthogonal dehydron scalars (+ missing mask) | “Do barcodes help at all?” |
| **Full** | + scalars + binned | **Deferred** until Scalars verdict (then 1D histogram first) |

### 8.1 Training constraints

- **P1 parent (locked 2026-07-17; relaunch path):** plain topology-three-vector
  master-cold under the **v6.6** entrypoint —
  `checkpoints/v66/runs/master_cold_topology_three_vector_v1/phase_2.pt`
  (`node_emb` width **3**, ge200). See [`ablation.md`](ablation.md).
  This is the clean representation baseline (§2: no loss/MoE/geometry-freeze /
  routing-stack / feeler recipe changes).
- **Reference only:** `checkpoints/v6/runs/master_cold_topology_three_vector_v1/phase_2.pt`
  was the same recipe launched via the **v6** entrypoint (labeling mistake, not
  architecture mistake). Do **not** silently relocate it under `checkpoints/v66/`.
- **Not P1 parents:** Fix-1+S4+SASA-gate+repulsion+scale-L2 controlled 3-D/4-D
  seeds (`fix1_s4_stack_initseed_controlled_*`); v66 feeler `p3_geom` / edge-barcode
  arms; abandoned v65 `dbh_*` cold runs. Those answer different questions.
- **Primary P1 channel:** node-global `--use-dehydron-barcode` (scalars; Full deferred).
  Feeler `--dehydron-edge-barcode` is a **separate** post-P1 track.
- Start from a **learned GNN** parent (`--master-cold-lineage`), **not** slim MoE +
  structural SSOT. See [`docs/audit/LEARNED_GNN_VS_SLIM_SSOT.md`](../../audit/LEARNED_GNN_VS_SLIM_SSOT.md).
- Do not combine barcode flags with `--slim-moe-structural-ssot` (launcher refuses unless `--allow-dead-feature-channel`).
- Prefer short controlled continues so arms are comparable; require `--feature-liveness-probe` green before claiming influence.
- Any cold arm that **widens** `node_emb` requires `isolated_init` (§6.2).

### 8.2 Success / fail criteria

**Investigation scoring surface (locked):** all investigation audits for this ablation
(`investigation_audit_corpus12.json`, 4OBE motif spot-checks) **must** use
`physics_investigation` (ρ/τ underwrap) or cone_depth/τ directly — **not** the evidential
`ale × (1−epi)` path. Evidential Investigation is a near-duplicated ρ-proxy with low dynamic
range (`docs/audit/VIEWER_INVESTIGATION_CORRECTNESS.md`); promoting on that metric would
re-enter a demonstrated failure mode through a different door.

**Success (promote scalars to default training feature; consider ingest):**

- 4OBE **physics** investigation motifs retained or improved (Switch I/II, α3 105–107, C-term pivot).
- Corpus-12: high-inv (physics path) remains rim-enriched (`depth_hi > depth_lo`) on ≥11/12 structures.
- Cone / τ probes not regressing vs baseline.
- MoE routing health not worse (no new starvation / eligibility collapse attributable to features).
- Feature-liveness probe green (`liveness_barcode_alive=1`).

**Fail:**

- Keep `use_dehydron_barcode` off by default.
- Do not promote to onboard contract.
- Optionally keep cache pipeline for viewer-only use.

---

## 9. Deferred work (post-P1)

1. **B — Hyperbolic barcodes for viewer:** run witness/Rips on learned Poincaré coords; overlay on disc / split viewer.
2. **Ingest promotion:** production job → Normalizer → `fact_phase3_persistence` + onboard-contract artifact + readiness probes.
3. **Edge channel (post-P1 / feeler track):** typed / local dehydron edge barcode —
   separate from P1 node-global scalars. Do not substitute for the plain
   master-cold baseline/scalars ablation.
4. **S2F-style multimodal pretrain:** dehydron topology as an extra structure modality channel.
5. **Sequence-only / predicted-structure proxies:** zero/mask or ESMFold-derived midpoints (explicitly out of P1).

---

## 10. Code map (implementation targets)

| Component | Role |
|-----------|------|
| `science/dtie/v3/phases/phase3_witness_persistence.py` | Reference filtration (adapt, do not silently change v3 contract) |
| New: `science/dtie/common/dehydron_barcode_features.py` (name TBD) | Midpoints → barcode → per-residue scalars/binned vectors |
| New: precompute script / Makefile target | Corpus cache builder |
| `science/dtie/common/residue_features.py` / graph assembly | Concat into `data.x`; feature-set id |
| `experiments/training/v6/_data.py` / corpus cache | Load barcode fields; bump cache key |
| `science/dtie/v66/gnn/model.py` (`GOSPConeMapperV66`) | P1 parent / continue model class (plain master-cold three-vector; no feeler flags) |
| `science/dtie/v6/gnn/model.py` (`GOSPConeMapperV6`) | Weight-compatible base used by the mislabeled v6-path reference run only |
| Shared `HyperbolicPrototypeGate` + `isolated_init` | Required when cold arms widen `node_emb` |
| `science/training/config.py` / launch flags | `use_dehydron_barcode`, `use_binned_dehydron`, thresholds |
| Tests | Unit: aggregation, missing mask, dim; golden: one PDB barcode stable under version pin |

---

## 11. Open parameters — **resolved 2026-07-16**

| # | Parameter | Resolution |
|---|-----------|------------|
| 1 | Scalar list + normalization | **Locked 2026-07-16 (`dehydron_barcode_v1_2`):** `total_persistence_h1`, `fraction_long_lived_h1`, `n_dehydrons_touching` + **corpus z-score at cache-build**. Within-SS inspection: helix `|ρₛ|=0.697` for `n_dehydrons_touching` is an isolated worst case (sheet/coil ~0.46–0.51) — keep. Peer reject: `max_persistence_h1` / `num_h1_bars`. SSOT: `…/dehydron_scalar_orthogonality_v1/scalar_subset_lock.json`. |
| 2 | Long-lived threshold | **Locked 2026-07-16:** `3.11 Å` = Stage A-12 pooled H1 p75 above 0.1 Å noise (carried into `dehydron_barcode_v1_2`). Dominance clear; no strong upper-half gap. SSOT: `checkpoints/v65/diagnostics/dehydron_bar_length_v1/threshold_lock.json`. |
| 3 | Landmark selection | **Reuse v3 Phase 1** k-means landmarks + **min-dehydron-count → missing-mask** (`MIN_DEHYDRON_MIDPOINTS_FOR_WITNESS = 5`; Stage A-12 min midpoints = 77). |
| 4 | Binned vector shape | **Defer entire Full arm** until Scalars verified. If later: 1D H1 histogram first, not 2D image. |
| 5 | Warm-start / cold-start | Continues: overlap-copy + zero-fill. **Cold barcode arms: mandatory `isolated_init`** (`isolated_torch_seed` / `derived_seed`) — width RNG-shift confound applies directly. |

### 11.1 Risks added after the July design (load-bearing)

**A. `node_emb` width RNG-shift on cold arms.** Changing `node_dim` at construction time without isolated seeding produces unrelated prototype-bank inits under a shared seed label. Already fooled the width ablation for weeks. Any v66 arm that widens `node_emb` (deprecated node-global scalars) **must** pass `--seed` so `init_seed` reaches `HyperbolicPrototypeGate`. Edge barcode does not widen `node_emb`.

**B. Ablation success criteria must not use evidential Investigation.** Score with `physics_investigation` / cone_depth/τ. See §8.2 and `VIEWER_INVESTIGATION_CORRECTNESS.md`.

**C. Design-doc lineage / parent drift (caught 2026-07-17).** Successive drafts pointed P1 at (1) abandoned v65 `dbh_*` cold, (2) v66 feeler edge barcode / `p3_geom`, (3) Fix-1 routing-stack controlled 3-D seeds. Each was a real experiment answering a **different** question. P1 parent is locked to plain topology-three-vector master-cold under **v6.6** tagging — see [`ablation.md`](ablation.md). Same class of silent drift as env-forwarding / stale colormap / hardcoded `v1` cache paths.

**D. Entrypoint mislabel (caught 2026-07-17).** Plain three-vector cold was first trained via `v6.launch_training` into `checkpoints/v6/…`. Content was right; lineage/path wrong. Decision: **relaunch** via `v66.launch_training` rather than graft the checkpoint into `checkpoints/v66/` without a paper trail.

### 11.2 Still open (operational, not design blockers)

- Whether matched continues or isolated cold arms are the primary promotion evidence (prefer **both** once Scalars subset is frozen; cold without isolation is invalid).
- Composable feature-set version key (Phase 1.5) before a second optional modality lands.

---

## 12. Approval record

| Decision | Choice |
|----------|--------|
| Filtration SSOT | **A** — Euclidean dehydron-midpoint witness |
| Payload | **C** — scalars always; binned deferred until Scalars win |
| First integration | Node features + corpus cache |
| First experiment | Baseline vs **orthogonal Scalars** (Full deferred) |
| Approved | 2026-07-09 |
| Open params resolved | 2026-07-16 (T1a / width-ablation / Investigation lessons) |

---

## 13. Downstream edge work (sequencing; locked 2026-07-17)

P1 node-global scalars do **not** create residue↔residue dehydron association.
Next chemical/topology edge work is ordered as:

1. **Chem-MVP `_struct_conn`** — bridge `fact_covalent_bond` into existing multi-rel MP
   ([`../struct-conn-typed-edges/design.md`](../struct-conn-typed-edges/design.md),
   [`../struct-conn-typed-edges/ablation.md`](../struct-conn-typed-edges/ablation.md)).
2. Optional: dehydron **typed shared-bar edges** (this design §9) after Chem-MVP.
3. **Hierarchical containment** — Chem-MVP closed; now **next buildable** under
   flow-motivated design v2 (diameter-stratified asymmetry acceptance) with the
   depth-collision lock unchanged
   ([design](../hierarchical-containment-edges/design.md),
   [`ablation.md`](../hierarchical-containment-edges/ablation.md),
   [`depth-collision.md`](../hierarchical-containment-edges/depth-collision.md)).

Chem-Full is a separately registered optional ingest-extension branch after
Chem-MVP; it does not alter the primary Chem-MVP → containment → Path 2
(directionality reward on diam ≤9) sequence. Typed shared-bar remains optional.
