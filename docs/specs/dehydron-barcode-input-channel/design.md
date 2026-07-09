# Dehydron Barcode Input Channel — Design (P1)

**Status:** Approved for implementation planning (2026-07-09)  
**Decisions locked:** Filtration **A** (Euclidean dehydron-midpoint witness); payload **C** (scalars always-on, binned vector optional)  
**Related:** [`../gnn-topology-input/design.md`](../gnn-topology-input/design.md), legacy `science/dtie/v3/phases/phase3_witness_persistence.py`

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
- Homology: **H0 + H1** (H1 is the primary leak/allostery signal; H0 retained for completeness).
- Noise filter: drop bars with persistence below a config threshold (start in `0.01–0.25` Å range; tune in ablation).

### 4.3 Independence

- Computed from **PDB geometry only**.
- Must not depend on GNN embeddings, MoE routing, or learned curvature.
- Safe to cache once per `(structure_id, chain, feature_version)` and reuse across experiments.

---

## 5. Payload (locked C)

### 5.1 Always-on scalars (~10–16 dims, normalized)

Per residue, aggregated over dehydrons that **touch** the residue (donor and/or acceptor):

| Feature family | Examples |
|----------------|----------|
| Counts | `#bars`, `#H1 bars` |
| Persistence | total / max / mean / std |
| Lifetime | fraction of bars above long-lived threshold |
| Birth/death | mean birth, mean death (H1-focused) |
| Optional | max H1 persistence only |

Aggregation default: **mean** over touching dehydrons for distributional stats; **max** for “strongest local defect” scalars; **sum** for counts. Exact mix is config-documented and versioned in metadata.

### 5.2 Optional binned vector (flag-controlled)

- Config: `use_binned_dehydron: bool` (default **`false`** for first training arms).
- Persistence histogram and/or birth–death image, **~32–64 dims**, **0.25 Å** bins.
- Same per-residue aggregation rule as scalars.
- If dim ≳ 32, optional small MLP projector before concat into `node_emb` input (ablation: projector on/off).

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
data.x = [ρ, τ, ss] ⊕ dehydron_scalars ⊕ [dehydron_binned?] ⊕ [missing_mask]
# node_dim increases; model.node_emb in_features must match (new runs only)
```

- Extend `stack_gnn_node_features` / training graph assembly; bump feature-set id for MLflow governance.
- **No** dehydron-typed edges in P1.

### 6.2 Checkpoint compatibility

- Existing checkpoints with `node_emb` in_features = 3 remain valid for baseline arms.
- Barcode arms are **new runs** (or warm-start with resized `node_emb` — prefer new cold/warm policy documented in the implementation plan; default = train barcode arms with matching `node_dim` from init or explicit resize recipe).

---

## 7. Caching and versioning

| Item | Spec |
|------|------|
| Location | Corpus sidecar and/or fields inside bumped `graphs_*.pt` cache key |
| Key | `(pdb_id, chain, barcode_feature_version)` |
| Metadata | structure_id, chain, code version, `max_alpha`, noise/long-lived thresholds, binning, aggregation rule, GUDHI version |
| Training loop | **Read-only** from cache — no GUDHI inside the epoch loop |
| Recompute | Explicit Makefile / script target when feature version bumps |

P1 does **not** require Normalizer / `fact_phase3_persistence` writes. Promote to ingest after ablation success.

---

## 8. Training / ablation plan

| Arm | Node features | Purpose |
|-----|---------------|---------|
| **Baseline** | `[ρ, τ, ss]` | Control |
| **Scalars** | + dehydron scalars (+ missing mask) | “Do barcodes help at all?” |
| **Full** | + scalars + binned | Richness check |

### 8.1 Training constraints

- Start from a **fixed** MoE / geometry lineage (recommended: `cold_start_v8_p3e` or `cold_start_v8_p2` best — representation ablation, not another routing round).
- Do not change slim MoE timeout/freeze recipe, cone loss coeffs, or unfreeze structural disc in the same experiment.
- Prefer short controlled continues or matched-epoch restarts so arms are comparable.

### 8.2 Success / fail criteria

**Success (promote scalars to default training feature; consider ingest):**

- 4OBE investigation motifs retained or improved (Switch I/II, α3 105–107, C-term pivot).
- Corpus-12: high-inv remains rim-enriched (`depth_hi > depth_lo`) on ≥11/12 structures.
- Cone / τ probes not regressing vs baseline.
- MoE routing health not worse (no new starvation / eligibility collapse attributable to features).

**Fail:**

- Keep `use_dehydron_barcode` off by default.
- Do not promote to onboard contract.
- Optionally keep cache pipeline for viewer-only use.

---

## 9. Deferred work (post-P1)

1. **B — Hyperbolic barcodes for viewer:** run witness/Rips on learned Poincaré coords; overlay on disc / split viewer.
2. **Ingest promotion:** production job → Normalizer → `fact_phase3_persistence` + onboard-contract artifact + readiness probes.
3. **Edge channel:** typed edges for residue pairs sharing a persistent dehydron.
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
| `science/dtie/v6/gnn/model.py` | `node_dim` / `node_emb` for new runs |
| `science/training/config.py` / launch flags | `use_dehydron_barcode`, `use_binned_dehydron`, thresholds |
| Tests | Unit: aggregation, missing mask, dim; golden: one PDB barcode stable under version pin |

---

## 11. Open parameters (set in implementation plan, not blockers)

- Exact scalar list and normalization (z-score vs corpus vs per-structure).
- Long-lived persistence threshold (Å).
- Landmark selection for witness complex (reuse v3 Phase 1 landmarks vs dehydron-only landmarks).
- Whether binned vector is persistence-only histogram vs birth–death 2D image flattened.
- Warm-start policy when expanding `node_emb`.

---

## 12. Approval record

| Decision | Choice |
|----------|--------|
| Filtration SSOT | **A** — Euclidean dehydron-midpoint witness |
| Payload | **C** — scalars always; binned behind flag |
| First integration | Node features + corpus cache |
| First experiment | Baseline vs scalars vs full ablation |
| Approved | 2026-07-09 |
