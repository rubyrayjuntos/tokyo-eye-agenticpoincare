# Tokyo Eye Next (v8) — Equiformer → Typed Sparse Hyp Stack

**Status:** FROZEN (signed off 2026-07-22) · **AMENDED 2026-09-15** (`tokyo_eye_equ_pure_hyp_v1`) · **RECONCILIATION ADDENDUM SIGNED 2026-09-16** (`tokyo_eye_v8_freeze_reconciliation_v1`)  
**Date:** 2026-07-22  
**Authoring context:** Clean-break modernization after v7 opacity / Cα-only MP / soft MoE / parked uncertainty  
**Supersedes for new work:** piecemeal continues on sealed/cold Hyp-MP funnel (parked archaeology)  
**Does not overwrite:** `HEALTHY_V7_CKPT`, Fix-1 champions  

**Amendment (2026-09-15):** Pure hyperbolic after lift — Einstein/Klein barycenters OK; tangent Linear/pool/mix as geometry substitute **forbidden**. Spec: [`2026-09-15-tokyoeye-equ-pure-hyp-freeze-amendment.md`](2026-09-15-tokyoeye-equ-pure-hyp-freeze-amendment.md) · Gate: `data/gates/tokyo_eye_equ_pure_hyp_freeze_amendment.json`. Without this, “pure hyp” is a lie relative to the code path.

**Reconciliation addendum (2026-09-16, SIGNED Ray Swan):** Rebuild against these clauses, not a blank-page redesign. Dispositions (REVERT / AMEND / KEEP) and this week’s earned additions (eval-mode MoE gate, dual pre/post-MoE geometry seals, MPtrj-inert frontend default, trunk-wide `pure_hyp_pass`): [`2026-09-16-tokyo-eye-v8-freeze-addendum.md`](2026-09-16-tokyo-eye-v8-freeze-addendum.md) · Gate: `data/gates/tokyo_eye_v8_freeze_addendum.json`. Signed: Option B lift REVERT; `lr_hyperbolic=1e-3` REVERT; MLflow canonical `tokyoeye/equiformer-v3-moe/geometric/full-stack` AMEND.

**Related:** user PDF *TokyoEyesHyperbolicV6* (ideas source; not code SSOT) · prior R0–R5 lock-in chat · E0–E3 MoE guild theory  

**Boundary:** Model + train/inference graph builder only. **Do not rewrite** PDB/CIF download or ingest. Atoms already land via existing RCSB `.cif.gz` path; R0–R5 is built in-process (same pattern as `biology_graph.py`). Persist via Normalizer only if later promoted for agent/viz reuse.

---

## 1. One-line

**EquiformerV3** extracts SE(3)-aware physical features; a **RadialAngularProjector** lifts into a single Poincaré ball; **relation-aware sparse hyperbolic attention** communicates only on a **frozen R0–R5 typed graph**; **E0–E3 MoE** specializes post-transport; **first-class Poincaré / flow telemetry** makes the model inspectable. Uncertainty is **in-lineage** (radius-aware evidential), not bolted onto sealed v7.

---

## 2. Why this exists

| v7 / cold failure | v8 response |
|-------------------|-------------|
| Bifurcated Euc construction vs hyp “views” fighting | One lift → one hyp refinement SSOT |
| Cα-only MP washes theory | One-shot R0–R5 sparse ontology (unalterable) |
| Soft MoE mush / E1 monopoly | Hard-commitment E0–E3 guilds + anti-monopoly |
| Blind to geometry | Designed-in diagnostics (not reactive probes) |
| Uncertainty parked / broken | New lineage evidential head + radius curriculum |
| Dense N×N hyp attention | Forbidden in MVP; sparse `edge_index` + `edge_type` only |

**Principle:** Without unified geometry + a complete communication graph, scorecards do not compose into a trustworthy model.

---

## 3. Lineage & isolation

| Field | Value |
|-------|--------|
| Working name | **TokyoEye-v8** (marketing name “HyperbolicV6” deferred; avoid colliding with frozen v6.x) |
| Module home | `science/tokyo_eye/v8/` (isolated package; do **not** mix with v7 `TokyoEye.py` / v66) |
| Checkpoints | `checkpoints/v8/` |
| MLflow | experiment `tokyoeye/equiformer-v3-moe/geometric/full-stack` (historical aliases: `tokyo-eyes-v8`, `tokyoeye/equiformer-v3-moe/geometric/hyperbolic-spine`) |
| Compare-only | `HEALTHY_V7_CKPT`, cold/angfill parks |
| Promote | Explicit gate + human promote only |

---

## 4. End-to-end spine (locked)

```
PDB / all-atom batch
  → EquiformerV3 (SE(3) scalars + vectors)
  → RadialAngularProjector (Option B: v = α · â)
  → expmap₀ → project_to_ball (curriculum τ_ceiling)
  → HyperbolicGraphAttention × L (sparse, relation-aware) on R0–R5
  → MoE gate + E0–E3 experts (post-hyp, hard commitment)
  → Heads (disc / mechanism / evidential) + Dual-space outputs as needed
  → PoincaréDiagnosticsEngine + TransparencyEngine (every train step)
```

**Hard geometric rule:** Message passing / hyp attention **only** along edges present in the R0–R5 graph. No dense N×N geodesic matrices in MVP. No silent Cα-only fallback that bypasses typed edges.

---

## 5. Communication graph contract (UNALTERABLE)

### 5.1 Relation IDs

| R | Name | Role | Gate (MVP) |
|---|------|------|------------|
| **0** | Covalent / peptide | Backbone conduit | Δseq = ±1 |
| **1** | Stable H-bond | Sheltered contacts | Geometry + DSSP-style energy ≤ −0.5 kcal/mol |
| **2** | Dehydron | Underwrapped sticky H-bonds | ≤ **1** wrapping non-polar carbons in double-cone (45° / 6.5Å; addendum §2.7 — AMEND from classical sphere-calibrated 19) |
| **3** | Hydrophobic / π–π | Packing / aromatic | Non-polar centroid ≤ 5.0 Å or planar face alignment |
| **4** | Salt bridge | Charged links | Asp/Glu O … Lys/Arg/His N ≤ 4.0 Å |
| **5** | Local neighborhood | Residual spatial envelope | Cβ–Cβ ≤ 8.0 Å **only if** not claimed by R1–R4 |

R5 is **demoted**: geometric proximity never co-equal with R1–R4 chemistry.

### 5.2 Exclusivity & priority resolvers (mandatory)

**Dual-layer extraction (R0 vs R1–R5):** The loader is **two passes**, not one squeezed primary ID:

1. **Layer A — R0 backbone:** For every sequential neighbor pair (`Δseq = ±1`), always emit directed R0 edges `(i→j)` and `(j→i)`.  
2. **Layer B — non-covalent R1–R5:** Independently evaluate chemistry/proximity (below).  

If `i,j` are sequence-adjacent **and** form a stable H-bond (or any R1–R4), the loader **must emit both** the R0 directed pair **and** the winning non-covalent directed pair (same `(i,j)` indices appear **twice** in `edge_index` with different `edge_type`). Squeezing into a single 1D primary type for that undirected pair is **forbidden** — it erases the backbone conduit and breaks sequential message flow.

**Descending evaluation (Layer B only)** for each unordered pair `{i,j}`:

1. Resolve **R1 vs R2** via wrap constraint (dehydron vs stable H-bond).  
2. Else **R3** if hydrophobic / π criteria met.  
3. Else **R4** if salt-bridge criteria met.  
4. Else **R5** if Cβ–Cβ ≤ 8 Å.  
5. Else **no Layer-B edge**.

**Overlap rule (Layer B only):** If a pair qualifies for multiple of R1–R4 (e.g. aromatic + H-bond), assign the **lowest numeric primary ID** as that Layer-B `edge_type`. Secondary chemistry is stored only as **static edge feature attributes** (bitflags / scalars) — **never** duplicate Layer-B indices with different R for the same undirected pair. **R0 is outside this rule** (coexists via dual-layer).

### 5.3 Bidirectional invariance

Store **directed symmetric pairs**: for every undirected relationship `{i,j}` with type R, emit both `(i→j)` and `(j→i)` with **identical** `R`. Prevents ordering bias in geodesic attention / Einstein midpoint aggregation. Applies to both Layer A and Layer B rows.

### 5.4 Isolates (Option A)

Residues with **no Layer-B neighbors** still retain R0 if sequence-adjacent to someone. True isolates (no R0 and no R1–R5): **self-transport** only. No fake neighbors. No Einstein barycenter on empty neighborhoods.

### 5.5 Forbidden

- Dense N×N hyp attention in MVP  
- Hard-coded betweenness / “spoke” edge types that short-circuit manifold hierarchy discovery  
- Silent replacement of R1–R4 by R5-only graphs  
- Cα-contact as sole MP ontology  
- Collapsing R0 into R1–R5 for ±1 pairs (single primary type)  
- **(AMENDMENT 2026-09-15)** Post-lift tangent Linear / pool / mix as geometry substitute — see §6.1  

---

## 6. Equiformer front-end & lift

- **Backbone:** EquiformerV3 (or pinned compatible Equiformer-v3 API) producing residue-pooled scalars `s` and vectors `v`.  
- **Projector:** RadialAngularProjector — Option B lift `v_lifted = σ(radial) · angular` scaled into curriculum ball radius, then `expmap₀`.  
- **Curvature:** Prefer **learned `c`** with contract passthrough (platform rule); if MVP pins `c`, document pin and migrate to learned before promote.  
- **Curriculum:** `CurriculumRadiusController` opens `τ_ceiling` from interior (~0.70) → near-boundary (~0.995) so early training cannot park mass on the rim.

### 6.1 Pure hyperbolic after lift (AMENDMENT 2026-09-15 — LOCKED)

**Invariant:** After the single Euclidean → hyperbolic lift, the pipeline **stays in pure hyperbolic geometry** through hyperbolic graph storage / on-manifold heads.

| Allowed | Forbidden (post-lift) |
|---------|------------------------|
| Einstein / Klein (or equivalent valid hyp) barycenters | `exp₀(W · log₀(z))` as Linear on ball points (Q/K/V, output, FFN) |
| Poincaré / hyp distances for logits | Tangent mean/sum pool then `exp₀` as the graph representation |
| Single lift `expmap₀` / `project_to_ball` at the Euc→H boundary | Tangent residual / mix as the primary message path |
| Read-only `log₀` / radius for **diagnostics** | Vendor / third-party layers that leave the manifold silently |
| | Claiming “pure hyp” while any forbidden pattern remains live |

**Gate:** EQU geometry trunk promotion requires `pure_hyp_pass=true` (see amendment + MLflow EQU SSOT run). Pearson / MoE alone **cannot** waive this. After a Fail boot: **correct start** under this amended freeze — not remediation/patch-continue of a tangent spine.

**Known at amendment time (Fail until remediated):** `attention.py` `_tangent_linear` / `W_o(log₀)` / tangent residual; `affinity_head.py` tangent pool.

---

## 7. Relation-aware hyperbolic attention (sparse)

### 7.1 Signature (contract)

```text
forward(z, edge_index, edge_type, edge_attr=None) -> z_out
```

- `edge_index`: `[2, E]` directed (may contain duplicate `(i,j)` with different `edge_type` when R0 coexists with Layer B)  
- `edge_type`: `[E]` in `{0..5}`  

**Sparse masking (mandatory):** Off-graph pairs are **absent from `edge_index`** — they are never materialized. Do **not** allocate an `N × N` logit matrix filled with `−∞`. Attention softmax runs **only over sparse neighborhood slices** of destination nodes via **segmented softmax** (e.g. `torch_geometric.utils.softmax` indexed by `edge_index[1]`). The prose “logit −∞” means “not in the sparse support,” not “write −∞ into a dense board.”

### 7.2 Logits

For each directed edge `(i→j)` with type R:

\[
\mathrm{attn\_logits}_{ij} = \frac{ -d_{\mathbb{H}}(Q_i, K_j)\,\gamma_R + \beta_R }{\sqrt{d_h}}
\]

`γ_R`, `β_R` from learned relation embedding / scalars. Aggregation via **Einstein midpoint / Klein** (or equivalent valid hyp barycenter) — not Euclidean `matmul` on ball points.

**Pure-hyp (amendment):** Q/K/V and output maps must **not** be implemented as `exp₀(W · log₀(z))` tangent Linears. That pattern is a **forbidden** geometry substitute even when aggregation claims Einstein/Klein. Live `_tangent_linear` means **out of pure-hyp Pass**; next train must be a **correct start**, not a patch of this path.

### 7.3 Depth

L ∈ {2,3} layers; **stop rule:** if layer 4 does not improve E1/E2 guild metrics on holdouts, do not deepen.

---

## 8. MoE guilds (locked)

Post-shared hyp transport only.

| Expert | Niche |
|--------|--------|
| **E0** Wrapped core | high ρ, τ≈0, high degree, H/E |
| **E1** Dehydron rim | τ≈1, lower ρ, coil/loop rich |
| **E2** Interface / spoke | mid ρ/τ, bridge-like degree, mixed SS |
| **E3** Coil / solvent | high coil, variable τ, lower degree |

**Routing:** topology-aware gate (ρ, τ, SS, degree, SASA, ± disc features) + **Gumbel-Softmax cool-down** to hard commitment.

**Hard commitment (mandatory):** During **training**, Gumbel-Softmax sampling **must** use `hard=True` (straight-through estimator). Soft relaxation persisting into the forward path is **forbidden** — it recreates the “Soft MoE mush” failure this lineage exists to eliminate. Temperature cool-down still applies to the Gumbel logits; hardness is not deferred to eval-only.

**Anti-collapse:** load-balance + in-structure majority hinge (and/or routing load floor).  
**Exclude:** angular-wedge experts, PDB-ID routing, SS one-hot copy as sole gate.

---

## 9. Transparency & diagnostics (first-class)

### 9.1 PoincaréDiagnosticsEngine

Per step: mean/max radius, boundary saturation %, Poincaré (radial histogram) entropy, layer-wise entropy. Warnings: core-collapse, boundary-saturation, over-smoothing.

### 9.2 TransparencyEngine

Flow entropy of attention; geodesic attention energy; optional effective resistance on sparse attn graph.

### 9.3 Telemetry boundary hooks (mandatory)

Expose **Rim-to-Core Flow Fractions**:

- Rim nodes: `r > 0.90`  
- Core nodes: `r < 0.30`  
- Metric: fraction / mass of **incoming messages** (attention-weighted) that cross rim→core (and core→rim), plus raw edge counts by R  

Also: message fraction by `edge_type` R0–R5; rim enrichment of R2 vs R1.

Logged to MLflow every epoch; available to Plasticity Explorer for edge-colored geodesic paths.

---

## 10. Uncertainty (in-lineage)

- Evidential / Dirichlet-style head allowed on **v8 only**.  
- Radius-aware vacuity curriculum (high `r` → higher epistemic pressure) is in-scope.  
- Does **not** unpark or rewrite sealed v7 uncertainty rematches.  
- Discovery Story / agent tools remain read-only over governed artifacts after ingest.

---

## 11. Losses & training protocol (MVP outline)

| Phase | Focus |
|-------|--------|
| P1 Geometric pretrain | Lift + hyp stack + disc health (entropy, saturation, ER); R0–R5 fixed |
| P2 MoE specialization | Gumbel cool-down with `hard=True`; E0–E3 commitment; anti-monopoly |
| P3 Biology / adversary | Holdout migration/basin; dehydron-rim ontology checks; hard negatives |

Multi-task details (margin labels, SDRP matrix CE, etc.) are **sprint-fillable** only if they do not change R0–R5 or the Equiformer→hyp spine. Prefer physics-aligned targets already in-repo where possible.

**Data diet (aspirational):** structure snapshots + MD flexibility + mutational stress — phased; not required for spine compile.

---

## 12. MVP vs implementation sprints

### MVP (must stand as one complete spine)

1. EquiformerV3 → projector → ball  
2. **Full R0–R5 loader** (dual-layer R0 + Layer B exclusivity, wrap ≤1, bidirectional)  
3. Sparse relation-aware HyperbolicGraphAttention (segmented softmax; `γ_R`, `β_R`)  
4. E0–E3 MoE with train-time `hard=True` Gumbel  
5. Poincaré + rim↔core + per-R telemetry  
6. Train loop + MLflow `tokyoeye/equiformer-v3-moe/geometric/full-stack`  

### Frozen sprint order (dependency)

| Order | Block | Why first |
|-------|--------|-----------|
| **Sprint 2** | R0–R5 PDB/CIF graph loader | **LOCKED / GREEN** — `science/tokyo_eye/v8/r0_r5_graph.py` (+ vendored `biophysics.py`) · tests `tests/v8/test_r0_r5_graph.py` |
| **Sprint 1** | Sparse HyperbolicGraphAttention | **LOCKED / GREEN** — `science/tokyo_eye/v8/attention.py` · tests `tests/v8/test_hyperbolic_graph_attention.py` |
| **Sprint 3** | Hard-commitment MoE | **LOCKED / GREEN** — `science/tokyo_eye/v8/moe.py` · tests `tests/v8/test_hard_moe.py` · train `hard=True` Gumbel STE + eval argmax + CV load-balance loss |
| **Sprint 4** | Top-level spine + train driver | **LOCKED / GREEN** — `model.py` (`TokyoEyesHyperbolicV8`) · `engine.py` (`train_v8_step`, curriculum radius, Gumbel schedule, `PoincareDiagnosticsEngine`) · tests `tests/v8/test_spine_integration.py` |
| **Sprint 5** | Equiformer bind + MLflow harness | **LOCKED / GREEN** — weight map `configs/equiformer_v3_weight_map.json` · `equiformer_frontend.py` · `experiments/training/v8/run_v8_experiment.py` · tests `tests/v8/test_sprint5_harness.py` · default ckpt path `checkpoints/v8/pretrained/equiformer_v3_baseline.pt` |

**Lineage isolation:** All v8 code lives under `science/tokyo_eye/v8/` and `tests/v8/` / `experiments/training/v8/` / `checkpoints/v8/`. No imports from `biology_graph`, v7 model, or v66 trainers. Shared platform only (torch/PyG/Bio.PDB/MLflow/container).

**Sprint 2 note:** H-bond candidacy uses atom-validated donor N → acceptor O (v8-vendored wrap SSOT) + seq/spatial gates; DSSP `E ≤ −0.5` energy filter is **baseline** (addendum §1.4 #21 — upgrade, keep). Geometry wrap gate is ≤**1** under the double-cone counter (addendum §2.7 AMEND; classical Fernández 19 was sphere-calibrated).

**Sprint 1 note:** Softmax/scatter index follows the Sprint-1 blueprint (`edge_index[0]` with logits on `(Q_i, K_j)`). Bidirectional R0–R5 emit makes neighborhoods symmetric.

Later fill-ins: uncertainty head, viewer edge-color, optional Normalizer persistence of R0–R5.

---

## 13. Explicit non-goals

- Editing or “fixing” `HEALTHY_V7_CKPT` in place  
- Building on cold/angfill weights as trunk  
- Dense N×N hyp attention / dense `−∞` logit boards  
- Soft MoE forward path in training (`hard=False`)  
- Hard-coded hierarchical spoke edges  
- Piecemeal graph growth (chemistry “later”)  
- Rewriting PDB/CIF download or ingest for the model swap  
- New Postgres SSOT bypassing `data/normalizer`  
- **(AMENDMENT 2026-09-15)** Calling the trunk “pure hyp” while `exp₀(W·log₀(z))` / tangent pool remain live geometry substitutes  
- Waiving `pure_hyp_pass` for Pearson / MoE / rim sat alone  

---

## 14. Success criteria (sign-off bars for later closeouts)

| Gate | Intent |
|------|--------|
| Graph contract | Dual-layer R0 + R1–R5; exclusivity tests green; ±1+Hbond emits both R0 and R1 |
| Geometry | Single lift SSOT; disc health without J-collapse under curriculum |
| Pure hyp (amendment) | `pure_hyp_pass=true` — no post-lift tangent Linear/pool/mix as geometry substitute; Einstein/Klein OK |
| Attention | Sparse segmented softmax only; no dense N×N materialization; Q/K/V not tangent-Linear |
| MoE | Train `hard=True`; no soft mush / 100% single-expert monopoly on Stage-A |
| Ontology | Rim/core correlates with R2 vs R1 (sealed-style dehydron-rim story) |
| Transparency | Rim↔core flow + per-R message fractions logged every run |
| Biology | Report-only then Pass bars for basin/migration on prereg holds |

---

## 15. Implementation watchouts

- Keep **one** `expmap₀` / `project_to_ball` / `poincare_dist` primitive set (no conflicting tanh formulas).  
- Gradient clip near boundary; curriculum τ before full rim permission.  
- Einstein midpoint: Option A isolates must short-circuit empty neighborhoods.  
- Equiformer dependency pin + GPU memory budget for Stage-A proteins.  
- Unit-test: sequence-adjacent H-bond pair → `edge_type` multiset contains both `0` and `1` for that directed pair family.  
- Unit-test: Gumbel path asserts `hard=True` in train mode (STE one-hot forward, soft grads).  
- **(AMENDMENT 2026-09-15)** CI / static gate: forbid `_tangent_linear` / tangent-pool patterns in EQU trunk forward after remediation card lands; until then stamp `pure_hyp_pass=false`.  
- Do not “fix” purity by renaming tangent ops or claiming Einstein while Q/K/V stay Euclidean in log₀.  

---

## 16. Sign-off

**Approver:** signed (user)  
**Date:** 2026-07-22  
**Freeze amendments locked in this revision:** dual-layer R0 coexistence; sparse segmented softmax (no dense −∞ board); train-time Gumbel `hard=True`; model-only / no ingest rewrite.

**Amendment 2026-09-15 (`tokyo_eye_equ_pure_hyp_v1`):** Pure hyperbolic after lift — Einstein/Klein barycenters allowed; post-lift tangent Linear/pool/mix as geometry substitute **forbidden**. Geometry trunk promotion requires `pure_hyp_pass=true`. Spec + gate stamp under `2026-09-15-tokyoeye-equ-pure-hyp-freeze-amendment`.

**Amendment 2026-09-16 (`tokyo_eye_v8_freeze_reconciliation_v1`):** Signed REVERT Option B lift; REVERT `lr_hyperbolic=1e-3`; AMEND MLflow canonical experiment to `tokyoeye/equiformer-v3-moe/geometric/full-stack`. Full dispositions: [`2026-09-16-tokyo-eye-v8-freeze-addendum.md`](2026-09-16-tokyo-eye-v8-freeze-addendum.md).

By this freeze: Path A Equiformer spine, **R0–R5 unalterable graph contract**, E0–E3 post-hyp MoE, first-class diagnostics, v8 isolation from sealed v7, and (as of amendment) **auditable pure-hyp** — not marketing.

**Equiformer checkpoint convention (Sprint 5):**  
`checkpoints/v8/pretrained/equiformer_v3_baseline.pt` + JSON weight map  
`science/tokyo_eye/v8/configs/equiformer_v3_weight_map.json`  
(`D_s=128`, `D_v=3`, `lr_backbone=1e-5`, `lr_hyperbolic=1e-3`). Missing ckpt → stub frontend smoke path.

**Next:** Drop a real EquiformerV3 state dict at the convention path (or `--equiformer-ckpt`), then `make train-v8-experiment`.
