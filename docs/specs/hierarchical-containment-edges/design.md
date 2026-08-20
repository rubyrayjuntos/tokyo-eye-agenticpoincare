# Hierarchical Containment Edges — Design (v2: Flow-Motivated Promotion)

**Status:** **Promoted to next buildable** — Chem-MVP closed (`partial`, filed 2026-07-17). Hierarchy materialization locked as **Path B** (train-side HELIX/SHEET parse stopgap; §9). Supersedes design.md v1 stub.
**Created / promoted:** 2026-07-17
**New motivation:** Independent evidence from [`../learned-flow-influence/ablation.md`](../learned-flow-influence/ablation.md) Stage A-12 — directional asymmetry fails specifically on large-diameter structures (diam ≥14 vs 6-layer MP hop budget), while hub-tracking (symmetric component) holds corpus-wide. Containment's parent-node shortcut was originally motivated by Tokyo Eye navigation; it is now independently the mechanistically correct fix for a receptive-field limitation this repo has directly measured.

**Related:**

- [`depth-collision.md`](depth-collision.md) — hard lock, **unchanged**
- [`../struct-conn-typed-edges/design.md`](../struct-conn-typed-edges/design.md) — multi-rel precedent
- [`../struct-conn-typed-edges/ablation.md`](../struct-conn-typed-edges/ablation.md) — Chem-MVP closed `partial`
- [`../learned-flow-influence/ablation.md`](../learned-flow-influence/ablation.md) — diagnostic + new acceptance criterion
- [`ablation.md`](ablation.md) — gates / win-partial-fail for this experiment

---

## 1. What changed from v1

| v1 (deferred stub) | v2 (this doc) |
|---|---|
| Motivation: Tokyo Eye navigation spine only | + measured receptive-field limitation on large structures |
| Vertical edges: unspecified direction | **Explicit up/down relation typing** (§3) |
| Validation: radial stratification, generic "dilution test" | **Diameter-stratified asymmetry floor**, reusing `jacobian_flow_influence.py` directly (§5) |
| Sequencing: after Chem-MVP | Chem-MVP closed → **this is next**; explicit directionality-reward (Path 2) follows, only on the sub-corpus where hop budget already reaches |
| Depth-collision lock | **Unchanged, still binding** (§2) |

Nothing in the hard lock moves. This promotion is scope and validation, not a relaxation of the paper decision already made.

---

## 2. Hard lock (restated, unchanged)

Per [`depth-collision.md`](depth-collision.md), locked 2026-07-17:

- Residue `cone_depth` / disc radius / `tau_dehydron_rim` remain **physics-only**.
- Parent nodes (domain / chain / assembly) may carry their own centripetal / tier-margin targets.
- No dual-loading residue radius. No "small auxiliary weight" compromise.

This doc does not reopen that decision. Containment's job here is **reach**, not **redefining what residue radius means**.

---

## 3. New: directional relation typing (up/down, not bidirectional)

**Problem this section exists to prevent:** a single symmetric parent↔child relation type extends *reach* (more hops effectively available via the shortcut) but does not, on its own, create *directionality*. The flow-influence probe measures `I(A→B) ≠ I(B→A)`; a bidirectional edge with one shared weight matrix has no mechanism to produce that asymmetry — it would only be expected to help the *symmetric* hub-tracking component, which already holds on 11/12 structures. Getting this wrong would mean training the feature, seeing reach improve, and still failing the acceptance test for the reason you actually care about.

**Fix, following the existing multi-rel precedent (`EquivariantConvMultiRel`, Option A — per-relation radial MLPs):**

- `contain_down` (parent → child): one relation ID, own radial MLP.
- `contain_up` (child → parent): separate relation ID, separate radial MLP.

Same pattern as role edges (`packing` / `dehydron` / `spoke` / `ribbon`) and chem edges (`disulf` / `covale`) — **not** a new mechanism, an extension of the same one. `num_relations` grows by **2**, not 1.

Suggested IDs (finalize at implement time against the live role+chem vocabulary):

| Relation | Direction | Notes |
|----------|-----------|-------|
| `contain_down` | parent → child | New ID after chem slots |
| `contain_up` | child → parent | Separate radial MLP |

---

## 4. Graph construction

### 4.1 Node types (heterogeneous)

Under **Path B** (§9.2), v1 materializes only what the local PDB parse provides:

- Level 2 (required): deposited HELIX/SHEET parents (`struct_conf` proxy) — one parent node per annotated range
- Level 3 (leaves): residues (existing nodes, unchanged)
- Level 0–1 (`core_assembly` / `core_polymer_entity`): optional if REMARK350/DBREF parse is cheap; **not** required for the diameter-asymmetry gate
- Domain-level (`rcsb_polymer_instance_feature` / CATH / CDD): **out of Path B v1** — Path A territory

Full four-level RCSB names remain the long-term Path A shape; Path B does not pretend they exist in governed tables.

### 4.2 Parent-node feature initialization

Still an open choice per v1 — **resolve before code, not during**:

- **Option A:** mean-pool of children's current features (ρ, τ, ss, barcode scalars, chem-edge summary)
- **Option B:** learned embedding per parent, initialized independent of children

**Locked for v1 of this experiment: Option A.** Mean-pooling is parameter-free and avoids adding a fourth thing this ablation has to separately verify (a learned-embedding init would need its own isolated-seed check, same as every new parameter tensor in this project has needed). Revisit B only if A's acceptance test fails and pooled features are suspected as the bottleneck.

### 4.3 Edge construction

- `contain_down` / `contain_up`: one pair per (parent, child) from Path B deposited SSE ranges (residue ∈ HELIX/SHEET → that parent). Optional assembly/entity edges only if Level 0–1 parents are built.
- Existing horizontal edges (proximity, role, chem) unchanged, untouched by this addition.
- No edges between parent nodes and residues outside their own subtree (no cross-domain parent shortcuts in v1 — that is a plausible v2 extension, not in scope here).

### 4.4 Attach path (train-side — Path B, locked)

Follow barcode / Chem-MVP *graph-attach* precedent, **not** Chem-MVP's `fact_*` bridge: parse deposited HELIX/SHEET ranges from local Stage A-12 PDBs, build Level-2 parent nodes + `contain_up`/`contain_down` entirely in the training pipeline, extend `EquivariantConvMultiRel` radial MLP count, `--containment-edge-mp` (name TBD at implement). **No Normalizer / governed-table writes in this experiment** — Path A (ingest extension) is a separate project that supersedes this stopgap (§9).

---

## 5. Validation — reusing the flow-influence probe directly

**Do not build a new metric.** The tool that found the problem is the tool that grades the fix.

Instrument: `experiments/diagnostics/jacobian_flow_influence.py`  
Baseline split: `checkpoints/v66/diagnostics/learned_flow_influence/stage_a12.json`  
Diameter reference: `.../asymmetry_vs_diameter.json`

### 5.1 Physics non-regression (unchanged gate family)

- Corpus-12 rim enrichment ≥11/12
- `probe_r_depth_tau` stable vs containment-off baseline
- MoE health (route_H, eligibility) not attributable-regressing

### 5.2 Oversmoothing-at-root re-check (v1's own named risk, now testable)

Reuse the T1c MP-relative-loss-null methodology, applied specifically to parent-node embeddings: compare rank/variance loss at `core_assembly` / `core_polymer_entity` nodes against a synthetic control with the same fan-in. This is the concrete version of v1's "will the root suffer massive oversmoothing" question — answer it with the same tool that already answered it once for residue-level MP.

### 5.3 Isolated-seed check (standard, before first cold run)

`contain_down` / `contain_up` add 2 new radial-MLP parameter sets → confirm prototype bank identity holds containment-off vs containment-on, same `init_seed` discipline as chem edges (`max |Δ| = 0` expected).

### 5.4 Primary acceptance criterion — diameter-stratified asymmetry

Pre-registered against the exact split already measured in `learned_flow_influence/stage_a12.json`:

| Group | Structures | Baseline asymmetry (measured) | Requirement |
|---|---|---|---|
| **Fail group** (diam ≥14) | 1BG1, 2Z6H, 1IVO, 2SHP, 1F88 | 0.023–0.032 | Must rise, individually, above the 0.05 floor on **≥3 of 5** |
| **Pass group** (diam ≤9) | 1UBQ, 1TEN, 1HHP, 1LYZ, 1MBN, 4OBE, 1TIM | 0.058–0.148 | Must **not regress** below their own baseline − tolerance (10% relative) |
| Hub-tracking (all structures) | all 12 | ρ ≈ 0.44–0.69 | Must hold or improve; no structure allowed to drop below its own baseline ρ |

**Win:** fail-group clears on ≥3/5, pass-group holds, hub-tracking holds corpus-wide.

**Partial:** fail-group improves (any rise) but doesn't clear 0.05 on ≥3/5, or pass-group shows mild regression within tolerance.

**Fail:** no movement on fail-group, or pass-group regression exceeds tolerance, or hub-tracking degrades anywhere.

No pooling. Same per-structure, no-generous-reading discipline as Chem-MVP. See [`ablation.md`](ablation.md).

---

## 6. Sequencing (updated)

1. **This experiment via Path B** (train-side HELIX/SHEET parents + containment shortcuts; diameter-stratified acceptance) — §9.2
2. **Path 2** (explicit directionality reward) — scoped *only* to structures where hop budget already reaches (diam ≤9) — run after this closes, regardless of outcome, since it answers a different question
3. Chem-Full (`hydrog` / `saltbr` / `metalc`) — unblocked, independent, can proceed in parallel if resourcing allows; not gated on this
4. **Path A ingest extension** (deposited SSE + real hierarchy tables; supersedes Path B; may absorb §9.1 fidelity fix) — separate project, not a silent side effect of (1)
5. Barcode typed shared-bar edges — still optional, unchanged priority

---

## 7. Known unknowns carried from v1, unresolved by this promotion

- Parent-node feature init (§4.2) — **defaulting to mean-pool for v1**, flagged for revisit.
- Cross-domain parent shortcuts — explicitly out of scope here; may be a natural v2 if this closes as a win.
- Whether `contain_up` needs a different radial-MLP capacity than `contain_down` (asymmetric information content flowing each direction) — not resolved; default to matching capacity for v1; revisit only if the acceptance test result specifically implicates one direction over the other.

---

## 8. Status

| Item | Status |
|------|--------|
| Depth-collision hard lock | **Binding** — unchanged |
| Chem-MVP prerequisite | **Satisfied** (`partial`, not promoted) |
| Flow-influence motivation | **Adopted** — diameter / hop-budget diagnosis |
| Up/down relation typing | **Locked** for v1 |
| Parent feature init | **Locked Option A** (mean-pool) for v1 |
| Acceptance criteria | **Pre-registered** in [`ablation.md`](ablation.md) |
| Hierarchy data coverage audit | **2026-07-17** — see §9 |
| SSE fidelity gap (deposited vs biotite) | **Flagged independently** — §9.1; not a containment blocker |
| Hierarchy materialization | **Path B locked** (train-side stopgap) — §9.2 |
| Implementation | **Unblocked for Path B** — plan: [`docs/superpowers/plans/2026-07-17-hierarchical-containment-edges.md`](../../superpowers/plans/2026-07-17-hierarchical-containment-edges.md) |

---

## 9. Hierarchy prerequisites (audit 2026-07-17)

Live DB was unreachable (docker down). Audit used: schema/migrations + ingest code + local Stage A-12 PDBs (HELIX/SHEET / REMARK 350 / DBREF) + training corpus cache.

### 9.0 Governed schema vs design names

| Design name | Tokyo Eye reality |
|-------------|-------------------|
| `core_assembly` | **No table** — `dim_structure.assembly_id` TEXT only |
| `core_polymer_entity` | **No table** — `dim_chain` (+ `entity_type`); `entity_id` in ingest parse only |
| `struct_conf` | **No table** — `dim_residue.sse_code` is per-residue H/E/C (often biotite P-SEA at feature time, **not** deposited HELIX/SHEET ranges) |
| `rcsb_polymer_instance_feature` | **No table** — closest is `fact_cdd_annotation` (optional Tier-2 CDD) |
| Residue→domain mapping | Only via `fact_cdd_annotation` start/end **if** CDD job ran; `dim_residue` has no `domain_id` |

### Source coverage (local PDB — all 12 Stage A)

| Coverage | Count | Structures |
|----------|------:|------------|
| Any SSE (HELIX/SHEET ≈ `struct_conf`) | **12/12** | all |
| SSE with residue-range mapping | **12/12** | all |
| SSE + REMARK350 + DBREF (assembly/entity proxy) | **12/12** | all |
| Zero SSE in deposited PDB | **0/12** | — |

Zeros are **not** the story — deposited SSE is intentional and rich. Artifact:
`checkpoints/v66/diagnostics/learned_flow_influence/hierarchy_coverage_audit.json`.

- Deposited **SSE** (HELIX/SHEET) is available for all 12 and maps to residue numbers.
- **Domain-level** CATH/SCOP / `rcsb_polymer_instance_feature` is **not** in governed schema (CIF fetch failed here). Closest store: optional `fact_cdd_annotation`.
- Training cache `domain_labels` = hardcoded KRAS-style bins on 4OBE only — **not** usable as Level-2 containment parents.

### 9.1 Separate finding: deposited SSE silently replaced by biotite

**Independent of containment.** Ingest sets `sse_code=None` at parse (`parser.py`); later feature code fills `sse_code` via biotite (P-SEA or equivalent). Deposited HELIX/SHEET **ranges are discarded**, then **re-derived by a different algorithm**.

That is a **data-fidelity gap**, not merely “missing infrastructure for parent nodes”:

- SSE-assignment methods routinely disagree at helix/sheet boundaries and on borderline residues.
- If `ss` / `ss_type` already feeds GNN node features (topology three-vector) and other pipeline consumers, the substitution may be intentional — or an accidental default nobody decided.
- **What this rules out:** treating `dim_residue.sse_code` as deposited author annotation, or using it as a drop-in proxy for HELIX/SHEET range parents.
- **What this does *not* rule out:** Path B reading deposited ranges from PDB files; Path A later materializing those ranges properly; continuing to use biotite `ss` as a *feature* if the discrepancy is small and accepted.

**Path B accepted inconsistency:** parent nodes are built from *deposited* HELIX/SHEET ranges; the GNN's residue `ss` / `ss_type` feature remains biotite's re-derived assignment. Hierarchy structure and the residue-level feature it relates to need not agree on where one helix ends and the next begins — known, accepted stopgap artifact, **not** something the diameter-asymmetry acceptance test must explain if results look odd specifically at SSE boundaries.

**Cheap side-check (when docker is up):** pick one Stage A structure, diff deposited HELIX/SHEET residue sets vs current biotite assignment on governed / feature-time `sse_code`. File disagreement size (boundary vs core residues). Does not block Path B.

### 9.2 Path A vs Path B (locked before any parse code)

Containment is **Chem-Full-shaped**, not Chem-MVP-shaped: there is no `fact_*` hierarchy bridge to attach. Two honest paths:

| Path | What it does | Cost / risk |
|------|----------------|-------------|
| **A — full ingest extension** | Parse deposited SSE ranges into real hierarchy (assembly / entity / SSE parents); decide whether `fact_cdd_annotation` / domain parents are in v1 or SSE-level alone; Normalizer writes. Training and app converge. | Larger project; “do it right once”; answers the ingest-unification instinct as a *product* decision, not as a side effect of this ablation |
| **B — train-side-only parse** | Read HELIX/SHEET from local Stage A-12 PDBs already on disk; construct parent nodes + `contain_up`/`contain_down` only in the training pipeline; no governed writes. Mirrors early barcode work. | Faster to grade diameter-stratified asymmetry; **same train-vs-app divergence risk** as barcode-before-ingest — acceptable only as an explicit stopgap |

**Decision (2026-07-17): Path B for this experiment.** Motivation is the flow-influence receptive-field result, not shipping hierarchy to the app. Path B is scoped as a **stopgap**, superseded when Path A lands — same discipline as barcode → eventual ingest unification. Choosing B here does **not** silently decide that unification question.

**v1 parent scope under Path B:** Level-2 SSE parents from deposited HELIX/SHEET ranges + residue leaves. Assembly / polymer-entity parents optional if cheap from REMARK350/DBREF; **not** required for the diameter-asymmetry gate. Domain-level CATH/CDD deferred (Path A territory).

### 9.3 Build readiness

| Ready? | Item |
|--------|------|
| Yes | Source HELIX/SHEET on all 12 local PDBs |
| Yes | Path B choice locked; attach plan §4.4 |
| No (not needed for Path B) | Governed hierarchy tables / `fact_*` bridge |
| Deferred | Path A ingest extension; live `dim_*` row counts when docker is up; §9.1 biotite-vs-deposited side-check |
