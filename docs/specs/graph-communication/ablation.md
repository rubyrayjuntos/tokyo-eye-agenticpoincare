# Heavy-atom–informed communication — matched ablation

**Date locked:** 2026-07-19  
**Status:** **PARKED / STOP** (2026-07-19) — D4 **Partial** (flat) + instrument_b par/worse; do **not** retrain; stay on chem-MVP communication. Re-engage SSOT: [`../chem-mvp-reengage/README.md`](../chem-mvp-reengage/README.md).  
**Parent:** [`design.md`](design.md) (D1–D4 locked)  
**Geometry contract:** [`../../audit/EUCLIDEAN_CONSTRUCTION_HYPERBOLIC_INFERENCE.md`](../../audit/EUCLIDEAN_CONSTRUCTION_HYPERBOLIC_INFERENCE.md)  
**Baseline lineage:** chem-MVP feeler (`role_edge_mp` + `chem_edge_mp`, `topology_three_vector`)

---

## 0. Purpose

Test whether **heavy-atom–informed Euclidean edges** (same residue nodes + same channel *families*, richer channel *definition/scoring*) improve **post-lift** recovery of pathway-relevant residues vs matched chem-MVP — without collapsing dehydron-rim / feeler geometry.

**This is a communication (graph construction) ablation**, not a loss ablation, not hyperbolic MP, not PPI-edge v1, not atom-node GNN.

---

## 1. Arms (matched)

| Arm | ID | Graph recipe |
|-----|-----|----------------|
| **Baseline** | `chem_mvp` | Current: packing / dehydron / spoke / ribbon + disulf/covale; Cα Δxyz+d; ρ/H-bond rules as today |
| **Treatment** | `ha_edges_v1` | **Same relation IDs and node census**; packing + dehydron (+ optional spoke) **existence/strength** use heavy-atom geometry / pair chemistry per §3 |

**Must match across arms**

- Corpus, seed, epochs, feeler loss stack, MoE, z-norm flags (both **off** for primary cold pair — same as `chem_mvp_stage_a12_cold_v1` recipe unless both arms explicitly z-norm)
- `GNN_INPUT_MODE=topology_three_vector` → `data.x = [ρ, τ, ss]`
- `num_layers=6`, `hidden=128`, `chem_edge_mp=True`, containment/coupling **OFF**
- No new losses, no geom angular prior, no node barcode, no hyp MP

**Only differ:** edge builder / `edge_attr` aux used by packing & dehydron (and documented spoke rule if changed).

Preferred baseline checkpoint (control, do not overwrite):  
`checkpoints/v66/runs/chem_mvp_stage_a12_cold_v1/v66_best.pt`

Treatment run dir (suggested):  
`checkpoints/v66/runs/ha_edges_v1_stage_a12_cold_v1/`

---

## 2. Part 0 — construction checks (before any train)

Fail any → do not train.

| # | Check | Pass rule |
|---|--------|-----------|
| 0.1 | Residue node count unchanged vs baseline on 4OBE / 1LYZ | Same N |
| 0.2 | Relation vocabulary unchanged | Still packing/dehydron/spoke/ribbon/disulf/covale IDs; no new mandatory relations |
| 0.3 | Structures with **zero** disulfides forward-identical chem path | No NaN; empty chem masks skipped |
| 0.4 | Isolated init: new aux dims / MLPs do not scramble gate/trunk seed vs baseline except intended new weights | Documented seed discipline |
| 0.5 | Edge liveness: packing + dehydron edge counts on 4OBE within sane band vs baseline (±50% unless justified) | No silent empty dehydron set |
| 0.6 | Tag audit: every primary metric labeled `euclidean_construction` or `hyperbolic_inference` or `disc_view` | No unlabeled “hub” claim |

Artifact dir: `checkpoints/v66/diagnostics/graph_communication/ha_edges_v1/part0/`

**Part 0 result (2026-07-19):** **PASS** on 4OBE + 1LYZ. Summary: `…/part0/summary.json`. Training was **not** run.

| Check | 4OBE | 1LYZ | Notes |
|-------|------|------|-------|
| 0.1 | PASS (N=169) | PASS (N=129) | |
| 0.2 | PASS | PASS | attr dim 19; no new relation IDs |
| 0.3 | PASS | PASS | empty chem path; forward finite |
| 0.4 | PASS (global) | — | HA = multiplicative scale only; no new MLPs |
| 0.5 | PASS (pack 255→286, dehy 173) | PASS (pack 144→152, dehy 133) | within ±50%; spoke/ribbon unchanged |
| 0.6 | PASS | — | all tagged `euclidean_construction` |

---

## 3. Treatment definition (`ha_edges_v1`) — construction only

### 3.1 Nodes (unchanged)

Residue-only; `data.x = [ρ, τ, ss]` as chem-MVP.

### 3.2 Channels (same families)

| Relation | Change vs baseline |
|----------|-------------------|
| **packing** | Contact / packing decision may use heavy-atom min-distance or Cβ/centroid proxies in addition to Cα 8 Å (exact rule fixed in implementation plan; must be pre-registered in Part 0 writeup) |
| **dehydron** | Keep underwrap H-bond semantics; allow donor/acceptor heavy-atom geometry already used for ρ to **score edge strength** (aux), not only binary membership |
| **spoke / ribbon** | Membership rules **unchanged** in v1 unless Part 0 shows packing change forces a documented spoke tweak |
| **disulf / covale** | Unchanged (already heavy-atom covalent) |

### 3.2.1 Frozen packing existence (Part 0)

Ordered↔ordered (`ρ ≥ τ`) **and**:

- `0.1 < Cα–Cα ≤ 8.0 Å` (baseline band), **or**
- `0.1 < heavy_atom_min_dist ≤ 4.5 Å` **and** `Cα–Cα ≤ 12.0 Å` (HA rescue)

Heavy-atom min-distance = min pairwise distance among non-H atoms of the two residues. Missing atoms → Cα-only fallback (identical to baseline existence).

Spoke membership remains Cα ≤ 8 Å with ordered≠disordered (unchanged).

### 3.2.2 Frozen dehydron strength (Part 0)

Membership unchanged: `rho_bond < τ` via `compute_bond_wrapping_count` (donor N → acceptor O).

Strength aux (not a new relation):

```
deficit = clip((τ − rho_bond) / τ, 0, 1)
prox    = clip((5.5 − d_NO) / (5.5 − 2.5), 0, 1)
strength = 0.5 * deficit + 0.5 * prox
```

Packing strength aux: `clip(1 − d_HA/8, 0, 1)` (falls back to Cα–Cα).

### 3.3 Edge attributes

Keep `GEO_DIM=4` Cα Δxyz+d for SE(3) (MP geometry stays Cα-based in v1 for matched SE(3) behavior).

**Add** (treatment only), in aux slots after one-hots — exact column map frozen in Part 0:

| Aux idea | Purpose |
|----------|---------|
| Heavy-atom min-distance (or C–C / N–O / C–N pair min among contact atoms) | Chem-specific proximity |
| Packing / dehydron strength scalar | Softer than binary channel |

#### Frozen aux column map (chem + HA) — Part 0

```
[0:4]   GEO Cα Δxyz+d
[4:11]  one-hots: packing, dehydron, spoke, ribbon, coupling, disulf, covale
[11]    spoke_rho / coupling_strength
[12:17] dehydron barcode (EDGE_BARCODE_DIM=5)
[17]    HA_MIN_DIST_NORM
[18]    HA_STRENGTH
→ EDGE_ATTR_CHEM_HA_DIM = 19
```

Flag: `--ha-edge-mp` / `TrainingConfig.ha_edge_mp` / `GOSPConeMapperV66.ha_edge_mp`.
Conv consumes `HA_STRENGTH` as `(1 + strength)` on packing + dehydron relation paths (no new MLPs).

**Note:** `EDGE_ATTR_ROLE_HA_DIM` (17) collides numerically with `EDGE_ATTR_CHEM_DIM` (17); chem pad uses `data.ha_edge_graph` to disambiguate.

### 3.4 Non-regression δ (filed Part 0)

Feeler rim enrichment (§5.1): hold on ≥ **baseline_hold_count − 1** Stage A-12 structures (`FEELER_RIM_DELTA_STRUCTURES = 1`). Not measurable without training — applied at grade time only.

**Forbidden in v1:** new relation IDs for “C-C edge type” as separate MP paths; atom nodes; thermal/PPI relations; changing `node_dim`.

---

## 4. Evaluation sets

### 4.1 Non-regression corpus

**Stage A-12** (`v6_corpus_stage_a_small_v1.json` / cache `graphs_38a6993d7a439aa4.pt` or successor with same stamp).

### 4.2 Pathway holdout (primary D4)

| Structure | Role | Registered pathway-relevant residues (auth seq, chain A unless noted) |
|-----------|------|------------------------------------------------------------------------|
| **4OBE** | KRAS WT (Stage A) | **Switch I** 30–40; **Switch II** 60–76; **P-loop/G12** 10–17; probe anchors **12, 151, 163** (legacy KRAS probes) |
| **4DSO** | KRAS G12D OOD | Same index map as existing hub-migration probes (`12, 151, 163`) — used for **transfer** read only, not training |

Optional secondary (report only, not Pass gate in v1): RAF1 PPI corpus structures from `manifests/v6_corpus_raf1_ppi_v1.json` — **disc audits forbidden as Pass**; any PPI site list must be filed before use.

**Site-list freeze:** `checkpoints/v66/diagnostics/graph_communication/ha_edges_v1/site_lists/4obe_pathway_residues.json` (`frozen=true` as of 2026-07-19 Part 0 PASS); do not edit after freeze.

---

## 5. Metrics

### 5.1 Non-regression (`euclidean_construction` + feeler health) — must hold

| Metric | Floor | Tag |
|--------|-------|-----|
| Physics rim / cone–ρ coupling (existing feeler instruments) | No worse than baseline by pre-set δ (file δ with Part 0; default: rim enrichment holds ≥ baseline−1 structure on Stage A-12) | feeler |
| MoE: starvation / route_H / eligibility | No new collapse attributable to arm | feeler |
| Trunk ER / disc occupancy watch | Report; soft watch only | construction |

### 5.2 Primary D4 — `hyperbolic_inference` (Pass gate)

**Metric:** Enrichment of registered pathway residues among top-K sites by **post-lift salience**.

**Salience score (locked for v1):**

\[
s_i = \mathrm{cone\_depth}_i
\]

(from ball `dist0` / cone pipeline already produced by the model — **not** disc radius, **not** Euclidean betweenness).

**Procedure (per structure, per arm):**

1. Forward model; collect `cone_depth` (or equivalent depth) per residue.  
2. Rank residues by \(s_i\) descending.  
3. Let \(S\) = registered pathway set (§4.2).  
4. Compute **precision@K** and **recall@K** with \(K = \max(10, \lceil 0.15\,N\rceil)\).  
5. Also report mean \(s_i\) on \(S\) vs mean \(s_i\) on complement (gap).

**Primary scalar for arm compare (4OBE):**  
\(\Delta = \mathrm{precision@K}(\texttt{ha\_edges}) - \mathrm{precision@K}(\texttt{chem\_mvp})\)  
(same K, same site list).

| Grade | Rule |
|-------|------|
| **Pass** | 4OBE \(\Delta \geq +0.05\) **and** recall@K not down by more than 0.05 **and** §5.1 non-regression holds |
| **Fail** | 4OBE \(\Delta \leq -0.05\) **or** §5.1 broken |
| **Partial** | otherwise (including flat \(\|\Delta\| < 0.05\)) |

**4DSO:** report same metrics (OOD transfer). **Does not** decide Pass alone.

### 5.3 Secondary (report only — not Pass)

| Metric | Tag | Note |
|--------|-----|------|
| Trunk `encoder_h` enrichment with same site list | `euclidean_construction` | Communication changed representation? |
| Classical betweenness enrichment | `euclidean_diagnostic` | Scaffolding only |
| Disc r / θ enrichment | `disc_view` | Forbidden as Pass |
| Jacobian flow-influence | — | **Forbidden** on z-norm-on; primary arms are z-norm-off — still not the Pass metric |
| Causal knockout of top cone_depth sites | causal | Optional after Pass/Partial; not required for v1 close |

### 5.4 Explicitly forbidden as Pass evidence

- Disc-alone precision/recall  
- Euclidean betweenness headline  
- Soft hold-count OR tricks  
- Agent Path 2 / directionality surrogate  
- “Viewer looks better”

---

## 6. Training protocol

```text
# PARKED — do not run:
# make train-v66-ha-edges-v1   # errors on purpose (2026-07-19)

# Baseline (active substrate — do not overwrite):
# checkpoints/v66/runs/chem_mvp_stage_a12_cold_v1/v66_best.pt

# Treatment checkpoint (compare-only):
# checkpoints/v66/runs/ha_edges_v1_stage_a12_cold_v1/v66_best.pt
```

Historical recipe (already completed once): Stage A-12 cold, `--v66-feeler-lineage --chem-edge-mp --ha-edge-mp`, seed=1, 20 ep, no warm-start. **Do not retrain.**

---

## 7. Artifacts

| Path | Content |
|------|---------|
| `.../ha_edges_v1/part0/` | Construction checks |
| `.../ha_edges_v1/site_lists/4obe_pathway_residues.json` | Frozen S |
| `.../ha_edges_v1/grade_4obe.json` | precision@K, recall@K, Δ, non-regression |
| `.../ha_edges_v1/grade_4dso.json` | OOD report |
| `docs/specs/graph-communication/ablation.md` | This file (SSOT) |

---

## 8. Outcomes log

| Date | Result | Notes |
|------|--------|-------|
| 2026-07-19 | **Pre-registered** | Awaiting Part 0 + implementation |
| 2026-07-19 | **Part 0 PASS** | 4OBE/1LYZ construction checks; site list frozen; `make train-v66-ha-edges-v1` stubbed; **training not run** |
| 2026-07-19 | **Partial** | Matched Stage A-12 cold `ha_edges_v1_stage_a12_cold_v1` (cuda, 20 ep, seed=1, chem+ha edge MP, z-norm off, no warm-start). Primary D4 4OBE cone_depth: P@K=0.3077 (K=26) vs chem_mvp 0.3077 → **ΔP=0.00**; R@K=0.2105 both → **ΔR=0.00**. §5.1 non-regression **holds** (rim 12/12 ≥ baseline−1; route_H≈1.378). 4DSO OOD report-only: ΔP=0.00, ΔR=0.00 (K=27). Artifacts: `…/ha_edges_v1/grade_4obe.json`, `grade_4dso.json`. No disc/betweenness/Jacobian Pass soft rules. |
| 2026-07-19 | **instrument_b (secondary)** | No new train. Both arms measured under `…/ha_edges_v1/instrument_b/`. Trunk `euclidean_construction` salience = ‖encoder_h‖₂ on frozen S: **P@K=0 / R@K=0 both** (S anti-enriched; gap≈−15); Δ flat/worse. Knockout mean out_effect on S: chem 0.0645 → HA 0.0582 (HA slightly down); gap S−comp remains negative. MoE hard loads: HA less 0/3-dominated (4OBE expert-2 share 0.024→0.154); route_H≈1.381 both. Packing mean HA_STRENGTH≈0.56 (HA only). **No biology / communication Pass** on these instruments — does not rewrite D4 Partial. See `instrument_b/summary.md`. |
| 2026-07-19 | **PARKED / STOP** | User decision: par or worse on important measures → **do not continue** HA packing aux. Stay on `chem_mvp_stage_a12_cold_v1`. Checkpoint kept for compare only. `make train-v66-ha-edges-v1` **errors on purpose**. Do not add thermal/PPI from this arm. |

---

## 9. After close

| If | Then |
|----|------|
| **Pass** | Keep `ha_edges_v1` as communication default candidate; schedule causal knockout secondary; consider PPI channel registration |
| **Partial** | ~~Inspect aux~~ → **closed as PARKED/STOP** (2026-07-19); revert to chem-MVP; no thermal/PPI from this lineage |
| **Fail** | Revert to chem-MVP communication; revise heavy-atom edge rule — do not “fix” with disc metrics |
| **PARKED** | Active path: [`../chem-mvp-reengage/README.md`](../chem-mvp-reengage/README.md) |
