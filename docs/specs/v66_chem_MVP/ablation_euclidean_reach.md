# Ablation Spec: Euclidean Reach vs. Signal Diffusion

**Target path:** `docs/specs/v66_chem_MVP/ablation_euclidean_reach.md`  
**Date locked:** 2026-07-19 (rev: Top-K multi-scale)  
**Status:** **CLOSED Fail** (manifold-sync authoritative) — `chem_mvp_euc_reach_manifold_v1` completed with shared `manifold_ssot` loader; suite grade **Fail** on cone enrichment (Δ≈0). Prior `chem_mvp_euc_reach_v1` remains **INVALIDATED** (construction mismatch autopsy only).

**Parent:** [`../chem-mvp-reengage/README.md`](../chem-mvp-reengage/README.md) (§4 measurement contract)  
**Lineage:** **v66 standalone only** — `experiments.training.v66` + `science.dtie.v66` (no v6/v65 imports)  
**Geometry:** [`../../audit/EUCLIDEAN_CONSTRUCTION_HYPERBOLIC_INFERENCE.md`](../../audit/EUCLIDEAN_CONSTRUCTION_HYPERBOLIC_INFERENCE.md)  
**Not this bet:** SSE containment (diameter FAIL); HA packing (PARKED); hyp MP; Path2; flat Cα≤8Å rule (v1 — empty on dense packing)

---

## 0. Problem frame (locked)

Root bottleneck is **signal diffusion / oversmoothing** with N: fixed 6-hop MP washes out on large graphs. Flat Å cutoffs either saturate large proteins or produce **zero** shortcuts when packing already owns the contact shell (v1 Part 0 on 4DSO).

**Governor:** Top-K spatial selection so shortcut count scales **O(N)**, not O(N²).

---

## 1. Hypothesis

Introducing **Top-K Euclidean shortcut** edges (outside the 6-hop sightline) improves recovery of **literature-backed** allosteric hubs / pathway residues and knockout influence on **medium and large** graphs vs locked 6-layer chem-MVP, **without regressing** the small/rigid tier or rim stability.

**Falsifier:** Aggregate Pass bars (§5) must clear across the multi-scale suite (§4); any hard Fail on a tier (regression on 1BE9, or no positive aggregate on medium/large) = **Fail**. Partial / mixed = **Fail**.

---

## 2. The Lever (Treatment) — Top-K spatial + topological

| Field | Lock |
|-------|------|
| **Action** | Single new edge class `euclidean_shortcut` (relation ID 7 on chem stack) |
| **`num_layers`** | **Locked at 6** |
| **SSE / hierarchy** | **Forbidden** |
| **Mutual exclusion** | Not with `containment_edge_mp` (same ID space) |

### Edge rules (all must hold)

1. **Topological exclusion:** \(D_{\mathrm{graph}}(i,j) > 6\) hops on the **baseline** undirected role+chem multigraph (packing ∪ dehydron ∪ spoke ∪ ribbon ∪ disulf ∪ covale), **before** shortcuts.  
2. **Sequence exclusion:**  
   - **Same chain:** \(|\mathrm{Seq}(i)-\mathrm{Seq}(j)| \ge 10\)  
   - **Different chains** (e.g. IGPS heterodimer): sequence rule **waived** — cross-chain pairs may shortcircuit the domain void  
   - Builder implements this as `if same_chain: apply seq≥10; else waive` — **not** a same-chain-only mask. Do **not** “fix” with `same_chain | interface_mask` (redundant / wrong).  
3. **Top-K spatial governor:** For each residue \(i\), among partners surviving (1)–(2) with \(D_{\mathrm{euc}}(i,j) \le 20\,\text{Å}\) (Cα), keep the **K = 2** nearest Euclidean neighbors.  
4. **Undirected emit:** Store unique unordered pairs; emit bidirectional MP rows (as other chem/role edges).

**Complexity bound:** At most \(O(NK)\) directed selections → undirected shortcut count \(\le NK\) (K=2 → \(\le 2N\)).

**Why not flat ≤8 Å:** Packing already claims most ordered Cα≤8 contacts → hop>6 ∩ ≤8Å is often empty (v1).

### Architecture notes

- Expands `num_relations` + one radial MLP; isolated-seed Part 0.  
- Edge attr: Cα `GEO_DIM=4`; no HA aux.  
- No Normalizer / `fact_graph_edge` writes.  
- Flag: `--euclidean-shortcut-mp` / `TrainingConfig.euclidean_shortcut_mp` (**v66 launcher only**).

---

## 3. Matched Train IDs

| Arm | Run ID | Notes |
|-----|--------|-------|
| **Baseline** | `chem_mvp_stage_a12_cold_v1` | Locked chem-MVP. **Do not overwrite.** |
| **Treatment (prior)** | `chem_mvp_euc_reach_v1` | **INVALIDATED** — Part0 vs grade construction mismatch. Keep for autopsy only. |
| **Treatment (manifold-sync)** | `chem_mvp_euc_reach_manifold_v1` | **Current** cold run. Preflight: `make preflight-v66-chem-mvp-euc-reach-manifold`. Train: `make train-v66-chem-mvp-euc-reach-manifold`. Shared loader: `manifold_ssot.py`. |

**Must match:** seed, epochs, feeler P1, MoE, z-norm **off**, `topology_three_vector`, `role_edge_mp` + `chem_edge_mp`, containment/coupling/HA/barcode/geom-prior **OFF**, no Path2/HA warm-start.

**Corpus note:** Stage A-12 remains the **train** corpus for matched cold. The **grade** suite (§4) is evaluated by forward on 1BE9 / 1GPW / 1F88 (may be OOD to train). Optional later: register a train corpus that includes the suite — separate registration.

---

## 4. Target data / site list: Multi-Scale Validation Suite

Victory on `cone_depth` and knockout Pass bars (§5) requires an **aggregate positive Δ** across this suite. **No individual protein may hard-fail** (flat OK on **1BE9**; **regression** on 1BE9 = Fail).

| Tier | PDB | Protein | Structural challenge | Frozen pathway set \(S\) (auth; literature) |
|------|-----|---------|----------------------|-----------------------------------------------|
| **Small / rigid** | **1BE9** | PSD-95 PDZ3 (~110 res) | Compact; baseline 6-hop should span. Top-K must not degrade local chemistry. | **H372** (binding hub), **I341**, **F340** (distal coupled). Refs: Lockless & Ranganathan, *Science* 1999; *PNAS* 2008 |
| **Medium / void** | **1GPW** | IGPS heterodimer (~453 res A+B) | V-type allostery across ~30 Å HisF↔HisH void. Grade **chains A+B only** (HisF=A, HisH=B; skip C–F). | PRFAR / conduit: **A:50 L50, A:59 R59, A:91 E91**; interface gates: **A:5 R5, A:46 E46, B:99** (GLY in 1GPW — literature often Lys99 on HisH; A:99 is LYS). Conduit: **B:10**. Refs: Loria group, *PNAS* 2012 / PMC 2010. **Pass note:** enrichment under Top-K≤20Å — A:5↔B:99 ~38Å has no direct shortcut; A:5 has eligible B≤20Å but Top-K may prefer same-chain. |
| **Large / 7TM** | **1F88** | Bovine rhodopsin (~348 res) | 7TM; longitudinal + lateral propagation beyond 6 hops. | **Lys296** (retinal), **Pro267** (TM6 hinge), E/DRY **Glu134, Arg135, Tyr136**, NPxxY **Asn302, Tyr306**. Refs: Palczewski et al., *Science* 2000; *PNAS* 2009 |

**Site-list freeze (before train):**

```
checkpoints/v66/diagnostics/euclidean_reach/site_lists/1be9_pathway_residues.json
checkpoints/v66/diagnostics/euclidean_reach/site_lists/1gpw_pathway_residues.json
checkpoints/v66/diagnostics/euclidean_reach/site_lists/1f88_pathway_residues.json
```

Each file: `frozen=true`, chain-qualified auth indices, literature citations, `ssot` pointer.

**KRAS 4DSO / 4OBE:** not Pass structures for this arm (optional secondary report only).

---

## 5. The Lock (Pass bars)

Evaluate **per structure** then aggregate.

| Metric | Per-structure | Suite Pass | Tag |
|--------|---------------|------------|-----|
| **`cone_depth` P@K** | Δ vs chem-MVP on frozen \(S\) | Mean Δ across {1GPW, 1F88} \(\ge +0.05\); **1BE9** Δ \(\ge 0\) (no regression) | `hyperbolic_inference` |
| **Trunk \(\|encoder_h\|_2\) P@K** | Δ | Mean Δ on {1GPW, 1F88} \(\ge 0\); 1BE9 Δ \(\ge 0\) | `euclidean_construction` |
| **Knockout out-effect on \(S\)** | Δ | Mean Δ on {1GPW, 1F88} \(\ge 0\); 1BE9 Δ \(\ge 0\) | causal |

**K (enrichment):** \(K = \max(10, \lceil 0.15\,N\rceil)\) (enrichment top-K; **not** the Top-K=2 edge governor).

**Hard floors:** rim non-regression on train corpus; MoE **hard** loads reported; no SSE fields on shortcut edges.

**Forbidden as Pass:** disc-alone; betweenness headline; shortcut-count alone; soft Partial; Path2 / Jacobian-on-z-norm.

---

## 6. Part 0 — construction checks (before train)

Fail any → do not train.

| # | Check | Pass rule |
|---|--------|-----------|
| 0.1 | Shortcuts obey Top-K + hop + seq/chain rules on **1BE9, 1GPW, 1F88** | Spot-check; no SSE |
| 0.2 | **O(N) bound:** undirected shortcuts \(\le 2N\) per structure | Linear governor |
| 0.3 | **Liveness:** `n_shortcuts > 0` on **1F88** under **grade_ssot** (shared manifold). **1GPW** may be sc=0 (diam≈7, hop>6 empty) — expected, not a Part0 fail. | 1F88 live; 1GPW documented |
| 0.4 | Baseline unchanged when flag off | Bit-identical chem graph prefix |
| 0.5 | Isolated init: only new radial MLP differs | Seed discipline |
| 0.6 | Empty-shortcut forward finite (synthetic) | No NaN |
| 0.7 | Metric tags present | construction / inference / feeler |
| 0.7b | **1GPW hub/chain freeze** | Site list length == unique `(chain_id, auth_seq)`; all in A+B graph; builder emits cross-chain A–B shortcuts; interface neighborhood enrichment (direct hub–hub not required when \(D_{\mathrm{euc}}>20\)) |

Artifacts: `checkpoints/v66/diagnostics/euclidean_reach/part0/` (incl. `1gpw_chain_hub_check.json`)

---

## 7. Training protocol (after Part 0)

```text
# Preflight (required):
# make preflight-v66-chem-mvp-euc-reach-manifold

# Manifold-sync treatment:
# make train-v66-chem-mvp-euc-reach-manifold
# → checkpoints/v66/runs/chem_mvp_euc_reach_manifold_v1/

# Prior INVALIDATED run (do not reuse as authoritative):
# checkpoints/v66/runs/chem_mvp_euc_reach_v1/
```

**Cold train result (locked):** finished ep 20; `v66_best` = global_epoch 20; best_score ≈ 3.674.

---

## 8. Artifacts

| Path | Content |
|------|---------|
| `…/euclidean_reach/part0/` | Construction checks (Part 0 structural **Pass**) |
| `…/euclidean_reach/site_lists/{1be9,1gpw,1f88}_pathway_residues.json` | Frozen \(S\) |
| `…/euclidean_reach/grade_first_ckpt_chem_mvp_euc_reach_v1.json` | First-ckpt (`epoch_000`) suite grade |
| `…/euclidean_reach/grade_best_ckpt_chem_mvp_euc_reach_v1.json` | Best-ckpt (`v66_best`) suite grade |
| `checkpoints/v66/runs/chem_mvp_euc_reach_v1/` | Treatment cold run |
| `…/euclidean_reach/SIGNAL_SIPHON_PROBE.md` | Post-grade autopsy: edge attribution method |
| `…/euclidean_reach/shortcut_signal_siphon_{1gpw,1f88}.json` | Ablation + geo grad×act fractions |
| `…/euclidean_reach/manifold_reconcile_1gpw_rho.json` | Path A vs B per-residue ρ/τ + graph stats |
| `…/euclidean_reach/MANIFOLD_RECONCILE.md` | Honest autopsy (constant ρ vs real ρ; sc 684 vs 0) |
| This file | SSOT |

---

## 9. Outcomes log

| Date | Result | Notes |
|------|--------|-------|
| 2026-07-19 | Pre-registered v1 | Flat ≤8Å + 4DSO — **obsolete** |
| 2026-07-19 | Part 0 v1 | PASS construction but **4DSO n_shortcuts=0** — lever empty |
| 2026-07-19 | **Pre-registered v2** | Top-K=2, \(D_{\mathrm{euc}}\le 20\), hop>6, seq≥10 (cross-chain waived); suite **1BE9 / 1GPW / 1F88** |
| 2026-07-19 | **Part 0 v2 PASS** | Suite live; hub freeze + A+B trim; builder cross-chain **allowed**. 1BE9 N=115 sc=0; **1GPW A+B N=453 sc=684** (48 cross-chain); 1F88 N=338 sc=95. **Note:** 1GPW sc=684 was later shown to be a **false Pass** (constant-ρ Part0 loader). |
| 2026-07-19 | **Cold train complete** | `chem_mvp_euc_reach_v1` vs baseline `chem_mvp_stage_a12_cold_v1`; ep 20; `v66_best` @ global_epoch 20; best_score ≈ 3.674. Makefile `train-v66-chem-mvp-euc-reach`. |
| 2026-07-19 | Suite grade numbers (under invalidated banner) | **First-ckpt** (`epoch_000`): mean Δ cone {1GPW,1F88}=+0.007; mean Δ trunk=−0.002. **Best-ckpt**: all Δ cone/trunk = 0.000 on suite; 1BE9 no regression. Knockout/rim **not run**. **1F88 grade sc=95** (live under Path B). These are factual measurements — **not** an authoritative CLOSED Fail while Path A≠Path B. |
| 2026-07-19 | Signal-siphon probe | Edge-class message-ablation + geo grad×act on `cone_depth[S]` (`SIGNAL_SIPHON_PROBE.md`). **1F88**: euc |Δ|≈0. **1GPW grade**: 0 shortcuts; diam~7. Part0-loader CF uses a different graph. Does **not** authorize a Fail close. |
| 2026-07-19 | Part0 dual-path restore | Part0 retains **both** `part0_legacy` (`_load_multichain_ca`, constant ρ≡12) and `grade_ssot` (`load_suite_prot`). Default suite = grade; `--reconcile` / `--loader both` expose side-by-side. See `MANIFOLD_RECONCILE.md`. |
| 2026-07-19 | **INVALIDATED — Pipeline / construction mismatch** | Retracted: "CLOSED Fail", "1GPW empty-lever artifact ⇒ ignore", and "primary next = governor redesign." Primary next = **manifold/loader reconcile**. Evidence: Path A never computed real ρ (constant 12); sc 684 vs 0 is construction mismatch, not ρ corruption of the train manifold. |
| 2026-07-20 | **Manifold-sync cold complete** | `chem_mvp_euc_reach_manifold_v1` ep20; best_score≈3.674; `euclidean_shortcut_mp=True`. Best-ckpt grade: all Δ cone/trunk=0; mean Δ cone {1GPW,1F88}=0 → **Fail** (authoritative under matched construction). See `TRAIN_COMPLETE.md`. |

---

## 10. After invalidate (pending reconcile)

| If | Then |
|----|------|
| **Path A vs B ρ/graph parity unresolved** | Status stays **INVALIDATED**. Keep dual-path Part0. No soft-Pass. No governor-redesign-as-primary. |
| **Autopsy under Part0-aligned probe (done 2026-07-19)** | Siphon not supported on existing weights; does **not** close as Fail/Pass. See `AUTOPSY_PART0_ALIGN.md`. |
| **Reconcile chooses intentional construction** | Re-register Pass bars + matched grade on that construction; then re-grade (or re-train only if registered). |
| **Historical grade numbers** | Keep under invalidated banner (e.g. 1F88 sc=95; first/best Δ) — not a Fail close |
