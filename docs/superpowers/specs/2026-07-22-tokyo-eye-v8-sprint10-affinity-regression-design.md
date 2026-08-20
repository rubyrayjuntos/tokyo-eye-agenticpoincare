# TokyoEye-v8 Sprint 10 — Protein-Only Affinity Regression (PDB-Bind)

**Date:** 2026-07-22  
**Status:** APPROVED 2026-07-22 — implement per plan  
**Depends on:** Sprint 8 biophys harden + Sprint 9 MoE rebalance (Mode C balanced guilds)  
**Scope:** **Option C** — protein-only \(-\log K\) head from post-MoE hyperbolic embeddings.  
**Deferred to Sprint 10.1:** ligand encoder / protein–ligand interface graph.  
**Frozen:** R0–R5 graph, DSSP/cone wrap, `v8_biophys_s8` cache, MoE E0–E3 architecture and Sprint 9 load floors.

---

## Problem

TokyoEye-v8 now has chemically gated graphs and balanced expert routing, but no governed path from whole-structure hyperbolic state to thermodynamic affinity. External PDB-Bind labels are required to test whether pocket-localized hyp geometry predicts \(-\log K_d\) / \(-\log K_i\) without memorizing sequence neighbors.

---

## Non-goals

- Ligand atom graphs / SE(3) ligand encoder (Sprint **10.1**)  
- Changing MoE guilds, CV/quota, or Gumbel schedule  
- Touching Normalizer / ingest / `science/dtie` Vina phase6b as the training label path  
- Claiming docking ΔG equivalence (labels are PDB-Bind experimental \(-\log K\) only)

---

## Architecture

```
PDB protein chain
  → frozen v8 loader / R0–R5 / live SE(3)-lite → spine
  → z_hyp (post-MoE) [N, d]
  → PocketGate: w_i = σ( Linear([h_inv_i ; r_i ; mech_i ; dehyd_i]) )
  → t_pool = Σ_i w_i · log₀ᶜ(z_i) / Σ_i w_i
  → z_graph = exp₀ᶜ(t_pool)
  → AffinityFFN( log₀ᶜ(z_graph) ) → ŷ ∈ ℝ   (−log K)
```

Reuse existing `log_map_zero` / `exp_map_zero` in `science/tokyo_eye/v8/attention.py`.

### Pocket gate inputs (per node)

| Feature | Source |
|---------|--------|
| `h_inv` | small MLP on `z_hyp` (or spine Euclidean skip if exposed) |
| `r_i` | `‖z_i‖` (rim proximity) |
| `mech_i` | `σ(mechanism_score)` |
| `dehyd_i` | graph dehydron incidence label (train) / predicted mech proxy (eval optional) |

Weights: `w = softmax(score)` over nodes (or sparsemax). Softmax preferred for Sprint 10 simplicity.

### Loss

\[
L = \mathrm{MSE}(\hat{y}, y) + \lambda_{\mathrm{aux}} L_{\mathrm{aux}}
\]

- Primary: MSE on \(y = -\log_{10} K\) (PDB-Bind standard; document unit once in loader).  
- Optional aux: keep dehydron BCE at **low** coeff (≤0.1) so backbone does not forget Sprint 8 signal when fine-tuning — **default on** for continue-from-s9; off for affinity-head-only probe.

**Train modes**

| Mode | Backbone | MoE | Affinity head |
|------|----------|-----|---------------|
| `head_only` | freeze | freeze | train |
| `finetune_hyp` | freeze SE(3) bank | train MoE+spine+head | train |
| `finetune_all` | train | train | train |

**Default Sprint 10:** `head_only` first (isolate readout), then one `finetune_hyp` rematch if Core Pearson plateaus.

---

## Data & leakage contract (three-tier)

### Sources

- **Train/val pool:** PDB-Bind **Refined** set (experimental affinities).  
- **External test:** **CASF-2016 / Core** — frozen; never in train or early-stopping selection.

### Cluster holdout (≤30% seq identity)

1. Map each complex → receptor PDB chain sequence (CA-complete chain used by v8 loader).  
2. Cluster Refined sequences at **30%** identity.  
3. Assign clusters intersecting Core receptor sequences → **purge entire cluster from train/val**.  
4. Remaining Refined clusters → split **80/20 train/val by cluster** (not by complex), seed=0.

### Lightweight identity tool (vendored under v8)

**Yes — vendor under `science/tokyo_eye/v8/seq_cluster.py` (and CLI `experiments/training/v8/build_pdbbind_splits.py`).**

| Priority | Backend |
|----------|---------|
| 1 | `mmseqs` easy-cluster if binary on PATH |
| 2 | Precomputed split TSV committed under `manifests/v8_pdbbind_*` (generated once) |
| 3 | Fallback: Biopython pairwise (dev-only; warn + size cap) |

No heavy DTIE / Normalizer dependency. Output: `manifests/v8_pdbbind_refined_cluster30_v1.json` with `train` / `val` / `core_test` protein lists + affinity labels.

### Label hygiene

- Prefer \(K_i\) / \(K_d\); convert to \(-\log_{10} K\) (M).  
- Drop IC50-only if marked unreliable in PDB-Bind index (flag in loader).  
- One complex → one receptor chain matching v8 Mode A/B conventions.

---

## Metrics & acceptance

Logged every epoch (val) and once on Core at end:

| Metric | Split |
|--------|-------|
| Pearson \(R\) | val, **Core** |
| Spearman \(\rho\) | val, **Core** |
| RMSE | val, **Core** |

| Gate | Pass |
|------|------|
| **`CORE_PEARSON`** | Core Pearson \(R \ge 0.40\) (provisional; rematch finetune if \(0.30 \le R < 0.40\)) |
| **`NO_LEAK`** | Assert zero Core PDB IDs / cluster IDs in train/val manifests |
| **`MOE_HELD`** | If MoE unfrozen: `moe_load_e* ≥ 0.05` on val batch (Sprint 9 floor) |
| **`NO_NAN`** | No NaN |
| Unit tests | Pool weights sum≈1; log/exp roundtrip; split purge removes Core-neighbors |

Fail below 0.30 Core Pearson after finetune rematch → new pre-reg (do **not** jump to ligand encoder as a silent fix).

---

## Files to add/touch

| Path | Role |
|------|------|
| `science/tokyo_eye/v8/affinity_head.py` | PocketGate + tangent pool + FFN |
| `science/tokyo_eye/v8/seq_cluster.py` | Cluster / identity holdout |
| `science/tokyo_eye/v8/pdbbind_loader.py` | Index parse + label → batch (protein-only) |
| `experiments/training/v8/build_pdbbind_splits.py` | Emit manifests |
| `experiments/training/v8/run_affinity_s10.py` | Train loop (isolated from dehydron harness or thin wrap) |
| `manifests/v8_pdbbind_refined_cluster30_v1.json` | Frozen split artifact |
| `tests/v8/test_affinity_sprint10.py` | Pool + split unit tests |
| Spec/plan under `docs/superpowers/{specs,plans}/` | This doc + plan after approval |

**Do not modify:** `biophysics.py` H-bond/cone, `r0_r5_graph.py` chemistry gates, graph cache version string (unless label metadata only).

---

## Frozen open points (approve to lock)

1. **Scope:** protein-only (Option C); ligand = Sprint 10.1.  
2. **Pool:** dehydron/mech/rim-gated tangent weighted mean at origin (`log₀`/`exp₀`).  
3. **Split:** Refined cluster@30% train/val; Core external test.  
4. **Seq tool:** vendored `seq_cluster.py` with mmseqs → precomputed TSV → Biopython fallback.  
5. **Default train:** `head_only` from `tokyo_eye_v8_mode_c_moe_rebalance_s9` (or Mode C best).  
6. **Label:** \(y = -\log_{10} K\) (M).

---

## Sprint 10.1 (out of scope here)

Ligand graph / interface edges / joint readout — only after protein-only Core gate passes or fails with a clear embedding-capacity diagnosis.

---

## Docs after approval

- Plan: `docs/superpowers/plans/2026-07-22-tokyo-eye-v8-sprint10-affinity-regression.md`  
- Then: splits artifact → head → unit tests → head_only train → Core metrics
