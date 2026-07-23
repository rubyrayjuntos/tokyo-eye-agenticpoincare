# TokyoEye-v8 Sprint 10.1 — Cross-Space Ligand Interface Graphs

**Date:** 2026-07-22  
**Status:** APPROVED 2026-07-22 — implement per plan  
**Depends on:** Sprint 10 protein-only affinity fail (Core Pearson \(R \approx 0.007\) after `finetune_hyp`) — **capacity diagnosis**  
**Parent:** [`2026-07-22-tokyo-eye-v8-sprint10-affinity-regression-design.md`](2026-07-22-tokyo-eye-v8-sprint10-affinity-regression-design.md)  
**Splits SSOT (unchanged):** `manifests/v8_pdbbind_refined_cluster30_v1.json`  
**Plan:** [`docs/superpowers/plans/2026-07-22-tokyo-eye-v8-sprint10-1-ligand-interface.md`](../plans/2026-07-22-tokyo-eye-v8-sprint10-1-ligand-interface.md) (10-channel table frozen there)

---

## Problem (why 10.1 exists)

Protein-only \(-\log K\) regression is **informationally underdetermined** when homologous pockets bind chemically distinct ligands. Sprint 10 `head_only` (Core \(R \approx 0.054\)) and `finetune_hyp` (Core \(R \approx 0.007\)) confirm the hyperbolic trunk cannot invent ligand identity from R0–R5 residue graphs alone. Per the frozen Sprint 10 contract: **do not** inflate metrics via seed/LR fiddling; activate the pre-registered pivot — **ligand chemistry into the readout field**.

---

## Non-goals

- Mutating R0–R5 chemistry gates, DSSP/cone wrap, or `v8_biophys_s8` protein graph cache semantics  
- Routing R6 through MoE / HypGraphAttention as a 7th spine relation (keeps Sprint 8/9 MoE guild contracts intact)  
- Full Open Babel / RDKit dependency stack for 10.1.0  
- Claiming docking ΔG equivalence (labels remain PDB-Bind experimental \(-\log_{10} K\))  
- Silent return to protein-only rematches as a “fix”

---

## Frozen operational contracts

| Contract | Lock |
|----------|------|
| Ligand source (production) | **Option C — Hybrid:** prefer official PDBBind ligand `.mol2`/SDF when staged; else filtered HETATM |
| Ligand source (Sprint **10.1.0**) | **Option A — Filtered HETATM** from `pdb_cache/{PDB}.pdb` (hardcoded bootstrap) |
| Protein graph cache | **Untouched** — R6 computed on-the-fly, never written into `pdb_cache/v8_graph_cache/` |
| R6 distance | \(d(\mathbf{x}_{\mathrm{res}}, \mathbf{x}_{\mathrm{lig}}) \le 4.5\,\text{Å}\) |
| Residue proxy | \(C_\beta\) (or \(C_\alpha\) for Gly) |
| Attention | Asymmetric interface cross-attn: \(Q \leftarrow z_{\mathrm{hyp}}\), \(K,V \leftarrow\) ligand atom feats along R6 |
| Splits / leak wall | Reuse Sprint 10 manifest; `NO_LEAK` still absolute |

---

## Architecture

```
pdb_cache/{PDB}.pdb
  ├─ chain residues → [FROZEN] R0–R5 loader / graph cache → spine → z_hyp [N_res, d]
  └─ HETATM (10.1.0) / mol2 (later) → ligand atoms x_lig [N_lig, d_lig]
                                            │
                         R6 on-the-fly: d(Cβ/Cα, lig_atom) ≤ 4.5 Å
                         edge_index_r6: [2, E6]  (res_i ↔ lig_j), bidirectional
                                            │
              JointPocketAffinityHead
                Q = Linear_q(z_hyp)
                K,V = Linear_{k,v}(Embed(onehot_lig))
                sparse cross-attn along R6 → residue msgs
                PocketGate (mech/dehyd/rim) × msgs → tangent pool → AffinityFFN → ŷ
```

**Invariant:** Spine forward identical to Sprint 10 except the affinity head consumes `(z_hyp, mechanism_score, dehydron_labels, lig_feats, edge_index_r6)`.

---

## 1. Ligand atom loader (`pdbbind_loader.py`)

### Sprint 10.1.0 — HETATM sanitization matrix (Option A)

Parse `HETATM` (and connected `TER`/`END` scoping) from the complex PDB. Drop:

| Rule | Match |
|------|--------|
| Water | residue name ∈ `{HOH, WAT, DOD, TIP}` |
| Common ions / solvent ions | residue name ∈ `{CL, NA, K, MG, ZN, CA, SO4, PO4}` (exact token; case-normalized) |

**Selection contract:** among remaining hetero residues (grouped by `(chain, resseq, icode, resname)`), select the **largest multi-atom** component (\(N_{\mathrm{atoms}} \ge 2\)). That residue index is the bootstrap ligand. If none qualify → soft-skip complex (same soft-skip path as unloadable proteins).

### Production Option C (post-10.1.0)

```
if ligand_mol2_or_sdf.exists():
    parse_mol2/sdf → atoms (element, formal_charge, coords)
else:
    filtered_hetatm_bootstrap(...)
```

Staging path (future): `data/pdbbind/ligands/{pdb_id}.mol2` (not required for 10.1.0).

### Ligand node features (bootstrap minimum)

Per ligand atom, static features before projection — **exact 10-channel layout is frozen in the implementation plan** (channels 0–6 element; 7–9 charge bins; HETATM default neutral). Spec stays channel-count-agnostic beyond requiring `float32 [N_lig, 10]` for Sprint 10.1.0.

---

## 2. R6 — protein–ligand contact edges

| Property | Value |
|----------|-------|
| Type code | `R6_PROTEIN_LIGAND = 6` (affinity-path constant only; **not** written into R0–R5 `edge_type` tensors used by MoE) |
| Geometry | Undirected → emit both directions |
| Residue endpoint | \(C_\beta\) if present else \(C_\alpha\) (Gly) |
| Ligand endpoint | Ligand atom center |
| Threshold | \(4.5\,\text{Å}\) inclusive |
| Cache | **Never** enter `v8_biophys_s8` graph cache |
| Empty R6 | Soft-skip or zero-message path with explicit `r6_empty=1` metric (prefer soft-skip if \(E_6=0\) after threshold) |

Implementation home: new helper `science/tokyo_eye/v8/ligand_interface.py` (keeps `r0_r5_graph.py` frozen). Called from affinity batch assembly, not from `build_r0_r5_graph`.

---

## 3. Joint pocket-gated readout (`affinity_head.py`)

### Asymmetric sparse cross-attention

For each R6 edge \((i \rightarrow j)\) with residue \(i\), ligand atom \(j\):

\[
\alpha_{ij} = \mathrm{softmax}_{j \in \mathcal{N}_{\mathrm{R6}}(i)}\!\left(\frac{(W_Q z_i)^\top (W_K \ell_j)}{\sqrt{d_k}}\right),\quad
m_i = \sum_{j} \alpha_{ij}\, W_V \ell_j
\]

Residues with empty neighborhood get \(m_i = 0\).

### Pocket gate + pool (reuse Sprint 10 geometry)

Keep dehydron/mech/rim gate on protein nodes. Fuse:

\[
h_i = [z_i ; m_i] \quad\text{or}\quad z_i + m_i
\]

(design default: **residual add** after `Linear_m(m_i)` to preserve Poincaré ball ops on \(z\)). Then existing:

\[
w = \mathrm{softmax}(\mathrm{score}(h_{\mathrm{inv}}, r, \sigma(\mathrm{mech}), \mathrm{dehyd}),\ \mathrm{dim}=0)
\]
\[
t_{\mathrm{pool}} = \sum_i w_i \log_0(z_i),\quad \hat y = \mathrm{FFN}(\log_0(\exp_0(t_{\mathrm{pool}})))
\]

Softmax **must** remain `dim=0` over variable \(N_{\mathrm{res}}\).

### Train modes

| Mode | Frontend | Hyp trunk / MoE | Ligand embed + R6 attn + AffinityFFN |
|------|----------|-----------------|--------------------------------------|
| `head_only` | freeze | freeze | train |
| `finetune_hyp` | freeze | train | train |

Default 10.1.0: `finetune_hyp` from Sprint 9 spine + new head (do **not** warm-start failed protein-only affinity FFN as the sole hope — random-init joint head OK; optional warm-start only of PocketGate trunk layers if shapes match).

---

## Approaches considered (readout placement)

| Option | Idea | Trade-off |
|--------|------|-----------|
| **A (recommended)** | R6 + ligand attn **only in affinity head**; spine stays R0–R5 | Clean freeze of MoE/cache; matches capacity diagnosis |
| B | Expand MoE `num_relations` to 7 and message-pass R6 in spine | Breaks frozen MoE load floors / cache hash; out of scope |
| C | Early-fuse ligand into Equiformer frontend | Touches SE(3) bank contract; deferred |

**Recommendation: A.**

---

## Metrics & acceptance (unchanged gates, new telemetry)

Reuse Core / val Pearson, Spearman, RMSE. Add:

| Metric | Meaning |
|--------|---------|
| `r6_edges_mean` | Mean \(E_6\) per complex |
| `lig_atoms_mean` | Mean ligand heavy atoms |
| `n_lig_skip` | Complexes dropped by sanitization / empty R6 |

| Gate | Pass |
|------|------|
| `CORE_PEARSON` | Core \(R \ge 0.40\) (provisional; rematch if \(0.30 \le R < 0.40\)) |
| `NO_LEAK` | Unchanged |
| `CACHE_INTACT` | Graph-cache hash / `v8_biophys_s8` unchanged |
| `R6_OFF_CACHE` | Unit test: R6 builder never calls graph-cache write |
| `NO_NAN` | No NaN |

Fail below 0.30 after one `finetune_hyp` rematch → **new pre-reg** (ligand feature richness / interface radius), not silent protein-only retry.

---

## Files to add/touch

| Path | Role |
|------|------|
| `docs/superpowers/specs/2026-07-22-tokyo-eye-v8-sprint10-1-ligand-interface-design.md` | This design |
| `science/tokyo_eye/v8/ligand_interface.py` | HETATM sanitize + R6 builder (on-the-fly) |
| `science/tokyo_eye/v8/pdbbind_loader.py` | Ligand extract API for affinity batches |
| `science/tokyo_eye/v8/affinity_head.py` | `JointPocketAffinityHead` (R6 cross-attn + gate) |
| `experiments/training/v8/run_affinity_s10.py` | Batch fields + head swap; mode flags |
| `tests/v8/test_ligand_interface_sprint101.py` | Sanitize / R6 distance / cache isolation / attn shapes |
| Plan (after approval) | `docs/superpowers/plans/2026-07-22-tokyo-eye-v8-sprint10-1-ligand-interface.md` |

**Do not modify:** `biophysics.py` H-bond/cone, `r0_r5_graph.py` relation IDs / exclusivity, graph cache version string.

---

## Open points (approve to lock)

1. **Ligand source:** Option C production + Option A Sprint 10.1.0 bootstrap — **LOCKED**.  
2. **R6 threshold / residue proxy:** 4.5 Å, Cβ/Cα(Gly) — **LOCKED**.  
3. **R6 placement:** affinity-head only (Approach A) — **LOCKED**.  
4. **Ligand one-hot channel table:** 10-channel (7 element + 3 charge) — **LOCKED in plan**.  
5. **Default train progression:** `head_only` smoke → `finetune_hyp` — **LOCKED**.

---

## Sprint 10 closeout pointer

Protein-only Option C closed as **capacity-limited**. Artifacts remain archaeology under `checkpoints/v8/runs/tokyo_eye_v8_affinity_s10_*`. New workstreams start from this Sprint 10.1 design, not from further protein-only LR sweeps.

---

## Docs after approval

- Implementation plan: `docs/superpowers/plans/2026-07-22-tokyo-eye-v8-sprint10-1-ligand-interface.md`  
- Then: ligand sanitize + R6 unit tests → joint head → `head_only` smoke → `finetune_hyp` Core eval  
