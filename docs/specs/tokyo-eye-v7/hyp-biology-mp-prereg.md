# Tokyo Eye v7 — Hyp biology MP child lineage (pre-registration)

**Status:** LOCKED — implement + continue-train under RUN_ID  
**Date:** 2026-07-21  
**Lineage / RUN_ID:** `tokyo_eye_v7_hyp_biology_mp_v1`  
**Θ resume (weights only):** `HEALTHY_V7_CKPT` (`v7_healthy_sealed.pt`)  
**Machine stamp:** [`data/gates/tokyo_eye_v7_hyp_biology_mp_prereg.json`](../../../data/gates/tokyo_eye_v7_hyp_biology_mp_prereg.json)  
**Train:** `make train-v7-hyp-biology-mp` → `experiments.training.v7.hyp_biology_mp_train`  
**Smoke:** `make grade-v7-hyp-biology-mp-smoke`  
**Parent design:** this file + [`hyp-biology-mp-design.md`](hyp-biology-mp-design.md)  
**Does not overwrite:** `v7_healthy_sealed.pt`, Fix-1 / v66 champions, default `TokyoEye-v7` sealed path

## Claim

A **child lineage** where Hyp MP neighbors are **structured biology edges only** (H-bond, dehydron, π-stack, salt bridge). Euclidean Cα contact may inform features / pre-lift priors but **must never** enter `MobiusGraphConv` / Hyp MP `edge_index`. Residual degree-0 nodes stay self-only (Option A) so v1 ablation maps 1:1 to the biology ontology.

## Continue-train (v1)

| Field | Value |
|-------|--------|
| Resume | `HEALTHY_V7_CKPT` |
| Corpus default | `manifests/v6_corpus_stage_a_small_v1.json` |
| Loss | physics / disc occupancy (B′-style coeffs); no allele / Jacobian |
| Disc hold | `disc_r_mean ≥ 0.25` (abort after 2 consecutive misses) |
| Best saver | `checkpoints/v7/runs/tokyo_eye_v7_hyp_biology_mp_v1/v7_hyp_biology_mp_best.pt` |
| Per-step | attach biology graph; assert `ca_in_mp=false` |

## Why (vs sealed Θ)

Sealed v7 may still resolve Hyp MP edges via Cα `edge_index` fallback (`resolve_hyp_mp_edges`). That is soft merge. This lineage fail-closes: `allow_ca_fallback=false`.

## Gated paths

| Flag | Value |
|------|--------|
| `hyp_biology_mp` | `true` |
| `hyp_mp_primary` | `true` |
| `se3_aux` | `false` |
| `allow_ca_fallback` | `false` |
| `degree_zero_nodes_allowed` | `true` |

## Edge ontology

### Allowed (Hyp MP only)

| Type | Detection (v1) | Edge attr (minimal) |
|------|----------------|---------------------|
| `hbond` | Existing SSOT backbone H-bond | underwrap / angle as today |
| `dehydron` | Existing underwrap witness (`ρ < τ`) | underwrap / angle |
| `salt_bridge` | +N (Arg NH1/NH2, Lys NZ, His ND1/NE2) ↔ −O (Asp OD1/OD2, Glu OE1/OE2); heavy-atom dist ≤ **4.0 Å** | `[charge_product, min_dist]` |
| `pi_stack` | Aromatic ring centroids (Phe, Tyr, Trp, His); centroid dist ≤ **5.5 Å**; face-to-face dihedral **&lt; 30°** or edge-to-face **&gt; 60°** | `[centroid_dist, dihedral_angle, alignment_type_idx]` |

### Forbidden in Hyp MP

`ca_contact`, `packing_euc`, `se3_rel_xyz`, and any silent fallback to Cα `edge_index`.

### Residual isolates (Option A)

- Degree-0 nodes: **no** neighborhood messages; pointwise Möbius / identity path only.
- Placement still from radial/angular pre-lift into `x_hyp`.
- **Not in v1:** virtual origin (B), post-lift hyp \(d_{\mathbb{H}}\) kNN (C), solvation-shell Cβ policy (1).
- **v2 pivot if needed:** Option C only (hyp proximity after pre-lift), still Cα-free.

## Pipeline (normative)

```
PDB residues
  → extraction: hbond | dehydron | pi_stack | salt_bridge
  → (no Cα edges attached for MP)
  → mark degree-0 (allowed)
  → radial/angular pre-lift → x_hyp
  → Hyp MP on biology edge_index only
  → audit: ontology counts + ca_in_mp==false
```

## Audit trail requirements

- `hyp_mp_edges=biology`
- `ca_in_mp=false`
- `expected_edge_ontology`: `hbond`, `dehydron`, `pi_stack`, `salt_bridge`
- `forbidden_edge_ontology` absent from MP edges
- Counters: per-type edge counts, degree-0 node count / fraction, assert zero Cα leakage at load and forward

## Lineage isolation

| Artifact | Path |
|----------|------|
| Run dir | `checkpoints/v7/runs/tokyo_eye_v7_hyp_biology_mp_v1/` |
| Prereg | `data/gates/tokyo_eye_v7_hyp_biology_mp_prereg.json` |
| Closeout (later) | `data/gates/tokyo_eye_v7_hyp_biology_mp_closeout.json` |
| MLflow | experiment `tokyo-eyes-v7`; run name = lineage_id; **do not** promote over sealed healthy without separate seal prereg |

Forbidden on this run: overwrite `HEALTHY_V7_CKPT` / `v7_healthy_sealed.pt`; enable Cα MP fallback; inject B/C connectivity; Jacobian hub grades as Pass criteria.

## Pass form (v1 smoke — report-first)

| Check | Bar |
|-------|-----|
| Forward completes with `hyp_biology_mp` | required |
| `ca_in_mp == false` | hard fail if false |
| Edge ontology ⊆ allowed; forbidden count == 0 | hard fail |
| Degree-0 allowed; count logged | report |
| Disc / basin / hub grades | **report-only** on v1 unless amended |

## Non-goals (v1)

- Option B/C connectivity
- Solvation-shell edges
- Changing sealed Θ default forward
- Chem-MVP reopen / Euc shortcut MP
