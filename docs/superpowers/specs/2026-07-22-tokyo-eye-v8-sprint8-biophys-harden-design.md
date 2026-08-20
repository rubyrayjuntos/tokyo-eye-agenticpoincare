# TokyoEye-v8 Sprint 8 — Harden Biophysical Filters & Mode C Scale

**Date:** 2026-07-22  
**Status:** APPROVED 2026-07-22 — open points frozen below  
**Depends on:** Sprint 6 loader + Sprint 7 live backbone  
**Goal:** Replace distance-envelope H-bonds + isotropic wrap spheres with DSSP electrostatic gating + double-cone nonpolar shielding, then scale training to the KRAS nucleotide basin (Mode C).

---

## Frozen open points (2026-07-22)

1. **H placement:** reconstruct amide H at **1.01 Å** along the planar angle bisector of ∠C(prev)–N–Cα.
2. **Double cone:** `abs(dot(v̂, û)) >= cos(45°) ≈ 0.7071` with û = H→O, v̂ = m→c.
3. **τ wrap:** start `DEHYDRON_WRAP_MAX=19`; if 4OBE `dehydron_frac >= 0.60` after Epoch 0, dump wrap histogram, set τ to the empirical median, then descend until `dehydron_frac < 0.60` (or τ=0).

---

## Problem

Mode A/B report `dehydron_frac ≈ 0.90+` on many structures. R1/R2 classification today is:

1. CA–CA spatial/seq envelope → candidate H-bond  
2. Isotropic carbon count within 6.5 Å of N–O midpoint → wrap ≤ 19 ⇒ R2  

That over-admits weak contacts and over-labels dehydrons, so `val_dehydron_auprc` saturates and loses allosteric sensitivity.

---

## Non-goals

- Full mkdssp / DSSP binary dependency (we implement the classic Kabsch–Sander energy in-process)  
- Changing hyp attention / MoE / SE(3)-lite contracts  
- Writing through Normalizer / ingest  

---

## Architecture

```
PDB residues (with CA)
  → place backbone H (N–H from C(i-1)–N–CA plane; Kabsch–Sander)
  → DSSP energy E on (N,H,C,O) quad → keep if E ≤ −0.5 kcal/mol
  → double-cone wrap count on nonpolar carbons about H⋯O axis
  → wrap ≤ τ_wrap ⇒ R2 else R1
  → cache graph tensors under pdb_cache/v8_graph_cache/
```

---

## 1. DSSP energetic gate (R1/R2 eligibility)

**Location:** `science/tokyo_eye/v8/biophysics.py` (+ call site in `r0_r5_graph._detect_backbone_hbonds`)

Classic Kabsch–Sander electrostatic surrogate (partial charges \(q_{\mathrm{NH}}=0.42\), \(q_{\mathrm{CO}}=0.20\)):

\[
E = 332\,q_{\mathrm{NH}}\,q_{\mathrm{CO}}\left(\frac{1}{r_{\mathrm{ON}}} + \frac{1}{r_{\mathrm{CH}}} - \frac{1}{r_{\mathrm{OH}}} - \frac{1}{r_{\mathrm{CN}}}\right)\ \mathrm{kcal/mol}
\]

with atom pairs from donor backbone **N, H** and acceptor backbone **C, O**.

| Constant | Value |
|----------|-------|
| `DSSP_ENERGY_CUTOFF` | **−0.5** kcal/mol (pair admitted iff `E ≤ cutoff`) |
| H placement | Ideal N–H along bisector of ∠C(prev)–N–CA when H missing |
| Prefilter | Keep existing seq/spatial envelope as cheap reject (optional) |

**Contract:** No energy-passing pair ⇒ no R1/R2 edge (R0/R3–R5 unchanged).

---

## 2. Double-cone wrapping (dehydron severity)

**Replace** isotropic ball count in `compute_bond_wrapping_count` with:

1. Axis \(\hat{u}\) along H→O (fallback N→O)  
2. Midpoint \(m\) of H–O  
3. Count nonpolar wrapping carbons \(c\) with:
   - \(\|c - m\| ≤ R_{\mathrm{wrap}}\) (6.5 Å default)  
   - angle to axis ≤ \(\alpha_{\mathrm{cone}}\) **or** ≥ \(180° - \alpha_{\mathrm{cone}}\) (double cone)  

| Constant | Default | Notes |
|----------|---------|--------|
| `WRAPPING_RADIUS` | 6.5 Å | unchanged |
| `WRAP_CONE_HALF_ANGLE_DEG` | **45°** | tunable; log in `graph_meta` |
| `DEHYDRON_WRAP_MAX` | **19** initially | **re-measure** after DSSP; expect lower dehydron_frac — may retune in closeout |

Polar sidechains still excluded via `counts_as_wrapping_carbon`.

---

## 3. Energy / graph cache (Quadro-friendly)

**Yes — cache enabled by default.**

Path: `pdb_cache/v8_graph_cache/{pdb}_{chain}_{hash}.pt`  
Hash covers: DSSP cutoff, cone angle, wrap radius, τ_wrap, parser version string `v8_biophys_s8`.

Contents: `edge_index`, `edge_type`, `edge_attr`, `meta`, `dehydron_labels`, `num_nodes`, `ca_coords`.

CLI: `--no-graph-cache` to force rebuild; Makefile `NO_GRAPH_CACHE=1`.

---

## 4. Mode C — KRAS nucleotide basin

- Manifest: `manifests/v8_kras_nucleotide_basin_v1.json` (thin wrap of `v7_phase_a_nucleotide_basin_v1.json` proteins + v8 description)  
- PDBs: `4LPK`, `5US4`, `6GOD`, `6GOF` (already in `pdb_cache` per platform)  
- Harness: `MANIFEST=manifests/v8_kras_nucleotide_basin_v1.json make train-v8-experiment`  
- Log per-step: `pdb_id`, `dehydron_frac`, `n_r1`, `n_r2`, `diag_manifold_entropy`, `val_dehydron_auprc`

---

## Acceptance

| Gate | Pass |
|------|------|
| 4OBE dehydron_frac | **≪ 0.90** after DSSP+cone (target band **0.15–0.55** provisional) |
| Energy unit test | Known DSSP-style pair admits/rejects at −0.5 |
| Cone unit test | Carbon on-axis counted; equatorial carbon at same radius rejected |
| Cache | Second load of same PDB hits cache (mtime/hash) |
| Mode C | 4 KRAS structures train ≥1 full round-robin without NaN |

---

## Open point (freeze on approval)

**Wrap threshold τ:** keep `19` for first Mode C run, then set `DEHYDRON_WRAP_MAX` from the empirical 4OBE wrap histogram median if dehydron_frac still > 0.6.

---

## Docs to write on approval

- Plan: `docs/superpowers/plans/2026-07-22-tokyo-eye-v8-sprint8-biophys-harden.md`  
- Then implement in `biophysics.py` / `r0_r5_graph.py` / `loader.py` + Mode C manifest + tests + run
