# Tokyo Eye v7 — Phase A nucleotide basin train (pre-registration)

**Status:** LOCKED before train  
**Date:** 2026-07-21  
**Θ resume:** `HEALTHY_V7_CKPT` (`v7_healthy_sealed.pt`)  
**Machine stamp:** [`data/gates/tokyo_eye_v7_phase_a_basin_train_prereg.json`](../../../data/gates/tokyo_eye_v7_phase_a_basin_train_prereg.json)  
**Make:** `make train-v7-phase-a-nucleotide-basin`  
**Hooks baseline:** `make grade-v7-hyp-mp-telemetry`  
**Post-train grade:** `make grade-v7-phase-a-nucleotide-basin`

## Claim

Continuing sealed Hyp-MP Θ with a **hyperbolic OFF↔ON basin contrastive** term teaches structure-level nucleotide macro-state separation on the ball, without allele micro-state training (Phase B deferred).

## Roster (nucleotide axis only — allele is label noise, not a loss)

| Basin | Nucleotide | PDB | Chain | Allele (ignored by loss) |
|-------|------------|-----|-------|--------------------------|
| OFF | GDP | **4LPK** | A | WT |
| OFF | GDP | **5US4** | A | G12D |
| ON | GppNHp | **6GOD** | A | WT |
| ON | GppNHp | **6GOF** | A | G12D |

Same Child-1 four-quadrant roster. Allele not used in Phase A loss.

## Geometry / stack locks

- `hyp_mp_primary=True`, `se3_aux=False`
- RiemannianAdam trunk (existing `build_optimizer`)
- Learned curvature; disc-health hold: **`disc_r_mean ≥ 0.25`**
- Do **not** overwrite `v7_healthy_sealed.pt` on this run

## Loss (locked)

Structure embedding \(z\): mean of `logmap0(x_hyp)` over residues in

\[
R_\star = (25\text{–}40) \cup (57\text{–}75) \cup N_{12}^{\mathrm{union}}
\]

(when \(N_{12}\) empty, use Switch∪all present \(R_\star\)).

For each train step, sample one OFF and one ON structure:

\[
L_{\mathrm{basin}} = \operatorname{ReLU}\!\big(m - d_{\mathbb{B}}(\exp_0(z_{\mathrm{OFF}}), \exp_0(z_{\mathrm{ON}}))\big)^2
\]

with **margin \(m = 0.50\)**, **coeff \(\lambda_{\mathrm{basin}} = 0.10\)**.

Plus existing health physics losses on each of the two forwards (cone / prototypes / disc floors as in `V7_BPRIME_STACK`). Same-basin attraction: **off** this phase (contrastive push only).

## Cheap Hyp-MP hooks (required deliverable)

One forward emits:

- per-edge geodesic length on Hyp-MP edges
- per-node strength = sum incident edge lengths
- top-10% nodes by strength (hub proxy)

Grade uses hooks for basin \(d_{\mathbb{B}}\) + hub lists; knockout is **spot-check only**.

## Pass form (post-train grade)

All required:

1. Disc hold: `disc_r_mean ≥ 0.25` on final / best eligible ckpt  
2. Cheap basin separation: mean \(d_{\mathbb{B}}\) of OFF centroid↔ON centroid **>** mean same-basin pair distance + **0.10** (on \(R_\star\) pooled embeddings)  
3. Hyp-MP audit: `hyp_mp_primary=true`  
4. Spot-check report: G12D migration ΔR and `raw_out_163_rises` vs sealed baseline (not a hard Pass this phase)

## Explicit non-goals

- Phase B allele / current-flow  
- Jacobian  
- Gini as train objective  
- Promoting over sealed healthy without separate seal prereg  
- Softening Child-1 allele bars

## Epochs / IO

| Field | Value |
|-------|--------|
| Epochs | 12 (default) |
| Run dir | `checkpoints/v7/runs/tokyo_eye_v7_phase_a_nucleotide_basin_v1/` |
| Manifest | `manifests/v7_phase_a_nucleotide_basin_v1.json` |
| Closeout | `data/gates/tokyo_eye_v7_phase_a_basin_train_closeout.json` |
