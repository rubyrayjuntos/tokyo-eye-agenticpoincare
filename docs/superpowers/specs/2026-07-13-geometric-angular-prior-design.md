# Geometric Angular Prior (Fix 1) — Design

**Date:** 2026-07-13  
**Status:** Implemented — **sibling gate table PASS** on `v66_best_disc.pt` (2026-07-14 re-audit); **does not claim v7 S1**  
**Parent diagnosis:** Pre-MoE `AngularHead` is an unconstrained MLP; disc `(r, θ)` is not geometrically grounded, so empty θ bins have 0% expert soft mass and MoE cannot “fill the gap.”  
**Audit SSOT:** `checkpoints/v66/diagnostics/geom_angular_prior_gate_audit/` — gate on `best_disc`, not `phase_12.pt` (latter can drop `geometric_angular_prior` from `training_config`).  
**Caveat:** Disc angular fill is partly **by construction** from dehydron/peptide `θ_prior`; registered gates test radial/structure consistency, not HTML occupancy. Cold 10-ep viewers are exploratory (doc epoch count = 15–20).

## Goal

Ground the **pre-MoE Poincaré disc angle** in dehydron / peptide geometry so `MöbiusLinear` receives a real compass. Keep radial / burial capacity intact.

## Non-goals

- Expert sector recruit / stronger rim fanout / HTML or Möbius rotation-as-fill  
- Learnable blend `κ` on the first sibling  
- True equivariant `1e` readout (Fix 3) — deferred  
- Changing Normalizer / ingest / onboard contract

## Architecture

### Injection point (critical)

Apply the prior at the **2D disc**, after Möbius projection (and after rim fanout if enabled), because ep164 uses `disc_radial_source=mobius`: disc θ is the Möbius shadow of `x_hyp`, not AngularHead’s H-vector itself.

```text
… → x_hyp → MöbiusLinear₂ᴅ → [rim fanout] → apply geometric θ prior + residual → project_disc₂ᴅ
                                                              ↑
                                                    û_prior from atoms
```

Keep RadialHead × AngularHead → `expmap₀` for the hyperbolic MoE trunk unchanged on the first sibling (depth / features). Only the **active pre-routing disc `(r, θ)`** is re-angled.

Disc update:

- Keep radius `r` from the current 2D embedding (burial channel intact).  
- Replace direction:  
  `θ = θ_prior + α · tanh(δ)`, `α = π/4` fixed,  
  `δ = DiscAngularResidualMLP(backbone_x)` scalar (new small head; not free H-dim AngularHead).  
- `xy ← r · (cos θ, sin θ)`.

### Compass cascade (SSOT wrapping)

Per residue, in a protein-global plane (top-2 PCA axes of Cα; fixed for the structure):

1. **Peptide fallback** `û_peptide`: unit projection of `Cα → C` (else local N–Cα–C in-plane axis) onto the global plane.  
2. **Dehydron primary**: for each underwrapped backbone H-bond touching the residue (`ρ_bond < τ`, same SSOT as role edges), take donor→acceptor unit vector → project onto global plane → weight `w_k = softplus(τ − ρ_bond,k)`.  
3. **Resultant mass** (cancellation-safe):  
   `v_dh = Σ_k w_k û_k`, `m = ‖v_dh‖`.  
4. **Blend:**  
   `σ(m) = 1 − exp(−m / κ)` with **`κ` fixed** (default `1.0` times mean mass scale; grid later if needed).  
   `v_blend = v_dh + (1 − σ(m)) · û_peptide`,  
   `û_prior = v_blend / ‖v_blend‖` (peptide ensures non-zero at `m→0`).  
5. `θ_prior = atan2(û_prior · e₂, û_prior · e₁)`.

No H-bond energy channel. No hard dehydron/peptide switch.

### Optional fidelity loss (first sibling: light)

`L_angle = min(1, r²) · (1 − cos(θ_disc − θ_prior))` with small coeff (e.g. `0.05–0.15`) so residual cannot walk away from the prior. Radial guard via `r²`.

## Training sibling

- Resume: `feeler_expand_23_rim_fanout_coverage_v2/epochs/epoch_164.pt`  
- Corpus: feeler expand 23; anchors `1F88`  
- Flag: `--geometric-angular-prior` (+ residual MLP, fixed `κ`, `α=π/4`)  
- Epochs: 15–20; lr `5e-5`; **do not** raise `rim_fanout_strength`; no sector recruit  
- New `RUN_ID` e.g. `feeler_expand_23_geom_angular_prior_v1`

## Gates

| Gate | Pass |
|------|------|
| `corr(r, depth)` / probe | ≥ baseline − 0.08 |
| 1F88 largest gap | not worse by > ~10° vs ep164 |
| 4OBE circ-R | ≲ 0.60 (or clear improvement vs ep164 free-disc baseline under same loader) |

## Tests

- Zero dehydron → peptide unit prior, no NaN  
- Opposite dehydrons → `m→0` → peptide fallback  
- Residual bound `|Δθ| ≤ α`  
- Forward with flag preserves `r` correlation path (smoke)

## Approval checklist

- [x] Disc-level injection (not H-dim AngularHead rewrite) accepted  
- [x] Dehydron wrapping-deficit blend + peptide fallback + `m=‖Σwu‖` accepted  
- [x] Fixed `κ`, `α=π/4` accepted  
- [x] ep164 sibling scope / gates accepted  

**Implementation:** `science/dtie/v66/geometric_angular_prior.py`, model/loss/launch wire,  
`make train-v66-feeler-geom-angular-prior`.
