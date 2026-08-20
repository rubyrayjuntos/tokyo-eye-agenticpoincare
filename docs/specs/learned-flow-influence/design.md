# Learned flow influence — design (Option B)

**Date:** 2026-07-17  
**Status:** Spec locked; pilot + OOD triangulation complete (2026-07-18)  
**Standing conclusion:** Trunk (chem-MVP, **z-norm off**) tracks **symmetric** classical hubs but does **not** carry the physics-detected **directional / causal** G12D long-range signal — see [`ablation.md`](ablation.md) §Standing conclusion.  
**Standing probe defect:** Jacobian flow-influence on **z-norm-on** lineages is untrustworthy until fixed — [`JACOBIAN_ZNORM_DEFECT.md`](JACOBIAN_ZNORM_DEFECT.md).  
**Parent decision:** Pivot away from further disc-geometry reshaping (Option A).
The structured graph the GNN already has is for **communication flow** —
hubs, resistance, directional influence — not another Poincaré reshape.

## Thesis

Message passing on proteins is about **flow of communication**. Nodes embody
hubs vs isolates; tensors are the many ways that flow expresses itself. What the
GNN can infer (and what we should measure) is:

- which residues are hubs in learned propagation
- where influence is weak / strong / counter-intuitive
- how flow changes under mutation (and, later, PPI)

Disc-projection distances are explicitly **not** the instrument
([`DISC_PROJECTION_NOT_TRUNK_PROXY`](../../audit/DISC_PROJECTION_NOT_TRUNK_PROXY.md)).

## Method: Jacobian node-to-node influence

For each residue B:

1. Forward once; capture layer activation (`encoder_h` or `hyp_projections_2d`).
2. Scalar \( s_B = \| \mathrm{layer}[B] \|^2 \) (epsilon-floored).
3. One backward: \(\mathrm{influence}(A \to B) = \| \partial s_B / \partial x_A \|\).

N backward passes per structure give the full N×N directional influence matrix.
No new training. No locked-mechanism changes.

**Layer discipline:** run the probe twice — trunk (`encoder_h`) and disc
(`hyp_projections_2d`) — report separately, never pool.

## Derived quantities

| Quantity | Definition |
|----------|------------|
| Flow out-centrality | \(\mathrm{out}(A) = \sum_B I(A\to B)\) |
| Flow in-centrality | \(\mathrm{in}(B) = \sum_A I(A\to B)\) |
| Asymmetry index | mean \(\|I_{ab}-I_{ba}\|/(I_{ab}+I_{ba})\) over finite pairs |

Asymmetry near zero ⇒ symmetric distance-decay / oversmoothing (informative
negative, not a probe failure).

## External ground truth

Classical Cα-contact metrics, independent of DTIE ρ/τ:

- betweenness
- current-flow betweenness
- \|Fiedler eigenvector\| + algebraic connectivity
- ANM mean-square fluctuation

Implemented in `science/dtie/common/classical_network_metrics.py` (fail loudly
on empty contact graphs — no silent zeros).

## Hyperbolic numerical-safety audit (pre-implementation)

Forward path uses **geoopt** stereographic ops (`expmap0` / `logmap0` /
`mobius_add`), not the numpy helpers in `hyperbolic_lorentz_ops.py`.

| Danger zone | Status in this codebase / geoopt |
|-------------|----------------------------------|
| expmap0/logmap0 near origin (`‖v‖` divide) | **Guarded** in geoopt — `norm.clamp_min(1e-15)` throughout `_expmap0` / `_logmap0` |
| Möbius denominators | **Guarded** — `denom.clamp_min(1e-15)` |
| atanh domain | **Guarded** — `artanh` clamps to `±(1−1e-7)` |
| acosh / dist | **Guarded** via `artan_k` + project |
| sqrt-at-zero in this probe's own norms | **Probe-local** — `safe_norm_sq` / `EPS=1e-6` on `s_B` and gradient norms |
| Numpy lorentz helpers (`hyperbolic_lorentz_ops`) | Has `1e-12` / `1e-15` floors — **not** on the GNN autograd path |

Probe-local extras (not re-adding geoopt guards):

- run diagnostic in **float64**
- NaN/Inf ⇒ exclude + count, never zero-fill
- flag nodes with layer-norm `< EPS` as numerically unmeasurable
- `retain_graph` only until last B; free graph between structures
- optional ep0 / near-ep1 control for known-degenerate reference

## Instruments

- `experiments/diagnostics/jacobian_flow_influence.py`
- `science/dtie/common/classical_network_metrics.py`
- Gates / outcomes: [`ablation.md`](ablation.md)
