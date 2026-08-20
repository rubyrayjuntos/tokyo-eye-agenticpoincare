# KRAS G12 hyperbolic latent graft — pre-registration (Child 4)

**Status:** CLOSED — Pass (see [`kras-g12-hyperbolic-latent-graft-closeout.md`](kras-g12-hyperbolic-latent-graft-closeout.md))  
**Date locked:** 2026-07-21  
**Workstream:** `gnn_perturbation_boundary` child **4**  
**Depends on:** Children 1–3 (Euclidean single-site + OFF/ON neighborhood) Pass  
**Checkpoint:** `FIX1_SPARSITY_CHAMPION_CKPT`  
**Make:** `make grade-v66-kras-g12-hyperbolic-latent-graft`  
**Stamp:** `data/gates/kras_g12_hyperbolic_latent_graft_prereg.json`  
**Artifact:** `checkpoints/v66/diagnostics/routing_sparsity/kras_g12_hyperbolic_latent_graft.json`

---

## Motivation

Euclidean neighborhood grafts showed conduit flex helps (~5–7× single-site Δρ) but absolute Δρ stays small under backbone inertia. Hypothesis: injecting the mut’s **post-lift** ball coordinates at the allele neighborhood moves **hyperbolic inference** fields toward mut more than Euclidean feature/Cα grafts — without relocating message-passing into hyperbolic space.

## Architecture honesty (non-negotiable)

Per [`EUCLIDEAN_CONSTRUCTION_HYPERBOLIC_INFERENCE.md`](../../audit/EUCLIDEAN_CONSTRUCTION_HYPERBOLIC_INFERENCE.md):

- MP trunk / `encoder_h` remain **Euclidean construction**.  
- This probe edits **`x_hyp` (or `logmap0(x_hyp)`)** after lift — **hyperbolic inference** intervention.  
- Disc / Klein projections are **report-only views**, not Pass metrics.  
- This is **not** hyperbolic message-passing and does **not** reopen “move the trunk to hyp MP.”

## Scope

**Primary arm:** OFF **`4LPK` / `5US4`** (chain A).  
ON arm optional report-only after OFF grades.  
Hop-2 Euclidean and G12V/C remain separate children.

## Protocol

1. Forward WT and mut with champion (same batch prep as neighborhood graft).  
2. Capture post-lift ball embeddings `x_hyp` (and curvature \(c\)) per residue; optionally `logmap0` tangents.  
3. Build \(N_{12}\) as in Child 2 (mut Cα 1-hop ∪ Switch locks ∩ WT).  
4. **Hyp neighborhood graft:** on WT’s forward graph, replace `x_hyp[i]` for each \(r \in N_{12}\) with mut’s aligned `x_hyp` at \(r\) (shared resseq); keep Euclidean `data.x` / edges as WT. Re-run **downstream-of-lift** heads only if the model API allows; otherwise measure immediate post-edit hyp fields (depth / pairwise hyp distance ranks) without claiming a second full MP.  
5. **Controls:**
   - Euclidean neighborhood graft (Child 2 recipe) on same arm — baseline Δρ on `out_effect` **and** on hyp depth ranks.  
   - Hyp scramble: distal mut `x_hyp` donors → WT \(N_{12}\) (same scramble scheme as Child 2).  
6. **Metric family (locked primary):** Spearman of **hyperbolic depth** `dist0(x_hyp)` (aligned shared residues) vs mut. Secondary report: cone_depth if available; `encoder_h` out_effect unchanged by hyp-only edit (expected).

**Implementation note:** If the forward cannot surgically splice `x_hyp` mid-graph without a full re-forward, the allowed recipe is: (a) full WT forward → capture; (b) splice `x_hyp`; (c) re-execute MoE/uncertainty from spliced ball state. Forbidden: re-running Euclidean MP after pretending hyp edit changed `data.x`.

## Pass form (OFF arm)

Both required:

1. Δρ_hyp = ρ(depth_hyp_graft, depth_mut) − ρ(depth_WT, depth_mut)  
   **>** Δρ_euc = ρ(depth_after_euc_neigh_graft, depth_mut) − ρ(depth_WT, depth_mut)  
   i.e. hyperbolic latent neighborhood graft improves depth concordance toward mut **more** than Euclidean neighborhood graft.  
2. ρ(depth_hyp_graft, depth_mut) > ρ(depth_hyp_scramble, depth_mut)

**Report-only:** |N₁₂|; disc overlays; ratio Δρ_hyp / Δρ_euc; ON arm if run.

## Interpretation

- **Pass:** post-lift manifold injection carries allele-neighborhood strain into hyp salience better than Euclidean conduit graft — supports “perturbation at hyperbolic inference layer.”  
- **Fail:** hyp latent graft does not beat Euclidean conduit — backbone/construction still dominate; hyperbolic MP remains deferred.

## Explicit non-goals

- Relocating EquivariantConv / role-edge MP into hyperbolic space  
- Disc-alone Pass  
- Jacobian rankings  
- Claiming physical spacetime curvature of the protein
