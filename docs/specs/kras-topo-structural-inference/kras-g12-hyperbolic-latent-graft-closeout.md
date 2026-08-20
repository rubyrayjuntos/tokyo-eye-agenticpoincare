# KRAS G12 hyperbolic latent graft — closeout (Child 4)

**Status:** PASS (OFF arm)  
**Date:** 2026-07-21  
**Machine stamp:** [`data/gates/kras_g12_hyperbolic_latent_graft_closeout.json`](../../../data/gates/kras_g12_hyperbolic_latent_graft_closeout.json)  
**Pre-reg:** [`kras-g12-hyperbolic-latent-graft-prereg.md`](kras-g12-hyperbolic-latent-graft-prereg.md)  
**Artifact:** `checkpoints/v66/diagnostics/routing_sparsity/kras_g12_hyperbolic_latent_graft.json`  
**Make:** `make grade-v66-kras-g12-hyperbolic-latent-graft`  
**Workstream:** `gnn_perturbation_boundary` child **4**  
**Checkpoint:** `FIX1_SPARSITY_CHAMPION_CKPT`

---

## Verdict

**Pass.** Post-lift `x_hyp` neighborhood graft at \(N_{12}\) improves hyperbolic-depth Spearman toward mut **more** than a full Euclidean neighborhood graft on the same depth metric, and beats hyp scramble.

| Condition | Result |
|-----------|--------|
| `4LPK` ← `5US4`, \|N₁₂\|=17 | |
| ρ(WT, mut) depth | 0.975 |
| ρ(hyp graft, mut) | 0.977 |
| ρ(euc neigh, mut) | 0.977 |
| ρ(hyp scramble, mut) | 0.899 |
| Δρ_hyp | +0.00199 |
| Δρ_euc_neigh | +0.00182 |
| Δρ_hyp > Δρ_euc | ✓ (thin) |
| hyp > scramble | ✓ (large) |

## Interpretation

- Embedding-space injection at the **inference** layer moves hyp-depth ranks toward mut without touching Euclidean MP / `data.x`.
- Margin vs Euclidean conduit on the same depth metric is **small** — backbone / construction still dominate absolute concordance (WT–mut depth already ρ≈0.975).
- Scramble collapse (0.899) shows the matched allele neighborhood is load-bearing for the hyp graft, not a generic ball-coordinate paste.

## Architecture honesty

Still **not** hyperbolic message-passing. Disc/Klein remain views. Does not reopen CB concordance or Ledger B.

## Next (deferred; need new pre-reg)

- Hop-2 Euclidean depth scaling  
- G12V/C allele fan-out  
- Hyperbolic MP trunk rewrite (architecture lock — separate decision)
