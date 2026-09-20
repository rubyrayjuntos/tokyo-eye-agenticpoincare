# Tokyo Eye EQU — MoE Eval Utilization (hard gate + Switch-style LB)

**Gate ID:** `tokyo_eye_equ_moe_eval_util`  
**Status:** APPROVED_LOCKED  
**Date:** 2026-09-16  
**Approver:** Bot (Ray ordered ahead of ckpt selection / spread_hold)  
**Predecessor:** geoopt_restore QUALIFIED; stripe crosstab = MoE expert↔radius 1.0; eval E0=E3=0  

## 1. Intent

Close the designed blind spot: train-step `moe_load_*` under Gumbel STE looked healthy while **eval argmax** on the QUALIFIED ckpt uses only E1/E2 and parks two fixed radii (~0.18 / ~0.38). Seal **eval-mode** expert utilization; strengthen load-balance aux on **soft** gate probs (Switch-style), not only hard STE one-hots.

## 2. Diagnosis (locked evidence)

1. Stripe membership ≡ MoE expert (ARI/NMI/acc = 1.0); biology/R0–R5/parity ≈ majority baseline.
2. `z_lift` stripe ̸≈ final stripe (ARI ≈ −0.02) → MoE (post-attn), not projector.
3. MLflow train `moe_load_e*` on geoopt_restore never dropped below ~0.10 — **false healthy**.
4. Same ckpt, eval argmax: mean load ≈ `[0, 0.63, 0.37, 0]`; train Gumbel-8 ≈ balanced ~0.25 each.
5. Existing `cv_loss` + `quota_loss` run on hard Gumbel routing during train — insufficient to keep eval argmax alive. `moe_liveness` / `h_norm` was **watch-only** on theme_restore.
6. E0–E3 “physical scale tiers” framing is **aspirational** at this QUALIFIED ckpt (only 2 tiers fire at eval).

## 3. Sealed gates (eval mode, probe panel)

On seal, run `system.eval()` over Stage-A probe (same panel as geometry hygiene):

| Gate | Rule |
|------|------|
| `moe_eval_min_load` | min_e mean_frac(nodes→e) ≥ **0.10** |
| `moe_eval_n_alive` | #{e : mean_frac ≥ 0.05} = **4** |
| `moe_eval_entropy_norm` | H(mean_load)/log(4) ≥ **0.85** |
| Existing geometry HOLD | unchanged (pure_hyp, sat, spread, finite, equiv) — AND with MoE gates |

Train-step Gumbel loads are logged but **do not** satisfy these gates.

## 4. Training change (standard, not exploratory)

Before next sealed train:

1. Add Switch-style aux on **softmax(logits)** soft probs:  
   `loss_lb = E * Σ_e (f_e * P_e)` with `f_e = mean soft assignment`, `P_e = mean soft gate mass` (or equivalent CV/quota on soft probs).  
2. Keep existing hard STE path + hard quota/CV as secondary.  
3. Log both `moe_load_e*_train_gumbel` and `moe_load_e*_eval_argmax` each epoch (probe eval pass).  
4. Do **not** patch geoopt_restore Fail-θ / re-QUALIFY without this card.

## 5. Non-claims

- Not spread_hold; not joint-gate selection first.  
- Not claiming soft MoE replaces hard commitment — hard STE remains train default; eval is argmax.  
- Not rewriting prior QUALIFIED stamps; annotate regression in docs.

## 6. Framing audit

Until eval utilization Pass: SBIR/patent/operator language must say E0–E3 tiers are **design intent**, not observed behavior on sealed QUALIFIED restore.


## 7. Retired metric (do not cite)

**`moe_load_e*` / `moe_load_min` logged during train under Gumbel-Softmax STE are RETIRED as evidence of deployed/eval routing.**  
Gumbel temperature noise pushes measured hard loads toward uniform regardless of peaked clean logits. This is structural, not a one-off miss — reproduced on geoopt_restore, cold_boot, correct_start. Never cite train-step `moe_load_*` or train-derived `h_norm` as proof of eval utilization, in history or going forward. Only **eval-argmax** loads (and soft pre-Gumbel probs for training LB) count.

## 8. Switch LB load-bearing detail

`switch_lb_loss` / `soft_quota_loss` use `F.softmax(logits)` on **raw pre-Gumbel gate logits**. Not Gumbel samples, not hard STE rows. Annotated in `moe.py`.

## 9. Geometry QUALIFIED does NOT stand independent (verified 2026-09-16)

Sealed `probe_sat` / `radius_spread` / `Δ_equiv` all consume `out["z_hyp"]`, and `model.py` sets `z_hyp = z_moe` (post-MoE).  
On geoopt_restore best ep19: mean per-structure `std(||z||)` — lift ≈0.0017, attn ≈0.0024, **moe ≈0.093**.  
Pre-MoE embeddings are a near-degenerate thin shell; sealed spread HOLD was dominated by MoE’s two parked radii. Do not treat geometry QUALIFIED as MoE-orthogonal.


## 10. Smoke trains (before sealed cold) — 2026-09-16

**Smoke A (Switch on T=1 softmax only, 3 ep, cold spine):** eval stayed collapsed (ep2 E2=1.000). `switch_lb` sat at ~1.01 (uniform min). Train Gumbel loads ~0.25. T=1 mean-softmax **cannot see** tiny-logit argmax bias — same masking class as Gumbel `moe_load_*`.

**Smoke B (add `softmax(logits / T)` T=0.05 proxy LB + quota, 2 ep, cold spine):**  
init eval `[0, 0.39, 0.61, 0]` Fail → ep1 **`[0.265, 0.378, 0.226, 0.131]` Pass** (min 0.131, 4 alive). `proxy_lb` 2.35 → 1.52 (working).  
Not a seal. Shows the load-bearing term is the **sharp clean-logit** proxy, not T=1 Switch. Sealed retrain only with this term active.
