# MLflow assistant notes: verify_grad_flow_decoupled runs (2026-09-25)

Everything below was read from the MLflow server (experiment 17). Nothing here comes from the repo code, which the assistant container cannot see.

## Runs

| Run | Name | mlflow.source.git.commit | Logging |
|---|---|---|---|
| 325ed1a2e04a4db7ae5615261e6da71a | verify_grad_flow_decoupled_r2_fold0_1MBNA | addbcab0aa2107a13de408a3b551f0613ee77f47 | 13 metrics, every 10th step (11 points) |
| 9352afc6cc7d4afd8fa9377d8576d57f | verify_grad_flow_decoupled_r2_fold0_1MBNA | 158491f128f3904218a61b59e6fbf0953a0ddf50 | 20 metrics, all 100 steps |

- Both: fold 0 (hold 1MBN:A), seed 0, 100 steps, all_gates=PASS, tags diagnostic=true and do_not_promote=true.
- Logged params are identical between the two runs. mech_lr_scale is not a logged param.
- Trajectories agree at every shared logged step to a max relative difference of about 5e-5 (loss_total about 2e-6). Close, not bit-identical; consistent with same seed plus GPU nondeterminism. Whether the two commits differ in the training path was NOT verified.
- Neither run records a dirty flag or diff. 325ed1a2 ran from a tree that did not match its recorded commit.

## Final values (9352afc6, step 99)

grad_l2_mechanism_head 0.1295, grad_l2_sdrp_head 0.0212, grad_l2_attn_layers 0.00211, grad_l2_moe 2.61e-4, grad_l2_attn_value_path 2.35e-4, loss_total 0.6021, preclip_norm 0.398.

## Clip bursts (9352afc6, dense history)

- clip_active: 27 of 100 steps. Clipped steps: 0-2, 44-50, 52-57, 62-64, 66, 68, 76, 77, 80, 85, 89, 94.
- From step 44 on: 24 of 56 clipped.
- Max preclip_norm 8.31 at step 49.
- explore_epsilon reaches 0 at step 13, so it does not line up with the bursts.
- loss_dehydron_bce: 0.60 at steps 43-44, then 0.77 / 0.79 / 0.67 / 0.71 / 0.77 at steps 45-49.
- loss_sdrp_ce ranges 0.364-1.481 over the run; its behaviour on the burst steps was not checked.
- Hypothesis only (untested): dual-input mechanism head (z_hyp + h_euc) is less stable as training progresses.

## Arm 1: dehydron_coeff 0.3 (run 98daaf50994347d0acde10701d1ba456, commit 46b6280, git_dirty=False)

Same seed/fold/steps as baseline 9352afc6; only dehydron_coeff differs (1.0 -> 0.3). All gates PASS.

| | baseline c=1.0 | arm c=0.3 |
|---|---|---|
| clipped steps | 27/100 | 4/100 (1, 51, 67, 68) |
| clipped from step 44 | 24/56 | 3/56 |
| max preclip_norm | 8.31 @ 49 | 1.68 @ 67 |
| mechanism-head grad min / final | 2.2e-2 / 0.129 | 8.3e-3 / 0.052 |
| mech/sdrp grad ratio median / max | 6.1 / 40.6 | 1.6 / 9.8 |
| loss_dehydron_bce @ step 99 | 0.566 | 0.580 |

- Burst window (~steps 50-70) persists at lower amplitude, so timing is not set by the coefficient alone.
- Fewer clips are partly mechanical (BCE gradient scales with the coefficient). loss_total is not comparable across arms.
- Single seed, fold 0. Does not test the dual-input mechanism-head hypothesis.

## Arm 2: mech_lr_scale 0.3 (run 726418c53bb547898f80b67ad4f9739e, commit 46b6280, git_dirty=False)

`--mech-lr-scale 0.3 --tag _mechlr0.3`, dehydron_coeff 1.0, same seed/fold/steps. All gates PASS.

| | baseline | arm1 c=0.3 | arm2 mechlr=0.3 |
|---|---|---|---|
| clipped steps | 27/100 | 4/100 | 38/100 |
| clipped from step 44 | 24/56 | 3/56 | 28/56 |
| first burst | step 44 | step 50 | step 37 |
| max preclip_norm | 8.31 @ 49 | 1.68 @ 67 | 4.03 @ 68 |
| mechanism-head grad min / final | 2.2e-2 / 0.129 | 8.3e-3 / 0.052 | 3.6e-2 / 0.712 |
| mech/sdrp grad ratio median / max | 6.1 / 40.6 | 1.6 / 9.8 | 8.1 / 40.9 |
| loss_dehydron_bce @ step 99 | 0.566 | 0.580 | 0.565 |

- Lowering the mechanism-head LR made clipping worse (more clipped steps, earlier and more sustained bursts, larger final mechanism-head gradient) with the same final BCE. The head's step size is not what drives the bursts.
- Loss weighting (arm 1) is the lever that changed burst amplitude; the ~steps 44-70 window persisted in every arm.
- Single seed, fold 0. The dual-input mechanism-head hypothesis is still untested.

## Curriculum schedules vs the burst window (from logged metrics, baseline/arm1/arm2 identical)

- tau_ceiling: linear ramp 0.700 -> 0.992 at +0.00295/step; no kink between steps 44 and 70 (0.830 -> 0.909).
- gumbel_temperature reaches its 0.3 floor by step 13 and stays there; explore_epsilon is 0 from step 13.
- No discrete stage transition or threshold crossing in the schedules coincides with steps ~44-70. Open: whether the growing tau_ceiling admits a batch of edges (data-driven "when"); not checkable from logged metrics. Would need per-step admitted-edge counts.
- Mechanism-head grad (every 10 steps): baseline 0.78, 0.35, 0.10, 0.04, 0.16, 1.35, 0.38, 0.08, 0.38, 0.28, 0.13; arm2 0.78, 0.35, 0.18, 0.13, 0.62, 0.58, 0.41, 0.72, 0.35, 0.19, 0.71. The slower head decays less and rebounds earlier; consistent with lagging a moving target, not confirmed.

## Storage (checked 2026-09-25)

- /workspace (the repo), /workspace/mlflow-artifacts and /app/mlflow-artifacts are all on the same device (/dev/nvme0n1p2). mlflow-artifacts is 37 GB and the device had 6.0 GB free.
- Same disk as the repo, gitignored: not backed up by git and not on a separate volume. Not lost, but at risk if the disk fails or fills.
- REFINED (per-experiment sizes via the MLflow API, 344 runs, 38.7 GB total): 38.4 GB is exp 13 `hermes/llm-lora` (25 runs, ~2 GB each, unrelated to tokyo-eye). All tokyo-eye experiments together are ~0.3 GB (exp 17: <0.01 GB, exp 11: 125 runs 0.12 GB, exp 14: 0.18 GB). Pruning tokyo-eye artifacts frees nothing.
- The device is 467 GB with 437 GB used (99%, 6.8 GB free), so the pressure is host-wide, not from this project's artifacts. Repo outside the MLflow stores: ~3.5 GB (pdb_cache 2.2, checkpoints 2.2, .git 0.38).
- 12-fold runner (wrap1_zhyp_m2_pool_decoupled.py) writes per-fold JSON results and a stamp (write_text) plus MLflow metrics; no torch.save in that script. Not verified: whether imported helpers save checkpoints to out_dir.
- 2026-09-25: at the user's request, the 16 `hermes/llm-lora` (exp 13) runs NOT referenced by any registered model version were soft-deleted via the MLflow API (IDs in experiments/hermes_deleted_run_ids.txt; restorable with `mlflow runs restore`). The 9 runs backing all 9 versions of `rswan-llm-lora` were kept (18.1 GB). User states all artifacts are backed up.
- Space is NOT freed yet (still 6.8 GB free): the server's artifact DELETE endpoint (DELETE /api/2.0/mlflow-artifacts/artifacts/...) hangs, even for a single small file, so artifacts could not be purged via the API. Freeing ~20.3 GB needs `mlflow gc` for those 16 run IDs on the server host (user action).

## Seed 1 replication (commit 46b6280, git_dirty=False, all gates PASS)

Runs: s1 baseline f30b8567dda049799cd1efd4cc0b94a2 (`_seed1_base`), s1 dehydron_coeff 0.3 809ed49de1e84a1b99d44e91e0326bfb (`_seed1_dcoef0.3`). Fold 0, 100 steps.

| | s0 base | s0 c=0.3 | s1 base | s1 c=0.3 |
|---|---|---|---|---|
| clipped steps | 27 | 4 | 45 | 5 |
| clipped from step 44 | 24 | 3 | 38 | 5 |
| first burst (step > 2) | 44 | 51 | 38 | 50 |
| max preclip_norm | 8.31 @ 49 | 1.68 @ 67 | 4.99 @ 44 | 3.10 @ 52 |
| mech/sdrp grad ratio median | 6.1 | 1.6 | 5.6 | 1.4 |
| loss_dehydron_bce @ 99 (train) | 0.566 | 0.580 | 0.549 | 0.569 |

- Baseline burst pattern replicates on seed 1 (heavier: 45% clipped, window ~37-72). Not seed-specific.
- c=0.3 damping replicates (27-45% -> 4-5% clipped), but the amplitude cut differs by seed (8.31 -> 1.68 vs 4.99 -> 3.10); a burst around steps 49-69 remains in both seeds.
- BCE cost of c=0.3: +0.014 (s0), +0.020 (s1) at step 99, on the training objective; gap at step 50 is +0.032 in s1. 100-step horizon only.
- The rationale draft "max norm >8.0 ... <5% clipped" does not hold across seeds (s1 baseline max 4.99; s1 c=0.3 is 5/100).

## Provenance findings on earlier runs (2026-09-25)

- 325ed1a2 started 00:32:49 UTC; 79b5b58 (decoupled architecture) was committed 01:37:15, so it ran from an uncommitted tree under HEAD addbcab. Its artifact's model_summary equals 9352afc6's (decoupled wiring), so it DID run the decoupled architecture; earlier note calling it "older code" was wrong on that point. Sparse logging and the missing per_step data are real. Exact tree bytes not recoverable from git/MLflow.
- 9352afc6 started 8 s after 158491f was committed: it ran committed code.
- Isolation run 0afb43f2 records 79b5b58 (has the dual-input wiring), all_checks PASS, bce_only z_over_h_l2 = 1.1012 (the ratio is 1.101, not 1.087). It also ran an uncommitted copy of verify_mechanism_grad_isolation.py (script first committed in 158491f, 10 min later).
- Card corrections for these are appended (uncommitted) in the prereg card's `corrections` field.

## 400-step baseline (run 719b9961ef234eb98fc1ae29bad53d4e, seed 0, fold 0, dehydron_coeff 1.0, commit e289916, git_dirty=False)

- MLflow tag all_gates = FAIL: gate_grad_l2_attn_layers and gate_grad_l2_moe fail; the other six gates PASS. The gates are "min over all steps > threshold", written for the 100-step run.
  - attn_layers: min 8.53e-4 @ step 376 (thr 1e-3); only 2 steps at/below thr (350, 376); final 2.48e-3.
  - moe: min 5.58e-5 @ step 383 (thr 1e-4); 19 steps at/below thr (first 306, last 396); final 1.47e-4.
  - Transient dips while training loss is tiny, not a dead gradient path (mechanism 8.9e-3 min, sdrp 3.7e-3 min, value-path 4.8e-5 min all > thr). Whether these gates are meaningful at 400 steps is a gate-design question for the operator.
- Training loss collapses: loss_dehydron_bce 0.698 (0), 0.554 (100), 0.143 (200), 0.012 (300), 0.018 (399). Per-structure BCE at 399: min 2.7e-4, max 0.106. This is the training objective on 11 training structures; it looks like near-memorization, and says nothing about held-out performance. So the "BCE cost of c=0.3" is a weak measure at this horizon.
- Clipping: 97/400 steps (24%), max preclip_norm 8.44 @ step 49. Blocks of 50 steps: 8, 21, 8, 7, 33, 10, 6, 4. First bursts again at steps 45-57 and 62-68, then a near-continuous run at steps 207-255 (per-structure BCE max 1.15 @ step 250 while the min was 0.005).
- Key finding: the early burst window sits at the SAME STEPS (~45-70) as in the 100-step runs even though tau_ceiling is stretched 4x (tau at step 50 = 0.737 here vs 0.847 in the 100-step runs). So the timing is NOT set by tau_ceiling (it is reproducible at fixed seed and coefficient; see the c=0.3 section below for the corrected reading). The gumbel/epsilon schedules are step-based (floor at step 13) and were not the coincident cause; other step-tied candidates (optimizer state) not tested.
- A ~36 min stall between steps 384 and 385 (no other gap > 34 s); host pause suspected, not confirmed. Does not affect results.
- Memorization check on the 400-step baseline (data only; no held-out metric exists in this diagnostic, so overfitting is consistent-with, not established):
  - Per-structure BCE max was > 0.5 on 214 of 400 steps but only 13 steps after step 260; mean BCE is ~0.01-0.03 from step ~300. Clipping: 77/260 steps (30%) before step 260 vs 20/140 (14%) after. Gate dips (attn 2 steps, moe 19 steps) all occur after step 300, in the near-fit regime.
  - Clip windows vs single-structure spikes: corr(clip_active, per-structure BCE max) = +0.23 (400-step) and +0.31 (100-step); clipped-step median BCE max 0.611 vs 0.517 (400-step), 0.708 vs 0.712 (100-step). Weak-to-moderate; does not show that clipping = one hard structure spiking. The 1.147 at step 250 is a single step; the 207-255 window has BCE max 0.15-0.63. Only max/min are logged, not which structure.
  - The early window (steps ~45-70) happens at loss 0.55-0.7, long before any fit, so memorization does not explain it.
- 22da0d3 (user commit) message mentions notes/card updates but the diff is only the 7 previously untracked data files. Nothing lost: HEAD's card already has the 4 corrections (ded6cae), no stash/other branch, reflog linear. Message is simply inaccurate; it is already on origin.

## 400-step dehydron_coeff 0.3 (run d531978982f643b39a265950b92cbb81, seed 0, fold 0, commit 22da0d3, git_dirty=False; 212 min)

| | 400-step baseline c=1.0 | 400-step c=0.3 |
|---|---|---|
| clipped steps | 97/400 (24%) | 7/400 (1.75%): 1, 51, 63, 67, 68, 203, 321 |
| max preclip_norm | 8.44 @ 49 | 1.56 @ 67 |
| steps with preclip > 0.5 | 263 | 49 |
| all_gates | FAIL (attn_layers, moe) | FAIL (attn_layers, moe) |
| attn_layers min / steps <= 1e-3 | 8.5e-4 @376 / 2 | 6.0e-4 @370 / 17 |
| moe min / steps <= 1e-4 | 5.6e-5 @383 / 19 | 5.2e-5 @382 / 42 |
| mean BCE @ 100 / 150 / 200 / 300 | 0.554 / 0.424 / 0.143 / 0.012 | 0.554 / 0.485 / 0.236 / 0.016 |
| mean BCE avg over steps 350-399 | 0.0241 | 0.0238 |
| steps with per-structure BCE max > 0.5 | 214 | 228 |

- c=0.3 cut clipping ~14x and peak preclip ~5x, with no difference in end-of-run training BCE (transient lag mid-run, steps ~150-250). Training objective only; no held-out metric.
- The two failing gates fail MORE at c=0.3 (17 and 42 steps below thr vs 2 and 19): gate rule (min over all steps) is not calibrated for 400 steps.
- The early window (preclip > 0.5 at steps 50-72) persists at c=0.3.
- Burst STEP INDICES replicate across horizons at fixed seed AND fixed coefficient: baseline peak @ step 49 in both the 100-step (8.31) and 400-step (8.44) runs; c=0.3 peak @ step 67 in both (1.68, 1.56), although tau_ceiling differs 4x. Seed 1 puts them elsewhere (baseline 4.99 @ 44, c=0.3 3.10 @ 52).
- CORRECTION (2026-09-26): an earlier version of this note and the chat summary said the timing was "not set by ... coefficient". That was wrong: the peak moved from step 49 (c=1.0) to step 67 (c=0.3) at the same seed. Timing is a function of (seed, coefficient) and is insensitive to the tau schedule.
- Cross-run check on preclip_norm, steps 3-99 (all first-100-step windows): same seed + coefficient, 100-step vs 400-step run: Spearman 0.96 (c=1.0) and 0.99 (c=0.3), top-10% steps overlap 10/10 (chance ~1). Same seed, c=1.0 vs c=0.3: Spearman ~0.59, top-10% overlap 1/10. Seed 0 vs seed 1: overlap 1/10 (c=1.0), 0/10 (c=0.3). At the other config's peak step there is no hidden smaller burst: step 49 in the c=0.3 run is 1.7x its median (400-step: 1.8x), step 67 in the c=1.0 run is 1.8x (1.7x). So the bursts are NOT one shared per-step trigger seen at different amplitudes.
- Reading: the training trajectory is highly reproducible for a given seed and coefficient (a 100-step run predicts the first 100 steps of a 400-step run), and the instability events emerge from that trajectory; where they land depends on the loss weighting and the seed. An RNG-mask trigger (dropout/drop-path/Gumbel) is NOT supported by this and not excluded (same masks at the same step could still only matter in some weight states). No direct test done.


## Held-out comparison plan (recorded 2026-09-26, BEFORE any held-out result exists)

Question (only): does dehydron_coeff 0.3 cost anything in held-out performance vs 1.0? The clip-rate reduction is treated as settled (replicated over 2 seeds x 2 horizons).

- Tool: `scripts/wrap1_zhyp_m2_pool_decoupled_diag.py` (new; wraps the sealed runner by rebinding DEHYDRON_COEFF, RESULT_DIR and CANONICAL_MLFLOW_EXPERIMENT, so the sealed script/hash, its resume-from-disk results folder and its MLflow experiment are untouched; refuses all-12-fold runs; a subset never stamps the card). Smoke-tested: overrides confirmed at call time.
- Results: `data/gates/diag_heldout_dcoef<c>/`; MLflow experiments `diag/heldout-dcoef1` and `diag/heldout-dcoef0.3`. Per-fold metrics of interest: heldout_macro_f1, train_macro_f1, train_min_f1, clip_active_fraction, c_drift.
- Step 1: fold `1MBN:A` (153 res) at coefficient 1.0, then 0.3; seed 0; 400 steps (~3.5 h each).
- Pre-committed next folds if fold 0 is ambiguous (chosen now, independent of fold-0 results): `1TEN:A` (89 res, 2nd smallest) and `1BG1:A` (558 res, largest). Fold sizes (residues): 1UBQ 76, 1TEN 89, 1HHP 99, 1LYZ 129, 1MBN 153, 4OBE 169, 1TIM 247, 1F88 338, 2SHP 491, 1IVO 511, 2Z6H 533, 1BG1 558.
- Pre-registered read (proposed by the operator 2026-09-26, recorded before any held-out number exists; revisable only until the first held-out result is read): |difference in heldout macro-F1 between c=1.0 and c=0.3| > 0.06 = "meaningful" (interim; revised 2026-09-26 by the operator from ~0.04-0.05, still before any held-out number); smaller = "ambiguous", pending multi-fold data. If the fold-0 difference is inside the band, the conditional next step is a second seed at fold 0 to measure seed-to-seed noise; if clearly outside, act without it. Anchor and its limits (checked against the result files):
  - The 0.307-0.349 (spread 0.042) figure comes from `tokyo_eye_equ_wrap1_zhyp_m2_result.json` (status INCONCLUSIVE_UNDERFIT, dehydron_coeff 0.0, train macro-F1 ~0.43, i.e. every fold sat near the floor). The pool-frontend card `..._m2_pool_lr_1e-4_result.json` (FAIL_NO_SIGNAL, dehydron_coeff 0.0, train macro-F1 ~0.96-0.98) spans 0.324-0.624 (spread 0.300) across the same 12 folds. Between-fold spread mostly reflects structure difficulty, and for an underfit model it is compressed.
  - The noise that matters for a same-fold paired comparison is seed-to-seed variance on that fold; it has not been measured. A second seed at fold 0 (2 more runs, ~7 h) would measure it directly.
  - How to read the band (operator, 2026-09-26): 0.06 is a soft upper bound for "definitely not noise", not a sharp dividing line. A fold-0 difference inside the band is neither settled-meaningful nor settled-noise; it triggers the second-seed run. A difference clearly outside it can be acted on without that run.
  - Where 0.06 comes from: 0.380 (pool bbtrain) - 0.319 (earlier M2 card) = 0.062 on fold 1MBN:A. That band is dominated by the underfit M2 card (train macro-F1 0.434, cold SE(3)-lite frontend, lr_frontend 1e-5); between the two fitted pool cards (0.374 vs 0.380) the difference is 0.006. So 0.06 is a conservative interim bar, not a noise floor.
  - The three reference rows are NOT one-dial variants, but not because "coefficient 0.0" means different things: all three have the same loss config (sdrp_coeff 0.1, dehydron_coeff 0.0, margin_coeff 0.0, moe_mode ablated), i.e. SDRP-only with the mechanism-head BCE off. They differ in frontend: zhyp_m2 = cold SE(3)-lite, lr_frontend 1e-5; pool lr_1e-4 = equiformer_v3_pool_cold_init, lr 1e-4; pool lr_1e-4_bbtrain = same plus backbone_train_mode. All three also predate the decoupled architecture (R2 decoupling, value-path attention, [z_hyp, h_euc] mechanism head), so the new runs differ from every reference row in architecture and loss, not just in the dehydron coefficient.
  - Reference points, fold 1MBN:A, seed 0, 400 steps, all dehydron_coeff 0.0: zhyp_m2 heldout 0.319 (train 0.434); pool lr_1e-4 heldout 0.374 (train 0.960); pool lr_1e-4_bbtrain heldout 0.380 (train 0.984). Two configs differ by 0.006 here, but they are different configs, not a noise estimate.
- One fold cannot justify the amendment either way; the amendment needs the extra folds' data.
- Open precondition for the sealed 400-step launch (independent of this comparison): the verify_grad_flow gates (min over all steps) fail transiently at 400 steps; a phase-aware rule is owed.

## Fold-0 held-out results (diag wrapper; seed 0, fold 1MBN:A, 400 steps; commit 4f18b08, git_dirty=False for both; ~212 min each)

MLflow: `diag/heldout-dcoef1` run 02f77ba040884f76b02005ee1e67fe74; `diag/heldout-dcoef0.3` run a38122c7... Results: `data/gates/diag_heldout_dcoef{1,0.3}/decoupled/seed0_hold_1MBNA.json` (untracked).

| | c=1.0 | c=0.3 |
|---|---|---|
| heldout macro-F1 | 0.3684 | 0.3185 (0.31854379977246877) |
| heldout sdrp_top1_acc (majority rate 0.9281) | 0.902 | 0.915 |
| heldout lift (top1 / majority) | 0.9718 | 0.9859 |
| train macro-F1 / min-F1 | 0.888 / 0.647 | 0.983 / 0.926 |
| clip_active_fraction (all steps) | 0.317 (0.318) | 0.000 (0.018) |
| g_fit_train_pass_fold, G_grad_spine | True, True | True, True |

- Difference in held-out macro-F1 (c=1.0 minus c=0.3) = +0.0499: INSIDE the 0.06 band, so per the pre-registered read it is neither settled-meaningful nor settled-noise and triggers the second-seed run. (Under the original ~0.04-0.05 wording it would have sat on the line.)
- c=0.3's held-out macro-F1 is bit-identical (17 digits) to the earlier underfit M2 card's fold-0 value 0.31854379977246877, i.e. it sits exactly at the floor-like score for this fold. c=1.0 is +0.05 above that floor. Both have lift < 1: neither beats the majority-rate top-1 baseline on held-out. Held-out signal on this fold is essentially absent for both (as in the pool cards, FAIL_NO_SIGNAL; their fold-0 values were 0.374 and 0.380 at coefficient 0).
- c=0.3 fits the training set better (train macro-F1 0.983 vs 0.888, min-F1 0.926 vs 0.647) with no clipping; c=1.0 with ~32% of steps clipped fits worse. The held-out score is no better, so the train/held-out gap is wider at 0.3.
- Power caveat for the pre-committed next folds: in the earlier pool lr_1e-4 card, `1TEN:A` (0.324) and `1BG1:A` (0.344) were near the floor, while `1UBQ:A` (0.624), `1HHP:A` (0.514) and `4OBE:A` (0.441) were the highest. The pre-committed folds may therefore have little power to separate the coefficients. This comes from earlier-card data, not from the fold-0 result, but any change to the pre-committed folds is an amendment for the operator to decide and record with this rationale.

## Open items

1. DONE (46b6280): git provenance guard in both runners.
2. DONE: both single-variable arms ran (see above).
3. Decide the 12-fold config (dehydron_coeff 1.0 vs 0.3): pending the held-out comparison above. Changing the sealed runner's DEHYDRON_COEFF needs a documented prereg amendment (the diag wrapper avoids editing the sealed script).
4. Push local commits to origin (needs GitHub credentials on the host; the assistant container has none). Still owed: phase-aware gate rule for a 400-step sealed launch; per-structure BCE logging (with structure tags) if the hard-structure question matters.
