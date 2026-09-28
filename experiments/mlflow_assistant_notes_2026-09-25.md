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

- Tool: `scripts/wrap1_zhyp_m2_pool_decoupled_diag.py` (new; wraps the sealed runner by rebinding DEHYDRON_COEFF, RESULT_DIR and CANONICAL_MLFLOW_EXPERIMENT, so the sealed script/hash, its resume-from-disk results folder and its MLflow experiment are untouched; refuses all-12-fold runs; the card result path is redirected into the diagnostic results dir (see CORRECTION below; the original claim that a subset never stamps was wrong)). Smoke-tested: overrides confirmed at call time.
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

## Why fold-0's c=0.3 score equals the earlier underfit card to 17 digits (checked 2026-09-26)

- Not a cache/path bug. Held-out macro-F1 (`wrap1_zhyp_g_fit.py` ~lines 175-217) is the mean of per-class F1 over the classes PRESENT in y, computed from integer confusion counts of argmax(sdrp_logits). Fold `1MBN:A` has n=153 with class counts {majority 142, 8, 3}. The confusion (majority TP=140, FP=11, FN=2, both minority classes F1=0) gives macro-F1 = (280/293)/3 = 0.31854379977246870 vs logged 0.31854379977246877 (last-digit float ordering only), and top-1 = 140/153 = 0.91503 exactly as logged. The earlier underfit M2 card's fold-0 top-1 is also 0.91503 and its macro-F1 is the same number, so it has the same confusion counts. Same integer counts give the same float.
- Consequence: both c=0.3 and the underfit card are majority-collapsed on this fold (all 11 minority nodes predicted as the majority class; 2 majority nodes predicted elsewhere). This is inferred from the arithmetic; per-node predictions are not stored and no checkpoint was saved.
- The runner has no code path that reads another run's held-out result (it reads only the card, folds_frozen.json and its own fold file for resume), and c=0.3's held-out metrics are in its own MLflow run (diag/heldout-dcoef0.3) and its own results dir.
- Granularity: from that confusion, ONE correctly predicted minority node adds +0.075 macro-F1 (class of 8) or +0.168 (class of 3). The whole c=1.0 vs c=0.3 gap (+0.0499) and the 0.06 band are smaller than one node. c=1.0 has LOWER top-1 (138/153 vs 140/153). So fold 0 cannot separate the two settings; a second seed on fold 0 would again resolve only a handful of nodes.

Per-fold resolution (sdrp class counts from the frozen batches; "1 node" = macro-F1 change from one correct minority node starting from majority-collapse; last column = earlier pool lr_1e-4 held-out macro-F1, coefficient 0):

| fold | n | class counts | 1 node (smallest / largest minority class) | pool lr_1e-4 held-out |
|---|---|---|---|---|
| 1MBN:A | 153 | 142/8/3 | +0.167 / +0.074 | 0.374 |
| 1LYZ:A | 129 | 120/5/4 | +0.133 / +0.111 | 0.416 |
| 1BG1:A | 558 | 517/24/17 | +0.037 / +0.027 | 0.344 |
| 1F88:A | 338 | 310/20/8 | +0.074 / +0.032 | 0.430 |
| 2Z6H:A | 533 | 495/30/8 | +0.074 / +0.022 | 0.416 |
| 1HHP:A | 99 | 84/9/6 | +0.095 / +0.067 | 0.514 |
| 1TEN:A | 89 | 83/4/2 | +0.222 / +0.133 | 0.324 |
| 1UBQ:A | 76 | 67/8/1 | +0.333 / +0.074 | 0.624 |
| 1TIM:A | 247 | 229/15/3 | +0.167 / +0.042 | 0.382 |
| 4OBE:A | 169 | 154/10/5 | +0.111 / +0.061 | 0.441 |
| 1IVO:A | 511 | 467/26/18 | +0.035 / +0.025 | 0.433 |
| 2SHP:A | 491 | 470/13/8 | +0.074 / +0.048 | 0.426 |

- The pre-committed folds are a poor pair for this question: `1TEN:A` has the coarsest resolution (one node = +0.13 to +0.22) and a near-floor earlier score; `1BG1:A` has good resolution (+0.03) but an earlier score near the floor. Folds with both fine resolution and clearly above-floor earlier scores: `1IVO:A` (0.433), `2Z6H:A` (0.416), `1F88:A` (0.430). This rests on label counts and the earlier card, not on fold-0 results. Swapping the pre-committed folds is an amendment for the operator to decide and record.
- A single 0.06 band is not coherent across folds: on the large folds it is ~2 nodes, on the small folds less than one node. Consider judging differences in nodes or with an interval.

## LESSON: a fixed macro-F1 delta is not portable across folds (2026-09-26)

The 0.06 threshold was derived from a rough cross-config comparison on fold 0 and then treated as if it applied to whichever fold came next. The same numeric gap is a different amount of actual prediction change on every fold (one correctly predicted minority node is worth +0.13 to +0.22 on `1TEN:A`, +0.075 to +0.17 on `1MBN:A`, but only +0.02 to +0.04 on `1BG1:A`/`1IVO:A`/`2Z6H:A`). Same class of mistake as reusing a bar (e.g. an earlier lift threshold) built for one label geometry on another. Standing rule: express any held-out difference threshold in node-level flips computed from the fold's own class counts, and check a fold's resolving power BEFORE spending compute on it.

## AMENDMENT to the held-out plan (2026-09-26, before any 1IVO/2Z6H result)

- Second-seed-at-fold-0 is dropped (fold 0 cannot resolve the comparison: see the granularity table). The pre-committed folds `1TEN:A` (coarsest resolution) and `1BG1:A` (near-floor earlier score) are replaced. Rationale rests on label counts and the earlier pool card, not on fold-0's outcome.
- New plan: `1IVO:A` (n=511, class counts 467/26/18) at coefficient 1.0 then 0.3 (seed 0, 400 steps); add `2Z6H:A` (n=533, 495/30/8) if the 1IVO result is "close" (below).
- Threshold in node units (proposed following the operator's suggestion; computed from each fold's class counts, starting from majority collapse, ignoring the small majority-F1 change). T1 = macro-F1 gain from ONE extra correct node in the fold's largest minority class; T3 = from THREE. `1IVO:A`: T1 ~ 0.025, T3 ~ 0.069 (largest minority class of 26; the class of 18 gives 0.035 / 0.095). `2Z6H:A`: T1 ~ 0.022, T3 ~ 0.061 (class of 30; the class of 8 gives 0.074 / 0.182).
- Reading of gap = |heldout macro-F1(c=1.0) - heldout macro-F1(c=0.3)| on a fold: gap >= T3 = meaningful on that fold; T1 <= gap < T3 = close, run `2Z6H:A`; gap < T1 = indistinguishable at this fold's resolution. Also report per-class recall and the number of held-out nodes whose predictions differ (confusion matrices and per-node predictions are now saved).
- CONFIRMED by the operator 2026-09-26, before any 1IVO/2Z6H result: the node-level reading above (>= 3 nodes meaningful; 1-3 nodes close -> run 2Z6H:A; < 1 node indistinguishable) is final. Decision vs description boundary: the conclusion rests ONLY on that node-level reading. From the saved confusion matrices and per-node predictions the report will state, as description only: each run's confusion matrix, per-class recall, whether either run is majority-collapsed, and the number of held-out nodes whose predictions differ between the two runs. Patterns in WHICH nodes flip (e.g. a structural motif) are not used to reach or adjust the conclusion; if pursued they are a separate, labeled exploratory analysis done afterward.
- Selection-bias statement for any write-up: folds were selected for resolving power and known above-floor earlier performance; a result here shows detectability where it is highest, NOT that the coefficient effect holds across the 12-fold corpus.
- Wrapper change: `wrap1_zhyp_m2_pool_decoupled_diag.py` now saves `confusions_seed<S>_<fold>.json` (confusion matrix + per-node y/pred for every scored structure, the last record is the held-out fold) in the results dir; logger is exception-safe; smoke-tested (12 records, recomputed macro-F1 matches the runner's to ~1e-16).

## CORRECTION: the diagnostic wrapper DID stamp the sealed card's result path (found 2026-09-26)

- Earlier statements in these notes, the wrapper docstring and the chat ("a subset of folds never stamps the card") were wrong. The sealed runner only skips scoring when FEWER folds finished than were REQUESTED; a single-fold request (`--folds 1MBN:A`) scores and writes `data/gates/tokyo_eye_equ_wrap1_zhyp_m2_pool_decoupled_result.json` (gate_id ...decoupled_result, signed "auto from wrap1_zhyp_m2_pool_lr.py").
- What happened: the fold-0 coefficient-0.3 diagnostic wrote that file (status FAIL_NO_SIGNAL, folds ['1MBN:A'], dehydron_coeff 0.3, mean macro-F1 0.31854, git_commit pin 158491f, script/prereg sha256s). It was committed and pushed in `aaad2b7`. It is NOT a sealed 12-fold result; it should not exist until a real 12-fold run completes. No repo code or tool other than the runner references the file (checked by grep), and the real run would overwrite it.
- Fix (commit after this note): the wrapper now redirects `card.result` to `data/gates/diag_heldout_dcoef<c>/NOT_A_CARD_RESULT_dcoef<c>.json` (CardPaths is a frozen dataclass; replaced via dataclasses.replace). Verified: unit check of the redirect, and the startup banner prints `card.result redirected: ...decoupled_result.json -> ...NOT_A_CARD_RESULT_...`.
- Second channel checked (MLflow tags): "signed: auto from wrap1_zhyp_m2_pool_lr.py" is just a literal string in the runner (also in three sibling runners); nothing keys off it. No repo code queries runs by the `card`/`gate_id` tags (the coordinator's lifecycle router queries by experiment name; the diag runs live in `diag/heldout-*`). But the sealed runner tags every run `diagnostic=false`, `card=m2_pool_<arm>`, `gate_id=..._prereg`, so the diag runs were mislabeled as card runs by tag. Fixed in the wrapper (commit 1726fbd): `diagnostic=true`, `card=DIAG_...`, `gate_id=NOT_A_GATE_diag_of_...`, `not_a_card_result=true`, `diag_wrapper=...` (verified end to end with a throwaway MLflow run). The three existing diag runs (02f77ba0 and a38122c7 for 1MBN:A, b3664684 for 1IVO:A) were corrected via the API and carry `tags_corrected_after_run`.
- Cleanup plan (operator-approved approach: delete-and-recommit, no history rewrite): after the 1IVO c=1.0 run finishes and re-stamps the sealed path once more, `git rm` data/gates/tokyo_eye_equ_wrap1_zhyp_m2_pool_decoupled_result.json in a new commit that references 71f9308; commit only after the c=0.3 run has recorded its start commit, so both runs keep clean provenance.
- Still open: (1) the `1IVO:A` c=1.0 run started BEFORE the fix and will overwrite the sealed-path stamp with a `1IVO:A` stamp when it finishes; (2) the committed stamp should be removed from the sealed path (its information is already in the per-fold JSONs in diag_heldout_*/decoupled/). Operator decision.

## 1IVO:A held-out results (seed 0, 400 steps; c=1.0 commit 00cf21f, c=0.3 commit 1726fbd, both git_dirty=False, ~190 min each; code identical between the two commits, diff is the redirect/tag-rewrite fixes and docs only)

| | c=1.0 (b3664684) | c=0.3 (ba58175a) |
|---|---|---|
| heldout macro-F1 | 0.3981 | 0.4663 |
| heldout top1 / majority rate | 0.8885 / 0.9139 | 0.8924 / 0.9139 |
| train macro-F1 / min-F1 | 0.897 / 0.742 | 0.982 / 0.921 |
| recall: class0 (n=26) / class1 (n=18) / majority (n=467) | 0.154 / 0.056 / 0.961 | 0.231 / 0.167 / 0.957 |
| held-out nodes whose prediction differs between the two runs | 45 / 511 | |  <!-- derived from the 1IVO:A c=0.3 files now at data/gates/diag_heldout_dcoef0.3/prior_run_1IVOA_ba58175a/ (moved 2026-09-28); the 45 has never had a fixed-seed nondeterminism floor: see the replicate section below -->

- Direction REVERSED from fold 0: here c=0.3 has the higher held-out macro-F1 (gap 0.0682, c=0.3 minus c=1.0). Neither is majority-collapsed (both have nonzero recall on both minority classes; recomputed macro-F1 matches logged to ~1e-16 for both, so the metric and confusion-matrix reasoning check out).
- Node-level reading (thresholds computed from 1IVO:A's own class counts, pre-registered before this result): T1 (class of 26) ~0.025, T3 (class of 26) ~0.070; class of 18 gives T1~0.035, T3~0.095. Gap 0.0682 sits just under the class-26 T3 and clearly under the class-18 T3, while clearly above both T1s. Per the pre-registered rule this is CLOSE (1-3 nodes), not settled either way -> triggers `2Z6H:A`.
- Both runs pass g_fit_train (train macro-F1 > floor) and G_grad_spine; the sealed-card gates are not evaluated here (this is the diagnostic wrapper, gates belong to verify_grad_flow_decoupled.py).

## 2Z6H:A held-out comparison (launched per the pre-committed plan; 1IVO:A was "close")

## Decision rule for the 3-fold outcome (recorded 2026-09-26, BEFORE the 2Z6H:A result)

Signs so far, held-out macro-F1 (c=1.0 minus c=0.3): fold 0 (1MBN:A) = +0.050 (inside its own ambiguous band); 1IVO:A = -0.068 (close, 1-3 nodes). Two folds, two different signs, both in ambiguous/close territory, not in either fold's "meaningful" zone.

Pre-committed reading of 2Z6H:A, decided now rather than after seeing it:
- 2Z6H:A meaningful AND agrees in sign with 1IVO:A (i.e. c=0.3 higher, gap >= its own T3): treat as 2-of-3 evidence for a real, fold-dependent-in-sign-but-real effect worth naming as "unstable/context-dependent, not a clean win for either coefficient." Does not by itself justify amending the sealed coefficient to 0.3 on held-out grounds; the training-fit and stability case for 0.3 stands on its own.
- 2Z6H:A meaningful AND agrees with fold 0 (c=1.0 higher): same as above, mirrored -- 2-of-3 for the opposite sign. Same conclusion: no clean, fold-independent held-out story either way.
- 2Z6H:A ALSO lands ambiguous/close (either sign): three folds, no fold reaching its own "meaningful" bar, is treated as the answer, not as insufficient data -- i.e. AT N=3 folds, STOP concluding on held-out grounds. This is positive evidence of no fold-generalizable held-out effect at this sample size, not a call for a 4th fold. Any further fold would be additional description, not a different verdict, unless it lands clearly meaningful in a way that reopens the question.
- In every branch above: the training-fit result (0.3 improves train macro-F1 and min-F1 on every fold and seed run so far: s0/s1 100-step, 400-step baseline, 1IVO:A) and the clip-rate reduction (also consistent across every run so far) are UNCHANGED and are reported as a separate, higher-confidence claim from the held-out question. The amendment, if any, should state these as two distinct claims with their own confidence levels, not a single blended verdict.

## Gradient/held-out coupling check: plan and data availability (recorded 2026-09-26, BEFORE 2Z6H:A c=0.3 lands)

Question: does the coefficient's effect on gradient magnitude (e.g. final grad_l2_mechanism_head) predict its effect on held-out macro-F1 or on which nodes flip, across the folds run so far?

- Data available:
  - Aggregate (all 3 folds, both coefficients): grad_l2_{mechanism_head, sdrp_head, attn_layers, moe, projector, euc_skip, _log_c} are logged at every dense/sparse step for every run so far (fold0 02f77ba0/a38122c7, 1IVO b366468/ba58175a, 2Z6H 3cafaff8 + the running c=0.3). So (heldout macro-F1 delta) vs (final or min grad_l2 delta) is computable for all 3 folds.
  - Per-node flip count: only 2 folds (1IVO:A, 2Z6H:A) have confusion_true_rows_by_pred_cols/y/pred saved. Fold 0's pair (02f77ba0, a38122c7) ran BEFORE the confusion logger was added (commit 00cf21f); no confusions_seed0_1MBNA.json exists on disk for either. This was not caught until checked just now, while looking for it -- worth noting fold 0's per-node data cannot be reconstructed after the fact. (2026-09-28: the 1IVO:A c=0.3 copies of these files are now in `data/gates/diag_heldout_dcoef0.3/prior_run_1IVOA_ba58175a/`; the new interval run writes fresh `confusions_seed0_1IVOA.json` / `decoupled/seed0_hold_1IVOA.json` in the original locations.)
- Caveat to state alongside any number produced: n=3 folds (aggregate) or n=2 folds (per-node) is far too small for a correlation coefficient to be interpretable as a hypothesis test. Any number reported here is DESCRIPTIVE ONLY (e.g. "the sign was the same in 2 of 3 folds"), not evidence of a real coupling, and should not be used to argue for or against the coefficient amendment on its own.
- This check runs alongside the pre-registered 2Z6H:A decision rule (see above) once the c=0.3 run finishes, not as a separate follow-up.

## 2Z6H:A held-out results, and the 3-fold verdict (commit 6aaea27, both git_dirty=False)

Runs: c=1.0 3cafaff898f340a7b26d19cc44e0f5e1, c=0.3 1212cdf8f81d45caaf2630262996f4dd. Class counts 495/30/8 (n=533).

| | c=1.0 | c=0.3 |
|---|---|---|
| heldout macro-F1 | 0.4435 | 0.4444 |
| heldout lift | 1.004 | 1.010 |
| train macro-F1 / min-F1 | 0.903 / 0.715 | 0.987 / 0.939 |
| clip_active_fraction | 0.293 | 0.024 |
| recall class0(n=30)/class1(n=8)/majority(n=495) | 0.133/0.125/0.994 | 0.233/0.000/0.996 |
| held-out nodes whose prediction differs | 17 / 533 | |

- Gap (c1.0 - c0.3) = -0.0010. Node thresholds from this fold's own class counts: T1(class30)=0.022, T3(class30)=0.062, T1(class8)=0.074, T3(class8)=0.183. The gap (0.001) is far below even the smallest T1 -- i.e. INDISTINGUISHABLE at this fold's resolution, more clearly than merely "close."
- Recomputed macro-F1 matches logged to ~1e-16 for both runs; neither is majority-collapsed.

### Applying the pre-registered node-level reading to all 3 folds

| fold | gap (c1.0-c0.3) | smallest T1 | smallest T3 | verdict |
|---|---|---|---|---|
| 1MBN:A | +0.0499 | 0.075 (class n=3: 0.168; class n=8: 0.075) | -- | below T1: indistinguishable (more precise than the earlier "ambiguous") |
| 1IVO:A | -0.0682 | 0.025 | 0.070 (class n=26) | between T1 and T3: CLOSE |
| 2Z6H:A | -0.0010 | 0.022 | 0.062 | below T1: indistinguishable |

No fold reaches its own T3 ("meaningful"). Per the rule pre-registered before this result (see "Decision rule for the 3-fold outcome" above): this is the outcome where all 3 folds land ambiguous/close/indistinguishable -> STOP concluding on held-out grounds. This is treated as positive evidence of no fold-generalizable held-out effect at this sample size (seed 0, 3 of 12 folds), not as a reason to run a 4th fold.

### Coupling check (run now, per the pre-registered plan; both parts DESCRIPTIVE ONLY, not hypothesis tests)

Aggregate (n=3 folds), heldout macro-F1 delta vs final grad_l2_mechanism_head delta (both c1.0 - c0.3):
| fold | heldout delta | grad_l2_mechanism_head delta | same sign |
|---|---|---|---|
| 1MBN:A | +0.0499 | +0.0339 | yes |
| 1IVO:A | -0.0682 | +0.0235 | no |
| 2Z6H:A | -0.0010 | +0.0112 | no |

Sign agrees in 1 of 3 folds. grad_l2_mechanism_head delta is positive (c1.0 has the larger mechanism-head gradient, as expected since dehydron_coeff scales that loss term) in ALL 3 folds regardless of which coefficient has the higher held-out score -- i.e. the gradient-magnitude difference does not track the held-out-score difference at all in this data.

Per-node (n=2 folds with confusion data: 1IVO:A, 2Z6H:A), grad_l2_mechanism_head delta vs count of held-out nodes whose prediction differs:
| fold | grad delta | node flips |
|---|---|---|
| 1IVO:A | +0.0235 | 45/511 (8.8%) |
| 2Z6H:A | +0.0112 | 17/533 (3.2%) |

The two points are consistent in direction (larger grad delta, more flips), but n=2 cannot support any claim beyond "consistent with, not evidence for."

### Two separate claims (per the pre-registered structure)

1. Training fit and stability (HIGH CONFIDENCE, consistent on every fold and seed run: s0/s1 100-step, 400-step baseline, 1MBN:A, 1IVO:A, 2Z6H:A): dehydron_coeff=0.3 improves train macro-F1 and min-F1, and sharply reduces clip_active_fraction, vs 1.0, in every single comparison run so far.
2. Held-out generalization (NO DETECTED EFFECT at n=3 folds, seed 0): none of the 3 folds tested shows a held-out macro-F1 difference exceeding its own node-level resolution. Direction of the (unresolved) difference is not consistent across folds (+, -, - as tested). This is not evidence that 0.3 helps or hurts held-out performance; it is evidence that if an effect exists, it is smaller than this experiment's resolution at 3 folds / 1 seed.

Any amendment to the sealed 12-fold card's dehydron_coeff should state claim 1 (training stability) as its rationale, and state claim 2 honestly as "no held-out effect detected at this sample size," not as "held-out performance is unaffected" or "held-out performance improves."

## Interval-eval option for the diag wrapper: design and verification (2026-09-28)

`wrap1_zhyp_m2_pool_decoupled_diag.py` now has `--interval-evals s1,s2,...` (single fold only) and `--dense-log`. Not launched yet (see "Not done").

- Eval at step s runs BEFORE step s (model after s updates; s=0 = untrained). Step 400 = the runner's own end-of-run numbers, not repeated. Evals score all 12 structures at the runner's fixed tau (cfg tau_end) using the ORIGINAL scoring function (captured before the confusion-logger shim), log `interval_heldout_{macro_f1,top1,lift}`, `interval_train_macro_f1_{mean,min}` and per-structure `interval_train_f1_<tag>` to MLflow at that step, and write `interval_evals_seed<S>_<fold>.json` (per-structure metrics + held-out confusion matrix and per-node predictions at every eval step) into the diag results dir. `--dense-log` sets DENSE_LOG_STEPS=400 (per-step logging throughout; LOG_EVERY unchanged so clip_active_fraction keeps its definition).
- Findings that shaped the design (measured on the real model, GPU):
  1. eval->train cycle restores all 14 backbone regularization modules (reg live/total 14/14 -> 0/14 in eval -> 14/14 after train()); module flags all restored. No `.train()`-override problem on this path.
  2. An eval forward CHANGES the CUDA RNG state (CPU RNG state unchanged). So CPU, CUDA, numpy and python RNG states are saved/restored around every eval.
  3. Non-perturbation test (3-step smokes, seed 0, c=0.3, GPU): two identical controls agree exactly; a run with evals at steps 1 and 2 matches them exactly (loss 0.3565/0.3726/0.3210, preclip 0.381/1.017/0.413, clip 0/1/0, euc_share identical). Negative control with the CUDA RNG restore DISABLED: step-1 loss 0.3741 (vs 0.3726), preclip 1.071 (vs 1.017), step-2 loss 0.3218 (vs 0.3210). So evals do perturb training without the restore, the test can detect it, and the restore removes it. Limits: 3 steps, 4-decimal printed metrics, smoke has no MLflow logging path; long-run drift not tested.
- Optimizer: plain `torch.optim.Adam(groups)` (PyTorch default betas 0.9/0.999, eps 1e-8, no weight decay) in both runners.
- MoE: the M2-pool runner hard-sets `system.set_moe_mode("ablated")`. `grad_l2_moe` in these runs is for an ablated MoE. Any "MoE-live gradient check" needs a different configuration (moe mode live) and a defined check; not built.
- Suggested schedule from the operator's note: 0,25,40,50,60,75,100,150,200,250,300 via `--interval-evals` (+ 400 from the runner's end-of-run numbers). Suggested launch (fold 1IVO:A, c=0.3 or 1.0 as decided): `python scripts/wrap1_zhyp_m2_pool_decoupled_diag.py --dehydron-coeff <c> --folds 1IVO:A --seed 0 --interval-evals 0,25,40,50,60,75,100,150,200,250,300 --dense-log --device cuda`.

### Not done (as of this entry)
- Interval run not launched: the operator's precondition (ebbdd89 pushed) was not met when checked (`master` 2 ahead of `origin`), and the assistant cannot push.
- The "four free analyses" (overlay on bce_per_structure_min, lead-lag over steps 35-60, early gradient energy vs hinge step, first-100-step window-average coupling rerun) were referenced from a message that is not in this session; their exact definitions, and what "hinge step" means, are not known here. Not run.
- Coefficient decision: not applied to the sealed card (operator's call). 

## Trajectory / hinge analysis (2026-09-28; `experiments/diagnostics/trajectory_hinge_analysis.py`, output regenerable from MLflow)

Purpose: operator's two-regime hypothesis (plateau, then escape at ~steps 45-55) and the free checks. Definitions were fixed in the script header before any alignment was looked at; section 6 (noise-scaled onsets) is **post hoc** and labelled so. n = 11 independent trajectories (6 M2, 5 verify; the 400-step verify runs duplicate the first 100 steps). Everything is descriptive: coefficient co-varies with early gradient scale, n is small, thresholds are arbitrary.

**What the data show (all re-derived from logged values)**
- Hinge (mechanism-head grad > 3x its steps 15-30 median, 2 consecutive steps) by coefficient: c=1.0 steps 36, 37, 40, 42, 44, 44; c=0.3 steps 47, 49, 51, 52, 54. Coefficient shifts the hinge by roughly 10 steps, with no overlap between the two groups. Moving with it: bce_per_structure_min drop, first clip, preclip_norm.
- The burst is confined to the mechanism head (and grad_l2_euc_skip, which coincides with it in 6/6). attn_layers, moe and sdrp_head never exceed 3x their plateau median in any run.
- **Plateau is not flat.** In steps 15-30 the mechanism-head gradient varies 7-32x max/min and bce_min moves 0.62-0.67: all 11 runs share a damped oscillation (period ~5-10 steps) through ~step 35. The gradient decays into a trough (c=0.3 down to ~0.005-0.01) before the escape. A fixed-window median is a mediocre baseline; onsets are threshold-sensitive by several steps (K=2/3/5 span 2-6 steps per run).
- **Ordering.** bce_min drop precedes the mechanism-head jump: median -3 steps (9/11 earlier by >1, 2 within 1, 0 later). Matched-scale pairs (K=2/D=.01, 3/.02, 5/.05): earlier in 32/33, tied in 1, later in 0. Noise-scaled (post hoc, z=2/3/4): median -4 in all three, earlier in 10/11, 8/9, 6/6 (fewer runs at higher z because the gradient onset is not detected). bce_max rise lags the gradient jump (+3 median, 8/11 lag). euc_skip_share leads (median -3.5; 4/6 lead, 2 tie, 0 lag). The lead is small (3-4 steps) and depends on threshold pairing, so read it as "the loss on the best-fit structure starts moving no later than the gradient", not as a causal claim.
- **Cross-correlation over steps 35-60 (check 2) is uninformative.** Peak lags for mech-head vs bce_min: -1,-1,-7,-1,-1,1,-2,4,8,-8,3 (three at the edge of the +-8 range). 6/11 sit within +-2 steps, mostly with bce_min leading by ~1; the rest are scattered. Two trending series over 26 points give unstable peaks; the onset ordering above is the more reliable statement.
- **Progress-variable alignment (check 1).** Overlaying on x = bce_min does not collapse the runs (see `experiments/trajectory_overlay.png`, untracked). bce_min at the hinge: mean 0.52, range 0.45-0.61 (CV 0.10), no tighter than the hinge *step* (CV 0.13). c=1.0 and c=0.3 do not separate cleanly in that space either. The flat-plateau caveat applies: loss is ~constant before the hinge, so alignment there cannot discriminate (L) from (A).
- **Early gradient energy vs hinge (check 3).** Pooled Spearman(E0_preclip, hinge) = -0.55 (n=11): high early energy (c=1.0) goes with an *earlier* hinge, the opposite of what an Adam second-moment artifact predicts. Within coefficient the sign flips (c=1.0 +0.89, n=6; c=0.3 +0.70, n=5; E0_mech +0.77/+1.00), but those within-group spreads are driven by different seeds and the mech_lr_scale 0.3 run (lr change confounds), or by tiny differences among near-identical seed-0 runs. My quantitative (A) prediction, hinge ~ E0_mech / g_plateau^2, gives log-log slope +0.00 (bootstrap 95% CI -0.42..+0.12; (A) predicts ~+1). **This does not support (A) in this form. It does not rule out (A)**: the proxy is norm-level and Adam is per-parameter, and the prediction is my own formalisation.
- **Optimizer (check 4).** Both runners use `torch.optim.Adam(groups)`: betas (0.9, 0.999), eps 1e-8, weight_decay 0; clipping at max_grad_norm 1.0 happens before Adam. beta2^40 = 0.961, so at step 40 the step-0 gradient still carries 96% of its weight in v_hat (the second moment is ~a uniform average, as hypothesised). Norm-level effective-step proxy r_t = |g_t|/sqrt(mean g^2): median over steps 15-45 is 0.21-0.28 (0.53 for the mech_lr 0.3 run), i.e. roughly a quarter of nominal, **not a tenth**. r rises to 1-4 at the hinge (circular: the hinge is the gradient jump). CV of r at the hinge (0.36) is larger than CV of the hinge step (0.13), so no support for a threshold in r.

**What this leaves open:** (L) vs (A) is not resolved by logged data. The reduced-effective-step window (r ~ 0.2-0.3 through the trough) is real at the norm level, but the early-energy dose-response does not follow the (A) prediction across coefficients. A discriminating test needs an intervention, e.g. a run with a modified Adam beta2 (0.99) or a warm second-moment state, which changes (A) and leaves (L) alone; not run.

**Limits:** late logging is sparse (every 10th step after 150) so nothing here says anything about steps >150; no held-out curve exists before step 400 (the interval-eval run addresses that); 11 trajectories are only ~3 distinct seeds/folds' worth of dynamics for c=1.0 (seed-0 runs are near-identical).

## Follow-ups to the trajectory analysis (2026-09-28, after operator review)

**Corrections to how the Adam test is framed**
- The pooled Spearman of -0.55 is confounded by the coefficient (c=1.0 has both higher early energy and an earlier hinge). Within coefficient the signs (+0.7 to +1.0, n = 5-6) point the way (A) predicts. The fitted slope rejects the specific formula only. Wording to use: **this formulation is not supported; the question is open.** Not "ruled out".
- **beta2 = 0.99 would not test (A).** Weight of the step-0 gradient relative to the current step at step 40 is beta2^40: 0.961 (0.999), 0.669 (0.99), 0.129 (0.95), 0.015 (0.9). The hinge sits ~40 steps in, so 0.99 leaves the second-moment memory largely intact. Use 0.9 or 0.95. Lowering beta2 changes the whole trajectory, so pre-register that ONLY the hinge step is read. Prediction under (A): hinge earlier than the 36-44 range seen at c=1.0 (say before step 30). No movement => second-moment memory is not the driver. ~55 min at 100 steps. NOT yet run. Open before launching it: the hinge definition in `trajectory_hinge_analysis.py` uses steps 15-30 as its baseline and searches from step 31, which would overlap a predicted hinge <30, so the baseline window/search start must be re-fixed (and written here) BEFORE that run.
- **Grad L2 is a membership check, not a measure of how much a bucket learns.** Under Adam a parameter moves ~ lr * m_hat / sqrt(v_hat): it follows gradient-sign consistency, not gradient size, so a bucket at 2.6e-4 L2 and one at 0.13 can move at similar rates. Any wording in this thread or these notes that treats attn/MoE as "starved" because their L2 is 60-500x below the head overstates it (I relied on L2 ratios earlier too). The right measure is the per-bucket update ratio ||delta theta|| / ||theta||; `scripts/diag_gradflow_pool_frontend.py` logs it, the M2 and verify runs did not (they only have grad_l2_*). No claim about update ratios can be made from existing M2/verify runs.
- "Clipping divides the spine's gradient by preclip" is too strong. On a clipped step the spine's contribution to m and v shrinks, but its step does not shrink proportionally (Adam renormalises); the effect is real for the head, whose spikes get capped, and smaller for the spine. Update ratios at c=1.0 vs c=0.3 would show it; only c=0.3 is planned for the interval run, so a c=1.0 counterpart is a separate ~3.2 h run if wanted.

**Read-only step loggers added to `wrap1_zhyp_m2_pool_decoupled_diag.py` (`--step-loggers`)**
- `bce_struct_<tag>` (tag with ':' removed, e.g. `bce_struct_1MBNA`) for all 11 training structures at every step. List order = batch order = `[t for t in all_tags if t != hold]` (runner line ~246). Min/max alone cannot say whether the same structure leads across runs.
- `update_ratio_<bucket>` for buckets `fe_backbone, fe_proj` plus the spine buckets under the same names as `grad_l2_*` (attn_layers, moe, mechanism_head, sdrp_head, projector, euc_skip, ...), requires_grad params only, at steps s % 10 == 0 AND every step in 30..70 (the dense window is my addition to the every-10-steps request, so the c=0.3 hinge at 47-54 is not sampled once per 10 steps; `--update-ratio-dense ''` turns it off). Snapshot is copied to CPU (no GPU memory on the 4 GB card); measured on the real model 0.30 s and 123 MB per logged step, i.e. ~25 s over the run.
- Local copy: `step_loggers_seed<S>_<fold>.json` next to the confusions file (per-step loss, preclip, per-structure BCE, update ratios, snapshot seconds).
- `--smoke-hold TAG` (smoke only) lets the 3-step smoke run hold a chosen fold; training-structure order is unchanged (only the hold tag is moved to the front of the tag list).
- **Verification (1IVO:A, c=0.3, 3-step smoke).** Step 0 matches the earlier real run (`ba58175a`): loss_total 0.356609117 vs 0.356609114, preclip 0.40989622 vs 0.40989623, per-structure BCE min/max agree to ~1e-7. Steps 1-2 differ from that run by ~0.4% in loss_total ONLY because a 3-step run has a different tau schedule (tau_ceiling at step 1: 0.798 in the smoke, 0.7007 in the 400-step run). Non-perturbation was tested against a same-schedule control instead: two smoke runs with a passive recorder and NO loggers vs the logger run, steps 0-2: loss_total agrees to ~1e-8, per-structure BCE to 6e-8 (identical between control-vs-control and control-vs-loggers), preclip to <=2e-7 (control-vs-control differs by the same amount at step 2). So the loggers are indistinguishable from GPU run-to-run noise. The MLflow-side naming and schedule are covered by `tests/test_diag_step_loggers.py` (mocked; not exercised against the server yet, that happens on the first real run).
- Interval schedule for the run: `0,25,40,45,50,55,60,75,100,150,200,250,300` (45 and 55 bracket the c=0.3 hinge at 47-54).
- Interval run design per operator: c=0.3 on 1IVO:A is a diagnostic choice and does NOT amend the sealed card. Held-out at the hinge discriminates "escape = learning" (held-out moves there) from "train-structure fitting" (it does not). The beta2 test comes after the interval run (one GPU job at a time).

**Coupling rerun (attn/MoE first-100-step window average): NOT done.** Its definition came from a message that is not in this session (see the earlier "Not done" entry), so nothing was run and no number exists. Given the L2-vs-update-ratio point above, a window-averaged L2 coupling for attn/MoE would be weak evidence anyway; if wanted, it needs a written definition first (which quantity is coupled to which).

## Interval run launched (2026-09-28 ~04:38 host clock)

- Run `a003d87def7c4be88ff2ef4277aae441` in `diag/heldout-dcoef0.3`, 1IVO:A, seed 0, c=0.3, commit `7850678` (pushed; origin/master confirmed at 7850678 by the operator), git_dirty=False. Command: `python scripts/wrap1_zhyp_m2_pool_decoupled_diag.py --dehydron-coeff 0.3 --folds 1IVO:A --seed 0 --interval-evals 0,25,40,45,50,55,60,75,100,150,200,250,300 --dense-log --step-loggers --device cuda`. Log: /tmp/interval_run_1IVOA_c0.3.log (container). ETA ~3.2 h.
- **Prior files moved aside (not deleted)**: the runner skips a fold whose result file exists and the confusion logger would overwrite its file, so `decoupled/seed0_hold_1IVOA.json`, `confusions_seed0_1IVOA.json` (both from run `ba58175a`) were moved to `data/gates/diag_heldout_dcoef0.3/prior_run_1IVOA_ba58175a/`, together with a copy of `NOT_A_CARD_RESULT_dcoef0.3.json` (the new run rewrites it). All untracked. Anything reading the earlier 1IVO:A c=0.3 result from `decoupled/` must look in that folder now.
- Startup checks passed: guard clean, step 0/1 loss and preclip match `ba58175a` (0.3566/0.410, 0.3681/0.989), `bce_struct_*` (11) and `update_ratio_*` (7 buckets) present in MLflow from step 0, interval eval at step 0 ran (held macro-F1 0.2738).

## Interval run: predictions, replicate design, risks (written 2026-09-28 04:45 UTC at step 12 (log confirmed at that moment); the ONLY interval eval seen so far is step 0: held macro-F1 0.2738, train mean 0.2731)

Run `a003d87d` (1IVO:A, c=0.3). ~28 s/step, so evals land at roughly: step 25 ~12 min of training, step 40 ~19 min, step 75 ~35 min, step 300 ~2.4 h; 400 steps ~3.1 h plus evals.

**Reference values (re-derived).** Class counts of the held-out structure: 26 / 18 / 467 (n=511; read from the earlier run's saved confusion matrix, rows [26, 18, 0, 0, 467]). All-majority floor macro-F1 = (2*467/(2*467+44))/3 = **0.3183** (my arithmetic; no floor value is logged in MLflow). Step 0 (0.2738) is below the floor. Node scale (tp-only, small majority-F1 change ignored): one node in the class of 26 = 0.025, class of 18 = 0.035; three nodes = 0.069 / 0.095; two nodes = 0.048 / 0.067.

**Predictions (operator's, with numbers filled in; the train-rise threshold is mine and can be amended before step 40 lands).** Held-out = `interval_heldout_macro_f1`; train = `interval_train_macro_f1_mean`.
- **Escape is learning:** held(60) >= held(40) + 0.07 AND held(75) >= held(40) + 0.07 (i.e. it moves ~3 nodes at the fold's scale and stays there through step 75). Steps 45, 50, 55 are read for the curve shape, not for the rule.
- **Escape is train-structure fitting:** |held(75) - held(40)| <= 0.035 (about one node) while train(75) - train(40) >= +0.05 (proposal).
- **Anything else is unresolved.** No stretched reading.
- **Step count:** peak = step of maximum held-out F1 among the interval evals. If the peak is at step <= 200 AND held(300) <= peak - 0.05 (about two nodes), 400 steps is too long. If held(300) >= held(250) >= held(200) (still rising), no case for shortening. Anything else: no conclusion. Caveat: one fold, one seed, evals every 50 steps after 100, and step 400 is the runner's own end-of-run number.
- **Train-side quantity, fixed:** the fitting branch uses `interval_train_macro_f1_mean` = the mean over the 11 training structures of each structure's macro-F1 (itself the mean of per-class F1 over the classes present in that structure). `interval_train_macro_f1_min` (the worst structure) is logged but is NOT used in any rule.
- **Reading discipline:** during the run only the pre-registered quantities are read: `interval_heldout_macro_f1` at each eval step, `interval_train_macro_f1_mean`, and (for the hinge context) the update ratios. The 11 `bce_struct_*` traces and per-bucket update-ratio patterns are exploratory: to be analysed after the run, labelled as post hoc, not read live for a story. (Startup verification of their presence/values at step 0 was a plumbing check, not an analysis.)
- Caveat on all thresholds: held-out macro-F1 is quantized and the fixed-seed nondeterminism floor is unmeasured (next section); if the replicate shows the floor is more than ~1 node, widen these thresholds accordingly and say so.

**Free replicate.** This run has the same fold, seed, coefficient and code lineage as the earlier 1IVO:A c=0.3 run `ba58175a` (final held-out macro-F1 0.46628, confusion rows true/cols pred for the three present classes: 6/2/18 of 26, 3/3/12 of 18, 14/6/447 of 467; files in `prior_run_1IVOA_ba58175a/`). Steps 0-10 already agree to loss 3e-8, preclip 9e-7, mechanism-head gradient 2.7e-6 (relative).
- Endpoint comparison at the end: number of held-out nodes whose predicted class differs between the two runs (same statistic as the 45/511 between c=1.0 and c=0.3) and the macro-F1 difference. Identical => run-to-run noise at fixed seed is negligible. Differing by k nodes => k is the nondeterminism floor for this setup, and the earlier 45/511 c=1.0-vs-c=0.3 flip count and the "three nodes is meaningful" rule should be re-read against it.
- **Replicate match rule (written at step ~15, before any endpoint exists).** Statistic: K = number of held-out nodes whose predicted class differs between `a003d87d` and `ba58175a` (using `held_pred` vs the earlier run's saved `pred`), plus |ΔF1|. K = 0: deterministic to this resolution. K = 1-3: fixed-seed floor of 1-3 nodes; the "three nodes is meaningful" rule then leaves little margin and must be reported as such. K >= 4: noise alone can produce that many flips, so widen the node rule to at least K + 3 before using it, and re-read the earlier 45/511 against K. **Limits of this rule:** one replicate is ONE draw of the floor, not its distribution (report it as "at least K", not "the floor is K"). And endpoint differences cannot be attributed: early agreement (3e-8 at steps 0-10) does not distinguish chaotic amplification of GPU-level numerical differences from amplification of a residual eval/logger perturbation, because chaos amplifies both equally. The only statement supported is the size of the total difference between a run with evals/loggers and one without.
- **Confound to state with any difference:** it is the SUM of GPU nondeterminism and any residual perturbation from the interval evals/loggers (the earlier run had neither). The smoke controls bound the logger side at ~1e-8 over 3 steps only. Also track how the per-step gap to `ba58175a` grows (chaotic amplification is expected); compare with a script in /tmp so the tree stays clean.

**Interval evals as they land (run `a003d87d`; appended live, values copied from the log/eval JSON):**
- step 0: held 0.2738, train mean 0.2731 (min 0.2558). Predicts some non-majority classes on train/held (untrained model).
- step 25 (04:51 UTC): held 0.3183, train mean 0.3189 (min 0.3060). Held-out confusion: every one of the 511 nodes predicted class 4 (majority) = the all-majority floor, exactly. **Step-25 memory test passed** (eval ran with Adam state resident; GPU 3620 MiB, no error).
- step 40 (04:58 UTC): held 0.3183, train mean 0.3189 (min 0.3060): identical to step 25 at 4 decimals; held-out again all-majority. **This is the baseline for the pre-registered rules.** Thresholds it implies: learning branch needs held(60) and held(75) >= 0.3883; fitting branch needs |held(75) - 0.3183| <= 0.035 and train mean(75) >= 0.3689.

**Steps 45-75 (05:16 UTC): pre-registered rules applied as written.** Held-out and train are at the floor at every eval from 25 through 75, identical to 4 decimals: held F1 0.3183 (top-1 0.9139 = majority rate), train mean 0.3189, train min 0.3060. Held-out confusion at steps 25, 40, 45, 50, 55, 60, 75 is identical: rows (true 26 / 18 / 467) = [0,0,0,0,26], [0,0,0,0,18], [0,0,0,0,467] over predicted classes 0-4 (all 511 predicted class 4; true positives 0 / 0 / 467). Step 0 for comparison: TP 0 / 0 / 347, rows [0,0,5,0,21], [0,0,8,0,10], [0,0,120,0,347].
- **Escape is learning:** FAILS (needs held(60) and held(75) >= 0.3883; both 0.3183).
- **Escape is train-structure fitting:** does NOT fire (held condition met, |held(75) - 0.3183| = 0; the train condition needs train mean(75) >= 0.3689, observed 0.3189, i.e. +0.000 from step 40).
- **Result: the "anything else is unresolved" bucket.** Through step 75 no structure's argmax (train or held-out) has moved off majority.
- Where the evals sit relative to the hinge (fixed definition of `trajectory_hinge_analysis.py`, run `a003d87d`, logged through step 74): mechanism-head hinge K=2/3/5 = 51/51/55; bce_min drop D=.01/.02/.05 = 45/48/50; first clip = 55. Identical, step for step, to the earlier run `ba58175a` (51/51/55, 45/48/50, 55). So the evals at 45, 50, 55, 60 bracket the hinge and the first clip, and nothing in the SDRP argmax changed across them. Note (observation, not a rule change): the hinge is in the mechanism head / dehydron-BCE path while the eval scores the SDRP class head, so the pre-registered link "hinge escape -> SDRP macro-F1 moves" was never guaranteed by the mechanism; nothing here supports it by step 75. The earlier run `ba58175a` ended at held 0.4663 (TP 6 / 3 / 447), so the leave-the-floor event happens between step 75 and step 400 in that run; steps 100, 150, 200, 250, 300 will show where (with the eval-point gaps 75-100 and 100-150 unread).
- Replicate (partial): the two c=0.3 1IVO:A runs agree on hinge and clip steps exactly; endpoint comparison still pending.

**Reading notes recorded at step ~41 (operator review), before steps 45-75 land:**
- Steps 25 and 40 are a FLOOR reading: all 511 held-out nodes predicted majority, so 0.3183 carries no information about what the model has learned; it is consistent with a real stall but cannot say why. The escape is visible in these evals only if the argmax on some minority nodes flips.
- Floor vs stuck eval path, checked: the eval records store only argmax-derived numbers (top-1, majority rate, lift, macro-F1, confusion), no per-node scores, so a logit comparison is not possible. The eval path DOES respond to weights (step 0 -> 25: held top-1 0.679 -> 0.914, held F1 0.2738 -> 0.3183, train mean 0.2731 -> 0.3189). Training-loss witness, steps 25 -> 41 (MLflow): loss_sdrp_ce 0.555 -> 0.467 (window means 15-24: 0.600, 30-41: 0.493); loss_dehydron_bce 0.682 -> 0.679 (window means 0.6823 -> 0.6793; chance = ln 2 = 0.693); bce_per_structure_min/max flat. So the model changes (SDRP CE falls ~15%) without any argmax crossing: consistent with a plateau in the argmax, not a frozen eval.
- Size of the learning bar: 0.3883 is 0.07 above the floor, i.e. ~3 nodes of 511. If held-out lands near 0.39 the reading is "three nodes of 511 flipped", not a change in generalisation, and the flipped nodes must be checked to be CORRECT predictions (true-positive counts per class in the confusion), not merely different ones. Every step 60/75 report gives the confusion counts, not just F1.
- Blind spot of the fitting branch: train mean rising but staying < 0.3689, or held-out moving by one node, fires neither branch; that is the "unresolved" bucket and stays unresolved.
- Schedule gap: no evals between 75 and 100 or between 100 and 150 (cannot be changed mid-run). If held-out sits at the floor until step 100 or later, the 200/250/300 evals answer the step-count question and there is no earlier peak to find; do NOT read the curve between eval points.

**Risks.** GPU 3620/4096 MiB (reserved). The step-0 eval ran before the optimizer had any state, so **step 25 is the first eval with Adam state resident and is the first real memory test** (eval activations run under no_grad and should be below the training peak, not proven). A crash writes no fold result, so a relaunch is clean; the MLflow run would be left non-FINISHED and the interval/confusion JSONs would be overwritten by the rerun.

## Interval run RESULTS (run `a003d87d`, 1IVO:A, c=0.3, seed 0; finished 07:48 UTC, 3.17 h, MLflow FINISHED, commit 7850678, git_dirty=False; read 13:10 UTC)

No OOM, all 13 evals ran (step 25 was the first with Adam state resident: fine). Held-out rows are true classes 26 / 18 / 467 over predicted classes 0-4.

| step | held F1 | train mean / min | held TP (26/18/467) | held rows |
|---|---|---|---|---|
| 0 | 0.2738 | 0.2731 / 0.2558 | 0/0/347 | [0,0,5,0,21] [0,0,8,0,10] [0,0,120,0,347] |
| 25-75 (7 evals) | 0.3183 | 0.3189 / 0.3060 | 0/0/467 | all 511 predicted class 4 |
| 100 | 0.3189 | 0.3644 / 0.3185 | 0/0/465 | [0,0,0,0,26] [4,0,0,0,14] [2,0,0,0,465] |
| 150 | 0.3985 | 0.5065 / 0.3813 | 1/2/462 | [1,0,0,0,25] [7,2,0,0,9] [3,2,0,0,462] |
| 200 | 0.4716 | 0.6970 / 0.4853 | 6/2/457 | [6,0,0,0,20] [3,2,0,0,13] [7,3,0,0,457] |
| 250 | 0.4889 | 0.8848 / 0.7599 | 6/3/456 | [6,1,0,0,19] [4,3,0,0,11] [7,4,0,0,456] |
| 300 | 0.4564 | 0.9550 / 0.8751 | 5/2/458 | [5,1,0,0,20] [4,2,0,0,12] [7,2,0,0,458] |
| 400 (runner) | 0.4593 | 0.9831 / 0.9329 | 6/3/444 | [6,2,0,0,18] [4,3,0,0,11] [17,6,0,0,444]; held top-1 0.8865 (< majority 0.9139), lift 0.970 |

Final: clip_active_fraction 0.0488 (logged points) / 0.0175 of all steps (7 of 400); g_fit_train PASS. `RESULT=FAIL_NO_SIGNAL` is the sealed card's S1/S2 signal gate applied to ONE fold (S2 needs several folds above threshold): mechanical, not a card result (written to `NOT_A_CARD_RESULT_dcoef0.3.json`).

**Pre-registered rules, final application**
- Escape is learning: FAILED (held(60) = held(75) = 0.3183 < 0.3883). Fitting branch: did not fire at step 75 (train +0.000). Verdict at the pre-registered read points: unresolved (as recorded above).
- Step count: peak among interval evals = step 250 (0.4889); held(300) = 0.4564 is 0.0325 below the peak (about one node, less than the 0.05 bar); the peak is not at <= 200, and held is not rising through 300 (300 < 250). Neither condition met: **no conclusion on shortening 400.** Descriptive only: held-out sits in ~0.46-0.49 from step 200 on.
- Timeline (descriptive, post hoc reading): hinge 51, first clip 55; SDRP argmax first leaves the all-majority state between steps 75 and 100 (step 100: a few non-majority predictions, still no minority true positive, F1 0.3189); first minority true positives at step 150; train mean starts rising by step 100 (0.3644). So the argmax escape trails the mechanism-head hinge by roughly 50-100 steps and the pre-registered link hinge -> held-out movement was not seen at step 60/75. The 75-100 and 100-150 gaps are unread.

**Replicate against `ba58175a` (same fold/seed/coefficient/lineage): K = 4 of 511 held-out predictions differ.** Same labels. The four flips: three true-467 nodes predicted class 4 -> class 0, and one true-18 node class 4 -> class 0 (both wrong before and after). Minority true positives are identical (6 of 26, 3 of 18), BUT the F1-relevant counts are not: class-0 FP +4 (17 -> 21; F1 0.2449 -> 0.2264), class-4 TP -3 / FP -1 / FN +3 (F1 0.9470 -> 0.9447), class 1 unchanged; contributions to the macro-F1 difference -0.0062, 0.0000, -0.0008. (Correction, 2026-09-28: an earlier line here said only majority-class nodes moved; one flipped node, index 458, is a true-18 node.) Held F1 0.45933 vs 0.46628 (diff 0.0069). Hinge and first clip identical step for step.
- Per the pre-registered rule: K >= 4 => widen the node rule to at least K + 3 = 7 nodes before using it, and re-read the earlier 45/511 against K. Report as "at least 4" (one draw of the floor, not its distribution).
- Confound (unchanged): the difference is the SUM of GPU nondeterminism and any effect of evals/loggers; not attributable.
- Statistic (operator decision 2026-09-28): count both, separately. All flips (K) measures nondeterminism directly. The DECISION statistic is the F1 gap expressed as per-class TP/FP/FN deltas, not TP alone (table above). The earlier note that the rule counts any flip while F1 depends on TP/FP/FN is resolved this way.
- Effect on earlier conclusions (corrected 2026-09-28 after review; an earlier version said "strengthened", which was wrong): widening the bar from 3 to 7 nodes puts the earlier 1IVO:A c=1.0-vs-c=0.3 gap (-0.0682, ~3 nodes) under it, but that MOVES THE BAR and adds no evidence; the 3-fold verdict is unchanged. A same-seed rerun is a lower bound on noise: seed-to-seed variance is unmeasured and probably larger, so this replicate cannot bound the coefficient comparison either way. The 45 differing predictions between the coefficients (vs K = 4 between reruns) shows the models differ, which is expected because the loss differs; it does not show their held-out quality differs.

**Still not done:** the exploratory analysis of the 11 `bce_struct_*` traces and the per-bucket update ratios (post hoc, to be labelled as such); the beta2 (0.9/0.95) hinge test, whose hinge definition must be re-fixed and committed first; commit of these notes (uncommitted markdown); the untracked result files under `data/gates/diag_heldout_dcoef*/` and `experiments/trajectory_overlay.png` (operator's call whether to commit).

## Follow-ups after the interval run (2026-09-28 ~13:45 UTC; scripts `experiments/diagnostics/posthoc_interval_run_1IVOA.py`; checks below are read-only)

**Step count (1IVO:A, one fold, one seed).** Held-out F1 from step 200 on: 0.4716, 0.4889, 0.4564, 0.4593 (range 0.033, about one node) while train mean rose 0.70 -> 0.98. Steps 200 -> 400 bought a large train-side gain and no measurable held-out change in either direction. The pre-registered rule (needs a drop of >= 0.05) cannot fire, so "no conclusion" is right, but the rule is one-sided and does not answer the cost question: if 200 steps sufficed, the 12-fold run halves (~38-42 h -> ~19-21 h). One fold/one seed justifies a test, not an amendment to the locked 400. **Quoting rule:** the pre-specified endpoint is step 400 (0.4593); the step-250 peak (0.4889) is a selected value and is not quoted as the result.
- Train-fit gate at the interval steps (card: `G_FIT_TRAIN_MACRO_MIN = G_FIT_TRAIN_MIN_STRUCT = 0.40`, i.e. train mean AND worst structure >= 0.40; verified in `wrap1_zhyp_m2_pool_decoupled.py:86-87`): train min = 0.3813 at step 150 (FAILS), 0.4853 at 200 (passes, margin 0.085), 0.7599 at 250; train mean 0.5065 / 0.6970 / 0.8848. The step-200 STATE OF A 400-STEP RUN passed the gate with modest margin and the step-150 state did not. (Corrected 2026-09-28: an earlier line said "a 200-step run would have passed"; the curriculum and Gumbel schedules stretch over the total step count, e.g. tau_ceiling at step 1 is 0.798 in a 3-step run vs 0.7007 in a 400-step run, so a real STEPS=200 run ends in a different state. The test supports "checkpoint at 200"; shortening STEPS needs its own run.)

**Held-out signal vs chance (permutation null: labels permuted, predictions fixed, 100k permutations, one-sided).** Caveat: assumes exchangeable nodes; nodes within a protein are not independent, so p-values are optimistic. Step 100: no minority hit (p = 1.0; macro-F1 p = 0.27). Step 150: class-26 TP 1 (null mean 0.56, p = 0.44), class-18 TP 2 (null 0.14, p = 0.007), macro-F1 p = 0.003. Step 200: class-26 6 (null 0.82, p = 8e-5), class-18 2 (null 0.17, p = 0.011). Step 250: 6 (p = 9e-5) and 3 (null 0.28, p = 0.002). Step 300: 5 (p = 7e-4), 2 (p = 0.011). Final (400): class-26 6 (null 1.38, p = 0.001), class-18 3 (null 0.39, p = 0.005), macro-F1 p < 1e-5. The prior run's final: 6 (null 1.17, p = 6e-4), 3 (null 0.38, p = 0.005). So from step ~200 the predictions carry label information beyond chance (permutation test only says "above chance", not how well it generalises; multiple evals/statistics, no correction). A below-majority top-1 (0.8865 vs 0.9139, lift 0.970) does not mean no signal: trading majority accuracy for minority recall lowers top-1 by construction. The claim that the SDRP target is built from edge-type histograms the model can read (so above-chance held-out = generalising the labelling rule, defect B) is the operator's and was not verified in this session.

**Per-node scores were NOT stored by the interval eval** (asked twice; answer: the records hold argmax-derived metrics, confusions and `held_pred` only). **Fixed in commit `504f377`:** every interval eval now saves held-out logits, mechanism scores and labels (`interval_scores_*.npz`) and logs macro AUROC/AUPRC (per class and for the mechanism head), which do not jump 0.03 per node. Verified by smoke (1IVO:A, c=0.3, evals at steps 0/1/2, with step loggers): npz holds (511, 5) logits per step, scores change between steps, step-0 held F1 0.2738 matches the real run, and loss / preclip / per-structure BCE agree with the no-logger controls within run-to-run noise (loss <= 7.5e-9 vs 7.5e-9 between the controls themselves; preclip <= 5.7e-8 vs 2.1e-7; BCE 6e-8). Unit tests compare the metrics with sklearn. Also new: `--results-suffix` (results dir `diag_heldout_dcoef<c><suffix>/`, so a fold that already has a result is neither resume-skipped nor overwritten; no file moves), and a fix so `--smoke` interval evals honour `--smoke-hold`. The 1IVO:A interval run itself has NO stored scores (it predates this change), so AUROC/AUPRC for it cannot be computed.

**Post hoc, exploratory, ONE run (`a003d87d`); descriptive only.**
- Update ratio (median of steps 45-60 vs 30-44): mechanism_head x2.4, euc_skip x3.3, fe_backbone x2.5, projector x2.3; attn_layers x1.06, moe x0.85, sdrp_head x1.04. So at the hinge the buckets that move more are the mechanism/euc_skip path plus backbone/projector; the SDRP head, attention and MoE do not.
- Steps 80-150 (single steps every 10, noisy): relative to the 30-44 trough, projector x2.7-3.9 and fe_backbone x2.8-3.7 stay elevated; sdrp_head x0.85-1.14, attn_layers x0.5-0.64, moe x0.7-1.3; mechanism_head x1.4. The SDRP argmax escape (75-150) is therefore accompanied by higher update ratios in projector/backbone, not in the SDRP head. (Single steps only after step 70; no window medians there.)
- Per-structure BCE: 1F88:A has the lowest BCE at 26 of the 31 steps in 40-70 (a level, not a fall order). BCE-fall step (5-step rolling median below its steps 15-30 median minus 0.05): 8 of 11 structures fall within steps 52-56 (1BG1 53, 1F88 52, 1HHP 52, 1LYZ 54, 1MBN 54, 2SHP 54, 2Z6H 54, 1UBQ 56); three lag (1TEN 63, 4OBE 64, 1TIM 67; at threshold 0.08: 85, 95, 114). So most structures fall together, with a lagging subset, not one after another (resolution: 5-step median and threshold, sensitivity 0.03/0.05/0.08 in the script output).
- Train-structure F1 escape (gain at steps 100/150 over each structure's step-75 floor) vs BCE-fall order: Spearman -0.05/-0.16 (thr 0.03), -0.13/+0.18 (0.05), -0.05/-0.02 (0.08), n = 11: no relationship. Vs structure size: -0.37 (gain@100), -0.46 (gain@150), n = 11: a weak tendency for smaller structures to leave the floor first (1UBQ, 1HHP, 1LYZ at step 100), not distinguishable from zero at this n.
- Window-mean losses (steps 0-24 ... 300-399): SDRP CE 0.745, 0.489, 0.403, 0.342, 0.294, 0.243, 0.150, 0.076 (smooth decline, no step at the hinge); dehydron BCE 0.692, 0.676, 0.623, 0.576, 0.535, 0.402, 0.132, 0.035 (leaves chance gradually, then falls fastest between steps 150 and 300).

**PROPOSED pre-registration for a second-fold interval run (to be confirmed by the operator BEFORE launch; written 2026-09-28, nothing launched).** Fold 2Z6H:A (class counts 495/30/8, n = 533), c = 0.3, seed 0, `--step-loggers`, `--dense-log`, per-node score saving (once committed), separate results directory (new flag, no file moves). Proposed eval steps: 0, 50, 75, 100, 125, 150, 175, 200, 225, 250, 300, 350 (step 400 = runner end-of-run); evals cost ~4 min in total over a run.
- Resolution problem with the "two nodes" rule on THIS fold: one node in the class of 30 = 0.0215 macro-F1, but one node in the class of 8 = 0.074 (tp 1 of 8: F1 0.222/3). So a single class-8 node exceeds "two nodes of the 30-class" (0.042). Proposed rule: "no evidence later steps help" iff |F1(400) - F1(200)| <= 0.075 (one class-8 node, the coarsest quantum on this fold) AND the train-fit gate would already pass at step 200 (train mean >= 0.40 and train min >= 0.40 at the step-200 eval). The continuous scores (macro AUROC/AUPRC at 200 vs 400, with a node-bootstrap CI) are reported alongside as descriptive; no threshold pre-registered for them because their noise level is unknown.
- Caveat: one fold and one seed; the answer applies to the locked 400-step question only as a test, not as an amendment.

## Second-fold interval run: FINAL pre-registration (2026-09-28, written before launch; rule A chosen by the operator, with two edits) and review qualifications

**Rule (signed, replaces the "PROPOSED" paragraph above).** Fold 2Z6H:A (495/30/8, n = 533), c = 0.3, seed 0. Let d = F1(400) - F1(200), both from the same run (step 400 = runner end-of-run; step 200 = interval eval).
- Later steps HELP: d > +0.075. Later steps HURT: d < -0.075. Otherwise: no detectable difference at this fold's resolution. (Signed, so a big drop is not read as "no evidence".)
- Precondition: the train-fit gate must already pass at the step-200 eval (train mean >= 0.40 AND train min >= 0.40). If it fails at 200, the reading is "gate not met at 200", regardless of d.
- Also report per-class TP/FP/FN at steps 200 and 400 (count changes are cleaner than F1).
- Why 0.075 and not two 30-class nodes (0.042): every node scale in this thread is a best case with zero false positives. One extra hit in the class of 8 is worth up to 0.074 (0.026 with 17 FPs already predicted in that class); in the class of 26 up to 0.0247 (0.0152 with 17 FPs; my arithmetic, slightly above the ~0.012 quoted in review, not material).
- Scope: one fold, one seed, a checkpoint question (see the corrected "200-step" note above); not an amendment to the locked 400.

**Run configuration.** `python scripts/wrap1_zhyp_m2_pool_decoupled_diag.py --dehydron-coeff 0.3 --folds 2Z6H:A --seed 0 --interval-evals 0,50,75,100,125,150,175,200,225,250,300,350,399 --dense-log --step-loggers --variant interval_scores --results-suffix _interval --device cuda`. Results go to `data/gates/diag_heldout_dcoef0.3_interval/` (nothing existing is skipped or overwritten). MLflow tags set on the run: `variant=interval_scores`, `results_suffix=_interval`, `interval_evals=<schedule>`, `step_loggers=true`, `dense_log=true` (both 2Z6H c=0.3 runs share the name `wrap1_zhyp_m2_pool_decoupled_hold_2Z6HA`; tags and experiment run id tell them apart).
- Step 399 vs the final eval: eval steps use before-step semantics, so step 399 is the model after 399 updates, NOT the state the runner scores (after 400). Step 399 is a near-endpoint sample, not an identity test. The identity test is a separate eval taken right after the LAST update and recorded as step 400 (same model state, same tau_end, same eval path as the runner's end-of-run scoring).
- **Identity check, pre-registered:** interval step-400 held-out macro-F1 and held-out predictions must equal the runner's end-of-run held-out numbers EXACTLY. A mismatch means the interval eval path (tau or eval mode) differs from the runner's, and every interval reading from this run and the 1IVO run is suspect. The 3-step smoke passed this check but only trivially (the model is at the all-majority floor at 3 steps, so both paths predict all-majority): it shows the plumbing works, not that tau/path match. The real check is at the end of this run, where predictions are non-trivial.

**Second replicate (pre-registered now).** The earlier 2Z6H:A c=0.3 run `1212cdf8f81d45caaf2630262996f4dd` (confusions in `data/gates/diag_heldout_dcoef0.3/`, committed) is compared with this run, which tests the new score-saving/final-eval code against a run without it and gives K on a second fold.
- Steps 0-50, relative difference in loss_total, preclip_norm, grad_l2_mechanism_head, bce_per_structure_min. Calibrated on the 1IVO pair (a different run pair): max over steps 0-10: 3e-8 / 9e-7 / 2.7e-6 / 2e-7; over 0-50: 2.4e-6 / 3.1e-5 / 3.5e-5 / 4.6e-6 (drift grows with step; ~1e-6 holds only for roughly the first 10-30 steps). Match: <= 1e-5 through step 10 for all four, and <= 1e-4 through step 50 (about 3x the 1IVO worst); report the step at which each metric first exceeds 1e-5.
- Hinge (grad_l2_mechanism_head > 3x its steps 15-30 median, 2 consecutive steps) and first clip within +-2 steps of `1212cdf8` (hinge 54, first clip 55).
- Endpoint: K = number of held-out predictions that differ, plus per-class TP/FP/FN deltas and the F1 gap, exactly as for the 1IVO replicate. Same caveats (one draw; a difference is nondeterminism plus any eval/logger effect).

**Review qualifications (2026-09-28) recorded against earlier statements**
1. Permutation p-values: three steps and two classes were quoted because they showed hits, so they are SELECTED. Only step 400 was pre-specified. Node-wise permutation at step 400: class-26 p = 0.001, class-18 p = 0.005 (Bonferroni x2: 0.002, 0.010). Nodes along a chain are correlated, so an exact cyclic-shift null (all 510 non-identity shifts of the labels against the predictions, floor p = 1/511 = 0.0020) was also run: step 400 class-26 TP 6 (null mean 1.36) p = 0.0020, class-18 TP 3 (null 0.38) p = 0.0020, macro-F1 p = 0.0020 (Bonferroni x2: 0.0039). Steps 150-300 are descriptive: step 150 class-26 p = 0.446 (not above chance), class-18 p = 0.0098; steps 200/250/300 class-26 p = 0.0020/0.0020/0.0039, class-18 p = 0.0137/0.0059/0.0137. Quote step 400 only.
2. Checkpoint vs shorter run: corrected above in the step-count paragraph.
3. The earlier statement that the burst is local to the mechanism head and euc_skip was about grad L2. The update-ratio (post hoc, one run, single noisy steps) result qualifies it: at the hinge update ratios also rise for fe_backbone (x2.5) and projector (x2.3) while attn_layers, moe and sdrp_head stay ~x1; frontend/projector then stay 2.7-3.9x above the pre-hinge trough through steps 80-150, when the SDRP argmax first leaves the floor. A hypothesis for later, not a finding: whether the SDRP escape needs the frontend/projector to move.
4. Score-logger tests: the metric tests were partly checked against sklearn, which is circular; hand-computed cases were added (AUROC 3/4 by pairwise counting, reversed ranking = 0, all-tied scores = 0.5 AUROC and prevalence AUPRC, class without positives skipped, mechanism AUROC 3/4 and AP 5/6).

## Open items

1. DONE (46b6280): git provenance guard in both runners.
2. DONE: both single-variable arms ran (see above).
3. Decide the 12-fold config (dehydron_coeff 1.0 vs 0.3): pending the held-out comparison above. Changing the sealed runner's DEHYDRON_COEFF needs a documented prereg amendment (the diag wrapper avoids editing the sealed script).
4. Push local commits to origin (needs GitHub credentials on the host; the assistant container has none). Still owed: phase-aware gate rule for a 400-step sealed launch. Per-structure BCE and update-ratio logging now exist (`--step-loggers`, see above). Interval run (c=0.3, 1IVO:A) waits on: push of ebbdd89 and the commit containing the loggers, then launch. Then the beta2 (0.9/0.95) hinge test (hinge definition to be re-fixed first).
