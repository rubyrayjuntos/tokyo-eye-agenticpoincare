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

## Open items

1. DONE (46b6280): git provenance guard in both runners.
2. DONE: both single-variable arms ran (see above).
3. Decide the 12-fold config (dehydron_coeff 1.0 vs 0.3). Seed 1 done (see above); remaining question is the 400-step horizon and the BCE cost there. The 12-fold runner has DEHYDRON_COEFF as a module constant, no CLI flag; changing it needs a documented prereg amendment.
4. Push 46b6280 (and any later commits); commit this note and the draft card.
