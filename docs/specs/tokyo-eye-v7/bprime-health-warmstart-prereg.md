# Tokyo Eye v7 — B′ surgical warm-start health (prereg)

**Status:** PREREG  
**Date:** 2026-07-21  
**Decision:** Surgical B′ (keep Fix-1 radial/gate/MoE; Hyp MP primary; no Euc trunk transfer)

## Donor

| Field | Value |
|-------|--------|
| Checkpoint | `FIX1_SPARSITY_CHAMPION_CKPT` (`fix1_s4_sparsity_confirm_continue_v2` / ep48) |
| Role | Compare-only biology baseline; weight **donor** only |

## Transfer policy

| Keys | Action |
|------|--------|
| `node_emb.*`, `radial_head.*`, `angular_head.*`, `gate.*`, experts, uncertainty, `log_c`, rim/geom auxiliaries, T1a buffers | **Transfer** |
| `convs.*`, `norms.*` (SE(3) Euc MP) | **Deny** — unused under Hyp MP primary |
| `hyp_mp.*` | **Cold init** (absent on donor) |

Runtime locks: `hyp_mp_primary=True`, `se3_aux=False`, `hyperbolic_mp_graph=False` (no S4 euc-conv-on-hyp-edges).

Warmstart artifact resets `global_epoch=0`, `phase=1` so curriculum restarts with warm weights.

## Mitigations carried from Fix-1

- `init_seed=2` stack discipline
- T1a input z-score + gate SASA
- Prototype repulsion + gate softplus floor ≈ 6.612
- Rim-fanout + geom angular prior (feeler lineage flags)
- **Do not** pass `--hyperbolic-mp-graph`

## Health Pass bars (Stage A-12 micro-run)

All required for seal:

1. Forward audit: `hyp_mp_primary=true`, `se3_aux=false`
2. Final `disc_r_mean` ≥ 0.25 (no origin collapse)
3. No monopoly blow-up: routing Gini not worse than donor band without note (telemetry only this phase)
4. Learned curvature finite and > 0
5. Train completes without NaN loss

**Non-claims:** biology Pass vs Fix-1; KRAS hubs; sparsity champion status.

## Commands

```bash
make seal-v7-bprime-warmstart
make train-v7-bprime-health
make train-v7-bprime-health-continue DEVICE=cuda   # from v7_best_disc; default 24 epochs
# later: grade/seal when bars met — do not promote until disc_r≥0.25 (+ eligible v7_best)
```

## Artifacts

| Path | Role |
|------|------|
| `checkpoints/v7/tokyo_eye_v7_bprime_warmstart.pt` | Surgical donor → TokyoEye |
| `checkpoints/v7/runs/tokyo_eye_v7_bprime_health_v1/` | Health micro-run (FAIL disc_r≈0.203) |
| `checkpoints/v7/runs/tokyo_eye_v7_bprime_health_continue_v1/` | Longer continue from disc ckpt |
| `data/gates/tokyo_eye_v7_bprime_health_prereg.json` | This lock |
| `data/gates/tokyo_eye_v7_bprime_health_closeout.json` | Micro-run closeout |
