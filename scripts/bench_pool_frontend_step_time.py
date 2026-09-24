#!/usr/bin/env python3
"""Benchmark: per-step wall time and peak GPU memory for one LOSO fold's full
mean-over-11-train-structures Adam step, on the EquiformerPoolFrontend
(cold_init) + activation-checkpointed backbone.

Not part of any sealed protocol -- pure operational sizing, used to set the
step/fold/arm scope in ``data/gates/tokyo_eye_equ_wrap1_zhyp_m2_pool_lr_prereg
.json`` (see that file's ``compute_budget`` section for the recorded result).
Committed here, not left as a one-off /tmp script, so the timing claim in
that prereg is independently re-runnable -- e.g. after a GPU change, or to
sanity-check the number before scoping a follow-up card.

Usage:
    python3 scripts/bench_pool_frontend_step_time.py [--hold 1MBN:A] [--steps 25]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import wrap1_zhyp_g_fit as G  # noqa: E402
from experiments.training.v8.run_v8_experiment import build_equiformer_pool_system  # noqa: E402
from science.tokyo_eye.v8.equiformer_frontend import DEFAULT_WEIGHT_MAP, load_weight_map  # noqa: E402
from science.tokyo_eye.v8.heads import sdrp_cross_entropy  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hold", default="1MBN:A", help="fold to benchmark (trains on the other 11)")
    ap.add_argument("--steps", type=int, default=25, help="consecutive steps to time")
    ap.add_argument("--max-neighbors", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    dev = torch.device(args.device)
    cfg = load_weight_map(REPO_ROOT / DEFAULT_WEIGHT_MAP)
    all_tags = G._all_structure_tags()
    train_tags = [t for t in all_tags if t != args.hold]
    print(f"benchmarking hold={args.hold}, {len(train_tags)} train tags: {train_tags}")

    torch.manual_seed(0)
    system, _ = build_equiformer_pool_system(
        cfg, equiformer_ckpt=None, device=dev, freeze_backbone=False,
        max_neighbors=args.max_neighbors, cold_init=True,
    )
    system.set_moe_mode("ablated")
    system.frontend._backbone.gradient_checkpointing_block_list = [1] * int(system.frontend._backbone.num_layers)
    opt = torch.optim.Adam([
        {"params": system.frontend.parameters(), "lr": 1e-5, "name": "backbone"},
        {"params": system.spine.parameters(), "lr": float(cfg["lr_hyperbolic"]), "name": "hyperbolic"},
    ])

    t0 = time.time()
    batches = [G._load_batch(t, dev) for t in train_tags]
    print(f"loaded {len(batches)} batches in {time.time() - t0:.1f}s; "
          f"sizes: {[(t, int(b['x'].shape[0])) for t, b in zip(train_tags, batches)]}")

    torch.cuda.reset_peak_memory_stats()
    system.train()
    times: list[float] = []
    for step in range(args.steps):
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for b in batches:
            out = system(b["x"], b["edge_index"], b["edge_type"], tau_ceiling=0.7, chem=b.get("gate_chem"))
            loss = (0.1 * sdrp_cross_entropy(out["sdrp_logits"], b["sdrp_target"])) / len(batches)
            loss.backward()
            loss_sum += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(system.parameters(), 1.0)
        opt.step()
        torch.cuda.synchronize()
        dt = time.time() - t0
        times.append(dt)
        print(f"step {step}: {dt:.2f}s loss={loss_sum:.4f} peak_mem={torch.cuda.max_memory_allocated() / 1e9:.2f}GB")

    avg = sum(times[1:]) / max(len(times) - 1, 1) if len(times) > 1 else times[0]
    print(f"\navg step time (excl. first/warmup): {avg:.2f}s")
    for steps in (100, 150, 200, 300, 400):
        per_fold_min = steps * avg / 60
        print(f"  {steps:4d} steps/fold -> {per_fold_min:6.1f} min/fold -> "
              f"{per_fold_min * 12 / 60:6.2f}h for 12 folds -> {per_fold_min * 24 / 60:6.2f}h for 12 folds x 2 arms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
