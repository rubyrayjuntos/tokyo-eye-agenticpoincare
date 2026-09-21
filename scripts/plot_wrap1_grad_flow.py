#!/usr/bin/env python3
"""Plot gradient-flow telemetry from wrap1_zhyp_* fold JSON.

Reads curve_every_10.bucket_grad_l2 + step0_grad from fold stamps and writes
a multi-panel PNG — the wiring view this card is actually diagnosing.

Example:
  PYTHONPATH=. python3 scripts/plot_wrap1_grad_flow.py \\
      --fold-dir data/gates/wrap1_zhyp_m2 \\
      --out data/gates/wrap1_zhyp_m2/grad_flow.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SPINE = ("attn_layers", "projector", "moe", "sdrp_head")
DARK = ("_log_c", "mechanism_head", "evidential_head")
EXTRA = ("euc_skip",)


def _load_folds(fold_dir: Path) -> list[dict]:
    rows = []
    for p in sorted(fold_dir.glob("zhyp_seed*_hold_*.json")):
        d = json.loads(p.read_text())
        curve = d.get("curve_every_10") or []
        steps = [int(r["step"]) for r in curve]
        series = {b: [] for b in SPINE + DARK + EXTRA}
        for r in curve:
            bmap = r.get("bucket_grad_l2") or {}
            for b in series:
                series[b].append(float(bmap.get(b, 0.0)))
        rows.append(
            {
                "path": p.name,
                "hold": d["heldout"][0],
                "G_grad_spine": bool(d.get("G_grad_spine")),
                "step0": d.get("step0_grad") or {},
                "steps": steps,
                "series": series,
            }
        )
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fold-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--hold",
        default=None,
        help="Optional single hold tag (e.g. 1MBN:A); default = first fold",
    )
    args = p.parse_args()

    folds = _load_folds(args.fold_dir)
    if not folds:
        raise SystemExit(f"no fold JSON under {args.fold_dir}")
    if args.hold:
        folds = [f for f in folds if f["hold"] == args.hold]
        if not folds:
            raise SystemExit(f"hold {args.hold!r} not found")
    # overview: step0 heatmap across folds + detail curve for first/selected
    detail = folds[0]

    n_folds = len(folds)
    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1.4, 1.2], hspace=0.45, wspace=0.3)

    # --- step0 heatmap ---
    ax0 = fig.add_subplot(gs[0, :])
    buckets = list(SPINE) + list(DARK)
    mat = np.zeros((n_folds, len(buckets)))
    ylabels = []
    for i, f in enumerate(folds):
        ylabels.append(f"{f['hold']} {'PASS' if f['G_grad_spine'] else 'FAIL'}")
        s0 = (f["step0"].get("grad_l2") or {})
        curve0 = {b: f["series"][b][0] if f["series"][b] else 0.0 for b in buckets}
        for j, b in enumerate(buckets):
            mat[i, j] = float(s0.get(b, curve0.get(b, 0.0)))
    im = ax0.imshow(np.log10(mat + 1e-12), aspect="auto", cmap="viridis")
    ax0.set_xticks(range(len(buckets)))
    ax0.set_xticklabels(buckets, rotation=30, ha="right")
    ax0.set_yticks(range(n_folds))
    ax0.set_yticklabels(ylabels)
    ax0.set_title("Step-0 bucket grad L2 (log10) — expect NZ on spine, ~0 on dark")
    fig.colorbar(im, ax=ax0, fraction=0.02, pad=0.02, label="log10(grad L2)")

    # --- spine curves ---
    ax1 = fig.add_subplot(gs[1, 0])
    for b in SPINE:
        ax1.plot(detail["steps"], detail["series"][b], label=b, linewidth=1.5)
    ax1.set_xlabel("optimizer step")
    ax1.set_ylabel("grad L2")
    ax1.set_title(f"Spine (expect live) — {detail['hold']}")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- dark + euc_skip ---
    ax2 = fig.add_subplot(gs[1, 1])
    for b in EXTRA + DARK:
        ax2.plot(detail["steps"], detail["series"][b], label=b, linewidth=1.5)
    ax2.set_xlabel("optimizer step")
    ax2.set_ylabel("grad L2")
    ax2.set_title(f"Contrast (dark expect 0) — {detail['hold']}")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # --- fold pass strip ---
    ax3 = fig.add_subplot(gs[2, :])
    ax3.axis("off")
    lines = [
        f"folds={n_folds}  detail={detail['hold']}  G_grad_spine="
        f"{'PASS' if detail['G_grad_spine'] else 'FAIL'}",
        f"source={args.fold_dir}",
        "Gate: SPINE_NZ_BUCKETS must be NZ under SDRP-live; "
        "mechanism/evidential/_log_c must be NONE/0 (coeffs omitted + c detached).",
    ]
    ax3.text(0.01, 0.6, "\n".join(lines), family="monospace", fontsize=9, va="center")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
