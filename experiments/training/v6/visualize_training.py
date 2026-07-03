"""
visualize_training.py — Training trajectories + epoch filmstrip from metrics.json.

Usage:
    python -m experiments.training.v6.visualize_training \\
        --run-dir checkpoints/v6/runs/shell_no_warm_p2

    python -m experiments.training.v6.visualize_training \\
        --run-dir checkpoints/v6/runs/shell_no_warm_p2 \\
        --filmstrip-proteins 1CSP,1CRN
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.training.v6.assess_checkpoint import collect_protein_bundle, load_v6_model
from experiments.training.v6.corpus import load_training_proteins
from experiments.training.v6.visualize_shell import _pearson, _scatter_xy

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("viz_training")


def load_metrics(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"No metrics.json in {run_dir}")
    return json.loads(path.read_text())


def _series(metrics: list[dict[str, Any]], *keys: str, source: str = "health") -> tuple[list[int], list[float]]:
    epochs: list[int] = []
    values: list[float] = []
    for entry in metrics:
        blob = entry.get(source, entry)
        val: Any = blob
        for key in keys:
            val = val.get(key, float("nan")) if isinstance(val, dict) else float("nan")
        try:
            v = float(val)
        except (TypeError, ValueError):
            v = float("nan")
        if np.isnan(v):
            continue
        epochs.append(int(entry["global_epoch"]))
        values.append(v)
    return epochs, values


def discover_filmstrip_checkpoints(run_dir: Path) -> list[tuple[int, Path, str]]:
    """Return (global_epoch, path, label) sorted by epoch."""
    found: list[tuple[int, Path, str]] = []

    epoch_dir = run_dir / "epochs"
    if epoch_dir.is_dir():
        for p in sorted(epoch_dir.glob("epoch_*.pt")):
            m = re.search(r"epoch_(\d+)", p.name)
            if m:
                ep = int(m.group(1))
                found.append((ep, p, f"ep{ep}"))

    best = run_dir / "v6_best.pt"
    if best.is_file():
        found.append((10_000, best, "v6_best"))

    for p in sorted(run_dir.glob("v6_phase*.pt")):
        found.append((20_000 + len(found), p, p.stem))

    found.sort(key=lambda x: x[0])
    # Deduplicate by path
    seen: set[str] = set()
    unique: list[tuple[int, Path, str]] = []
    for item in found:
        key = str(item[1])
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def plot_training_curves(metrics: list[dict[str, Any]], output: Path, run_name: str) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    fig.suptitle(f"Training trajectories — {run_name}", fontsize=12)

    plots = [
        (axes[0, 0], ["total"], "losses", "Total loss", "#2d3436"),
        (axes[0, 1], ["cone_range_mean"], "health", "cone_range_mean", "#6c5ce7"),
        (axes[0, 2], ["probe_r_depth_sasa"], "health", "r(depth, SASA)", "#00b894"),
        (axes[1, 0], ["probe_r_epi_sasa"], "health", "r(epi, SASA)", "#0984e3"),
        (axes[1, 1], ["routing_entropy"], "losses", "routing entropy", "#e17055"),
        (axes[1, 2], [], "score", "checkpoint score", "#fdcb6e"),
        (axes[2, 0], ["cone_depth_std_mean"], "health", "cone_depth_std", "#a29bfe"),
        (axes[2, 1], ["disc_r_std_mean"], "health", "disc_r_std", "#55efc4"),
        (axes[2, 2], ["probe_r_disc_sasa"], "health", "r(|proj|, SASA)", "#ffeaa7"),
    ]

    eligible_epochs = {
        int(e["global_epoch"]) for e in metrics if e.get("checkpoint_eligible") is True
    }

    for ax, keys, source, ylabel, color in plots:
        if source == "score":
            xs = [int(e["global_epoch"]) for e in metrics if "score" in e]
            ys = [float(e["score"]) for e in metrics if "score" in e]
        else:
            xs, ys = _series(metrics, *keys, source=source)
        ax.plot(xs, ys, color=color, linewidth=1.5, marker="o", markersize=4)
        for ex in eligible_epochs:
            if ex in xs:
                idx = xs.index(ex)
                ax.scatter([ex], [ys[idx]], s=80, facecolors="none", edgecolors="lime", linewidths=2, zorder=5)
        if ylabel in ("r(depth, SASA)", "r(|proj|, SASA)"):
            ax.axhline(0.0, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
        if ylabel == "r(depth, SASA)":
            ax.axhline(0.4, color="green", linestyle=":", alpha=0.5, linewidth=0.8)
        if ylabel == "disc_r_std":
            ax.axhline(0.15, color="green", linestyle=":", alpha=0.5, linewidth=0.8, label="spread floor")
        ax.set_xlabel("global epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote curves %s", output)


def plot_filmstrip(
    run_dir: Path,
    proteins: list[dict[str, Any]],
    *,
    device: str,
    output: Path,
    max_frames: int = 12,
) -> None:
    checkpoints = discover_filmstrip_checkpoints(run_dir)[:max_frames]
    if not checkpoints:
        logger.warning("No checkpoints for filmstrip in %s", run_dir)
        return

    n = len(checkpoints)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.2 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle(f"Epoch filmstrip — depth vs SASA ({run_dir.name})", fontsize=11)

    prot = proteins[0]
    pdb_id = prot.get("pdb_id", "?")

    for i, (epoch, ckpt, label) in enumerate(checkpoints):
        row, col = divmod(i, cols)
        ax = axes[row, col]
        model = load_v6_model(ckpt, device)
        bundle = collect_protein_bundle(model, prot, device)
        sasa = bundle["sasa"]
        depth = bundle["cone_depth"]
        r_ds = _pearson(sasa, depth)
        _scatter_xy(
            ax,
            sasa,
            depth,
            title=f"{label} ({pdb_id})",
            xlabel="SASA",
            ylabel="depth",
            point_size=10,
        )
        ax.text(0.02, 0.98, f"r={r_ds:.3f}", transform=ax.transAxes, va="top", fontsize=8)

    for j in range(n, rows * cols):
        row, col = divmod(j, cols)
        axes[row, col].axis("off")

    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote filmstrip %s (%d frames, protein %s)", output, n, pdb_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="V6 training metrics curves + epoch filmstrip")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=Path("manifests/v6_corpus_benchmark.json"))
    parser.add_argument("--pdb-dir", type=Path, default=Path("science/dtie/assets/benchmark_pdbs"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--filmstrip-protein", default=None, help="PDB id for filmstrip (default: first loaded)")
    parser.add_argument("--max-filmstrip-frames", type=int, default=12)
    parser.add_argument("--output-curves", type=Path, default=None)
    parser.add_argument("--output-filmstrip", type=Path, default=None)
    parser.add_argument("--no-filmstrip", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir
    metrics = load_metrics(run_dir)
    run_name = run_dir.name

    curves_out = args.output_curves or run_dir / "training_curves.png"
    plot_training_curves(metrics, curves_out, run_name)

    if args.no_filmstrip:
        return

    proteins, _ = load_training_proteins(args.pdb_dir, args.corpus, max_proteins=16)
    if args.filmstrip_protein:
        proteins = [p for p in proteins if p.get("pdb_id") == args.filmstrip_protein.upper()] or proteins[:1]
    if not proteins:
        logger.warning("No proteins for filmstrip")
        return

    filmstrip_out = args.output_filmstrip or run_dir / "epoch_filmstrip.png"
    plot_filmstrip(
        run_dir,
        proteins[:1],
        device=args.device,
        output=filmstrip_out,
        max_frames=args.max_filmstrip_frames,
    )


if __name__ == "__main__":
    main()
