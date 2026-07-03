"""
visualize_shell.py — Matplotlib dashboard for v6 checkpoint shell geometry.

Aggregate + per-protein panels with |proj| / depth histograms.

Usage:
    python -m experiments.training.v6.visualize_shell \\
        --checkpoint checkpoints/v6/runs/shell_no_warm_b1/v6_best.pt

    python -m experiments.training.v6.visualize_shell \\
        --checkpoint .../shell_no_warm_b1/v6_best.pt \\
        --compare .../shell_no_warm_b2/v6_best.pt \\
        --per-protein-panels 6
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.training.v6.assess_checkpoint import (
    collect_inference_bundle,
    collect_protein_bundle,
    load_v6_model,
    moe_metrics,
)
from experiments.training.v6.corpus import load_training_proteins
from experiments.training.v6.train_loop import measure_geometry_health

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("viz_shell")

AGGREGATE_ROWS = 4


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 3:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _disc_radius(hyp2d: np.ndarray) -> np.ndarray:
    return np.linalg.norm(hyp2d, axis=1) if hyp2d.ndim == 2 else np.abs(hyp2d)


def _draw_unit_disc(ax: plt.Axes) -> None:
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), "k-", linewidth=0.6, alpha=0.35, zorder=0)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")


def _scatter_disc(
    ax: plt.Axes,
    hyp2d: np.ndarray,
    values: np.ndarray,
    *,
    title: str,
    cmap: str = "viridis",
    label: str = "",
    point_size: float = 8,
) -> None:
    _draw_unit_disc(ax)
    sc = ax.scatter(
        hyp2d[:, 0],
        hyp2d[:, 1],
        c=values,
        cmap=cmap,
        s=point_size,
        alpha=0.75,
        edgecolors="none",
    )
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=label)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("disc x")
    ax.set_ylabel("disc y")


def _scatter_xy(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    color: np.ndarray | None = None,
    point_size: float = 6,
) -> None:
    r = _pearson(x, y)
    kw: dict[str, Any] = {"s": point_size, "alpha": 0.45, "edgecolors": "none"}
    if color is not None:
        sc = ax.scatter(x, y, c=color, cmap="plasma", **kw)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="|proj|")
    else:
        ax.scatter(x, y, c="#6c5ce7", **kw)
    ax.set_title(f"{title}\nr = {r:.3f}", fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def _hist(ax: plt.Axes, values: np.ndarray, *, title: str, xlabel: str, color: str) -> None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        ax.set_title(f"{title}\n(no data)")
        return
    ax.hist(finite, bins=30, color=color, alpha=0.75, edgecolor="white")
    ax.axvline(float(np.median(finite)), color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_title(
        f"{title}\nmed={np.median(finite):.4f} std={np.std(finite):.4f}",
        fontsize=9,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")


def _stats_text(health: dict[str, float], moe: dict[str, Any], label: str) -> str:
    depth_std = health.get("depth_std", health.get("cone_depth_std_mean", float("nan")))
    disc_std = health.get("disc_r_std", health.get("disc_r_std_mean", float("nan")))
    r_disc_sasa = health.get("probe_r_disc_sasa", float("nan"))
    return "\n".join(
        [
            label,
            f"cone_rng {health.get('cone_range_mean', 0):.4f}",
            f"depth_std {depth_std:.4f}",
            f"disc_std {disc_std:.4f}",
            f"r(d,s)   {health.get('probe_r_depth_sasa', float('nan')):.3f}",
            f"r(e,s)   {health.get('probe_r_epi_sasa', float('nan')):.3f}",
            f"r(|p|,d) {health.get('probe_r_proj_depth', float('nan')):.3f}",
            f"r(|p|,s) {r_disc_sasa:.3f}",
            f"|proj| med {health.get('disc_r_median', float('nan')):.4f}",
            f"route_H  {moe.get('routing_entropy_mean', 0):.3f}",
        ]
    )


def _bundle_disc_stats(bundle: dict[str, Any]) -> dict[str, float]:
    disc_r = _disc_radius(bundle["hyp_projections_2d"])
    return {
        "disc_r_median": float(np.median(disc_r)),
        "disc_r_std": float(np.std(disc_r)),
        "depth_std": float(np.std(bundle["cone_depth"])),
    }


def _plot_aggregate_column(
    fig: plt.Figure,
    col: int,
    ncols: int,
    bundle: dict[str, Any],
    health: dict[str, float],
    moe: dict[str, Any],
    title: str,
) -> None:
    hyp = bundle["hyp_projections_2d"]
    depth = bundle["cone_depth"]
    sasa = bundle["sasa"]
    epi = bundle["epistemic"]
    disc_r = _disc_radius(hyp)
    health = {**health, **_bundle_disc_stats(bundle)}

    axes = [
        fig.add_subplot(AGGREGATE_ROWS, ncols, row * ncols + col + 1)
        for row in range(AGGREGATE_ROWS)
    ]

    _scatter_disc(
        axes[0],
        hyp,
        depth,
        title=f"{title}\ndisc × cone_depth",
        label="cone_depth",
    )
    _scatter_disc(
        axes[1],
        hyp,
        sasa,
        title="disc × SASA",
        cmap="YlOrRd",
        label="SASA",
    )
    _scatter_xy(axes[2], sasa, depth, title="cone_depth vs SASA", xlabel="SASA", ylabel="cone_depth")

    _hist(axes[3], disc_r, title="|hyp_proj_2d| histogram", xlabel="|proj|", color="#0984e3")

    # Extra panels when comparing single checkpoint: use inset on row 2 via second column... 
    # For multi-col compare, add depth hist + proj vs depth on row 4 split - use twinx on axes[2]
    ax_extra = axes[2].inset_axes([0.55, 0.08, 0.42, 0.42])
    _scatter_xy(
        ax_extra,
        disc_r,
        depth,
        title="|proj| vs depth",
        xlabel="|proj|",
        ylabel="depth",
        point_size=3,
    )
    ax_extra.tick_params(labelsize=6)
    ax_extra.set_title(f"|proj| vs depth\nr={_pearson(disc_r, depth):.3f}", fontsize=7)

    axes[0].text(
        0.02,
        0.02,
        _stats_text(health, moe, title),
        transform=axes[0].transAxes,
        va="bottom",
        fontsize=7,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    axes[1].text(
        0.02,
        0.98,
        f"r(epi,sasa)={_pearson(epi, sasa):.3f}",
        transform=axes[1].transAxes,
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )


def _plot_per_protein_figure(
    model: Any,
    proteins: list[dict[str, Any]],
    device: str,
    output: Path,
    *,
    max_panels: int,
    title: str,
) -> None:
    n = min(max_panels, len(proteins))
    if n <= 0:
        return
    cols = min(4, n)
    rows = math.ceil(n / cols)
    fig = plt.figure(figsize=(3.8 * cols, 3.4 * rows * 2))
    fig.suptitle(title, fontsize=11)

    for i, prot in enumerate(proteins[:n]):
        bundle = collect_protein_bundle(model, prot, device)
        pdb_id = bundle["pdb_id"]
        hyp = bundle["hyp_projections_2d"]
        depth = bundle["cone_depth"]
        sasa = bundle["sasa"]
        r_ds = _pearson(sasa, depth)
        row, col = divmod(i, cols)
        ax_disc = fig.add_subplot(rows * 2, cols, row * cols + col + 1)
        ax_sc = fig.add_subplot(rows * 2, cols, (rows + row) * cols + col + 1)
        _scatter_disc(
            ax_disc,
            hyp,
            depth,
            title=f"{pdb_id} n={bundle['n_residues']}",
            label="depth",
            point_size=14,
        )
        _scatter_xy(
            ax_sc,
            sasa,
            depth,
            title="depth vs SASA",
            xlabel="SASA",
            ylabel="depth",
            point_size=10,
        )
        ax_sc.text(0.02, 0.98, f"r={r_ds:.3f}", transform=ax_sc.transAxes, va="top", fontsize=8)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote per-protein %s", output)


def run_bundle(
    checkpoint: Path,
    proteins: list[dict[str, Any]],
    device: str,
) -> tuple[Any, dict[str, Any], dict[str, float], dict[str, Any]]:
    model = load_v6_model(checkpoint, device)
    health = measure_geometry_health(model, proteins, device)
    bundle = collect_inference_bundle(model, proteins, device)
    moe = moe_metrics(bundle)
    return model, bundle, health, moe


def visualize_checkpoints(
    checkpoints: list[tuple[Path, str]],
    proteins: list[dict[str, Any]],
    *,
    device: str,
    output: Path,
    per_protein_panels: int = 0,
) -> Path:
    ncols = len(checkpoints)
    n_res = sum(len(p["data"].x) for p in proteins)
    fig = plt.figure(figsize=(6.5 * ncols, 4.2 * AGGREGATE_ROWS))
    fig.suptitle(
        f"V6 shell geometry — {len(proteins)} proteins, {n_res} residues",
        fontsize=12,
        y=0.995,
    )

    primary_model = None
    primary_label = checkpoints[0][1]
    for col, (ckpt, label) in enumerate(checkpoints):
        logger.info("Rendering aggregate %s from %s", label, ckpt)
        model, bundle, health, moe = run_bundle(ckpt, proteins, device)
        if col == 0:
            primary_model = model
        _plot_aggregate_column(fig, col, ncols, bundle, health, moe, label)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", output)

    if per_protein_panels > 0 and primary_model is not None and ncols == 1:
        per_out = output.with_name(output.stem + "_per_protein.png")
        _plot_per_protein_figure(
            primary_model,
            proteins,
            device,
            per_out,
            max_panels=per_protein_panels,
            title=f"Per-protein shell — {primary_label}",
        )

    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Matplotlib shell geometry dashboard for v6 checkpoints")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--compare", type=Path, default=None)
    parser.add_argument("--corpus", type=Path, default=Path("manifests/v6_corpus_benchmark.json"))
    parser.add_argument("--pdb-dir", type=Path, default=Path("science/dtie/assets/benchmark_pdbs"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-proteins", type=int, default=8)
    parser.add_argument("--max-residues", type=int, default=600)
    parser.add_argument("--per-protein-panels", type=int, default=4, help="0 to skip per-protein grid")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--compare-label", default=None)
    args = parser.parse_args()

    proteins, failed = load_training_proteins(
        args.pdb_dir,
        args.corpus,
        max_proteins=args.max_proteins,
        max_residues=args.max_residues,
    )
    if not proteins:
        raise SystemExit(f"No proteins loaded ({failed} failed)")

    primary_label = args.label or args.checkpoint.parent.name
    checkpoints: list[tuple[Path, str]] = [(args.checkpoint, primary_label)]
    if args.compare is not None:
        checkpoints.append((args.compare, args.compare_label or args.compare.parent.name))

    per_panels = args.per_protein_panels if len(checkpoints) == 1 else 0
    output = args.output or args.checkpoint.parent / (
        "shell_compare.png" if args.compare else f"shell_{primary_label}.png"
    )

    visualize_checkpoints(
        checkpoints,
        proteins,
        device=args.device,
        output=output,
        per_protein_panels=per_panels,
    )


if __name__ == "__main__":
    main()
