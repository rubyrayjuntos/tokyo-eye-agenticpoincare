"""Matplotlib plotting tools for the agent.

The agent can generate publication-quality figures from pipeline data.
Plots are saved as image files and returned as file paths for the
frontend to display or for the user to download.

Available plot types:
- poincare_disc: Residues in hyperbolic 2D projection, colored by metric
- uncertainty_profile: Per-residue uncertainty along the sequence
- cone_depth_histogram: Distribution of cone depths
- wt_vs_mutant: Displacement comparison between two structures
- persistence_barcode: Topological persistence diagram
- source_leak_map: Source-leak candidates highlighted on sequence

All plots pull data from the governed layer via the DB adapter.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PLOT_OUTPUT_DIR = Path(os.getenv("PLOT_OUTPUT_DIR", "./data/local_objects/plots"))

# Valid plot types the agent can request
PLOT_TYPES = {
    "poincare_disc",
    "uncertainty_profile",
    "cone_depth_histogram",
    "wt_vs_mutant",
    "persistence_barcode",
    "source_leak_map",
}


@dataclass
class PlotResult:
    """Result of a plot generation."""

    success: bool
    file_path: str | None = None
    plot_type: str = ""
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


async def generate_plot(
    structure_id: str,
    plot_type: str,
    parameters: dict[str, Any] | None = None,
    db: Any = None,
) -> PlotResult:
    """Generate a matplotlib figure from pipeline data.

    The agent calls this tool to create visualizations. The plot type
    determines which renderer is used and what data is queried.

    Args:
        structure_id: Structure to plot data for.
        plot_type: One of PLOT_TYPES.
        parameters: Plot-specific options (colors, thresholds, etc.)
        db: Database adapter for querying governed data.

    Returns:
        PlotResult with the file path to the generated image.
    """
    if db is None:
        return PlotResult(success=False, message="No database connection provided")

    if plot_type not in PLOT_TYPES:
        return PlotResult(
            success=False,
            message=f"Unknown plot type '{plot_type}'. Valid types: {', '.join(sorted(PLOT_TYPES))}",
        )

    params = parameters or {}

    # Dispatch to the appropriate renderer
    renderer = _RENDERERS.get(plot_type)
    if renderer is None:
        return PlotResult(success=False, message=f"Renderer not implemented for '{plot_type}'")

    try:
        result = await asyncio.to_thread(renderer, structure_id, params, db)
        return result
    except Exception as e:
        return PlotResult(success=False, message=f"Plot generation failed: {e}")


# ---------------------------------------------------------------------------
# Renderers (CPU-bound, run in thread)
# ---------------------------------------------------------------------------


def _render_poincare_disc(structure_id: str, params: dict, db: Any) -> PlotResult:
    """Render residues on the Poincaré disc, colored by a metric."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Query data synchronously (we're already in a thread)
    import asyncio
    rows = asyncio.run(_fetch_disc_data(structure_id, db))

    if not rows:
        return PlotResult(success=False, message=f"No embedding data for {structure_id}")

    # Extract coordinates and metric
    metric = params.get("color_by", "cone_depth")
    xs = [r["x"] for r in rows]
    ys = [r["y"] for r in rows]
    colors = [r.get(metric, 0) or 0 for r in rows]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Draw unit disc boundary
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), "k-", linewidth=0.5, alpha=0.3)

    # Scatter residues
    sc = ax.scatter(xs, ys, c=colors, cmap="viridis", s=20, alpha=0.7, edgecolors="none")
    plt.colorbar(sc, ax=ax, label=metric.replace("_", " ").title())

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.set_title(f"Poincaré Disc — {structure_id} (colored by {metric})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    file_path = _save_figure(fig, structure_id, "poincare_disc")
    plt.close(fig)

    return PlotResult(
        success=True,
        file_path=str(file_path),
        plot_type="poincare_disc",
        message=f"Poincaré disc plot saved ({len(rows)} residues, colored by {metric})",
        metadata={"residue_count": len(rows), "color_by": metric},
    )


def _render_uncertainty_profile(structure_id: str, params: dict, db: Any) -> PlotResult:
    """Render per-residue uncertainty along the sequence."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    import asyncio
    rows = asyncio.run(_fetch_uncertainty_data(structure_id, db))

    if not rows:
        return PlotResult(success=False, message=f"No uncertainty data for {structure_id}")

    indices = [r["residue_index"] for r in rows]
    epistemic = [r.get("epistemic_uncertainty") or 0 for r in rows]
    aleatoric = [r.get("aleatoric_uncertainty") or 0 for r in rows]

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.fill_between(indices, 0, epistemic, alpha=0.6, label="Epistemic", color="#ff6b6b")
    ax.fill_between(indices, 0, aleatoric, alpha=0.4, label="Aleatoric", color="#4ecdc4")
    ax.set_xlabel("Residue Index")
    ax.set_ylabel("Uncertainty")
    ax.set_title(f"Uncertainty Profile — {structure_id}")
    ax.legend()

    # Mark threshold
    threshold = params.get("threshold", 0.3)
    ax.axhline(y=threshold, color="red", linestyle="--", alpha=0.5, label=f"Threshold ({threshold})")

    file_path = _save_figure(fig, structure_id, "uncertainty_profile")
    plt.close(fig)

    return PlotResult(
        success=True,
        file_path=str(file_path),
        plot_type="uncertainty_profile",
        message=f"Uncertainty profile saved ({len(rows)} residues)",
        metadata={"residue_count": len(rows)},
    )


def _render_cone_depth_histogram(structure_id: str, params: dict, db: Any) -> PlotResult:
    """Render distribution of cone depths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import asyncio
    rows = asyncio.run(_fetch_uncertainty_data(structure_id, db))

    if not rows:
        return PlotResult(success=False, message=f"No data for {structure_id}")

    depths = [r.get("cone_depth") or 0 for r in rows]
    bins = params.get("bins", 30)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.hist(depths, bins=bins, color="#6c5ce7", alpha=0.7, edgecolor="white")
    ax.set_xlabel("Cone Depth")
    ax.set_ylabel("Count")
    ax.set_title(f"Cone Depth Distribution — {structure_id}")
    ax.axvline(x=1.5, color="red", linestyle="--", alpha=0.5, label="Source-leak threshold")
    ax.legend()

    file_path = _save_figure(fig, structure_id, "cone_depth_histogram")
    plt.close(fig)

    return PlotResult(
        success=True,
        file_path=str(file_path),
        plot_type="cone_depth_histogram",
        message=f"Cone depth histogram saved ({len(depths)} residues)",
        metadata={"residue_count": len(depths), "bins": bins},
    )


def _render_wt_vs_mutant(structure_id: str, params: dict, db: Any) -> PlotResult:
    """Render WT vs mutant displacement comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mutant_id = params.get("mutant_structure_id")
    if not mutant_id:
        return PlotResult(success=False, message="mutant_structure_id required in parameters")

    import asyncio
    wt_rows = asyncio.run(_fetch_uncertainty_data(structure_id, db))
    mut_rows = asyncio.run(_fetch_uncertainty_data(mutant_id, db))

    if not wt_rows or not mut_rows:
        return PlotResult(success=False, message="Missing data for one or both structures")

    # Match by residue_index
    wt_by_idx = {r["residue_index"]: r for r in wt_rows}
    mut_by_idx = {r["residue_index"]: r for r in mut_rows}

    common = sorted(set(wt_by_idx.keys()) & set(mut_by_idx.keys()))
    displacements = [
        abs((mut_by_idx[i].get("cone_depth") or 0) - (wt_by_idx[i].get("cone_depth") or 0))
        for i in common
    ]

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.bar(common, displacements, color="#ff00ff", alpha=0.7, width=1.0)
    ax.set_xlabel("Residue Index")
    ax.set_ylabel("|Δ Cone Depth|")
    ax.set_title(f"WT vs Mutant Displacement — {structure_id} vs {mutant_id}")

    file_path = _save_figure(fig, structure_id, "wt_vs_mutant")
    plt.close(fig)

    return PlotResult(
        success=True,
        file_path=str(file_path),
        plot_type="wt_vs_mutant",
        message=f"WT vs mutant plot saved ({len(common)} matched residues)",
        metadata={"matched_residues": len(common), "mutant_id": mutant_id},
    )


def _render_persistence_barcode(structure_id: str, params: dict, db: Any) -> PlotResult:
    """Render persistence barcode diagram."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import asyncio
    rows = asyncio.run(_fetch_persistence_data(structure_id, db))

    if not rows:
        return PlotResult(success=False, message=f"No persistence data for {structure_id}")

    # Extract barcodes from persistence_data JSON
    barcodes = []
    for r in rows:
        pd = r.get("persistence_data") or {}
        for b in pd.get("barcodes", []):
            barcodes.append((b.get("birth", 0), b.get("death", 0)))

    if not barcodes:
        return PlotResult(success=False, message="No barcodes found in persistence data")

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for i, (birth, death) in enumerate(sorted(barcodes, key=lambda x: x[1] - x[0], reverse=True)):
        ax.barh(i, death - birth, left=birth, height=0.8, color="#00b894", alpha=0.7)

    ax.set_xlabel("Filtration Value")
    ax.set_ylabel("Feature Index")
    ax.set_title(f"Persistence Barcode — {structure_id}")

    file_path = _save_figure(fig, structure_id, "persistence_barcode")
    plt.close(fig)

    return PlotResult(
        success=True,
        file_path=str(file_path),
        plot_type="persistence_barcode",
        message=f"Persistence barcode saved ({len(barcodes)} features)",
        metadata={"barcode_count": len(barcodes)},
    )


def _render_source_leak_map(structure_id: str, params: dict, db: Any) -> PlotResult:
    """Render source-leak candidates highlighted on the sequence."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    threshold = params.get("uncertainty_threshold", 0.3)
    min_depth = params.get("min_depth", 1.5)

    import asyncio
    rows = asyncio.run(_fetch_uncertainty_data(structure_id, db))

    if not rows:
        return PlotResult(success=False, message=f"No data for {structure_id}")

    indices = [r["residue_index"] for r in rows]
    depths = [r.get("cone_depth") or 0 for r in rows]
    uncertainties = [r.get("epistemic_uncertainty") or 0 for r in rows]

    # Identify source leaks
    is_leak = [
        (u >= threshold and d >= min_depth)
        for u, d in zip(uncertainties, depths)
    ]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Top: cone depth with leaks highlighted
    colors = ["#ff4444" if leak else "#aaaaaa" for leak in is_leak]
    ax1.bar(indices, depths, color=colors, width=1.0, alpha=0.7)
    ax1.axhline(y=min_depth, color="red", linestyle="--", alpha=0.5)
    ax1.set_ylabel("Cone Depth")
    ax1.set_title(f"Source Leak Map — {structure_id}")

    # Bottom: epistemic uncertainty
    colors2 = ["#ff4444" if leak else "#4ecdc4" for leak in is_leak]
    ax2.bar(indices, uncertainties, color=colors2, width=1.0, alpha=0.7)
    ax2.axhline(y=threshold, color="red", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Residue Index")
    ax2.set_ylabel("Epistemic Uncertainty")

    leak_count = sum(is_leak)
    fig.suptitle(f"{leak_count} source-leak candidates (red)", fontsize=10, y=0.02)
    plt.tight_layout()

    file_path = _save_figure(fig, structure_id, "source_leak_map")
    plt.close(fig)

    return PlotResult(
        success=True,
        file_path=str(file_path),
        plot_type="source_leak_map",
        message=f"Source leak map saved ({leak_count} leaks out of {len(rows)} residues)",
        metadata={"leak_count": leak_count, "total_residues": len(rows)},
    )


# ---------------------------------------------------------------------------
# Data fetchers (async, called from within threads via asyncio.run)
# ---------------------------------------------------------------------------


async def _fetch_disc_data(structure_id: str, db: Any) -> list[dict]:
    """Fetch 2D projection data for the Poincaré disc plot."""
    rows = await db.fetch_all(
        """
        SELECT r.residue_id, r.residue_index, r.residue_name,
               e.hyp_projections, e.cone_depth, e.cone_width,
               e.epistemic_uncertainty, e.aleatoric_uncertainty
        FROM fact_gnn_node_embedding e
        JOIN dim_residue r ON r.residue_id = e.residue_id
        JOIN embedding_space es ON es.space_id = e.space_id
        WHERE e.structure_id = :structure_id
          AND es.space_type = 'hyperbolic'
        ORDER BY r.residue_index
        """,
        {"structure_id": structure_id},
    )

    result = []
    for r in rows:
        x, y = 0.0, 0.0
        proj = r.get("hyp_projections")
        if isinstance(proj, list) and len(proj) >= 2:
            x, y = float(proj[0]), float(proj[1])
        result.append({**r, "x": x, "y": y})
    return result


async def _fetch_uncertainty_data(structure_id: str, db: Any) -> list[dict]:
    """Fetch per-residue uncertainty and depth data."""
    return await db.fetch_all(
        """
        SELECT r.residue_index, r.residue_name,
               e.cone_depth, e.epistemic_uncertainty, e.aleatoric_uncertainty,
               e.total_uncertainty
        FROM fact_gnn_node_embedding e
        JOIN dim_residue r ON r.residue_id = e.residue_id
        JOIN embedding_space es ON es.space_id = e.space_id
        WHERE e.structure_id = :structure_id
          AND es.space_type = 'hyperbolic'
        ORDER BY r.residue_index
        """,
        {"structure_id": structure_id},
    )


async def _fetch_persistence_data(structure_id: str, db: Any) -> list[dict]:
    """Fetch Phase 3 persistence data."""
    return await db.fetch_all(
        """
        SELECT phase3_id, persistence_data, barcode_length, max_alpha
        FROM fact_phase3_persistence
        WHERE structure_id = :structure_id
          AND residue_id IS NULL
        ORDER BY phase3_id
        """,
        {"structure_id": structure_id},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_figure(fig: Any, structure_id: str, plot_type: str) -> Path:
    """Save a matplotlib figure to the output directory."""
    PLOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{structure_id}_{plot_type}_{uuid.uuid4().hex[:8]}.png"
    path = PLOT_OUTPUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    return path


# ---------------------------------------------------------------------------
# Renderer registry
# ---------------------------------------------------------------------------

_RENDERERS: dict[str, Any] = {
    "poincare_disc": _render_poincare_disc,
    "uncertainty_profile": _render_uncertainty_profile,
    "cone_depth_histogram": _render_cone_depth_histogram,
    "wt_vs_mutant": _render_wt_vs_mutant,
    "persistence_barcode": _render_persistence_barcode,
    "source_leak_map": _render_source_leak_map,
}
