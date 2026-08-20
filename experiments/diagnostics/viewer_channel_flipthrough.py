"""Per-residue channel flip-through for a GNN split-viewer HTML (or fresh inference).

Pulls the seven Color-by channels from a regenerated ``*_split_viewer.html``
POINTS payload and prints:

1. Univariate summaries + min-max color-bin fractions (blue / mid / red)
2. Focus pairwise Pearson + Spearman (depth↔r, depth↔τ, ρ↔τ, heads↔ρ, …)
3. Full pairwise Pearson matrix

This is the cheap check that separates render mismatches from real geometry —
the same discipline that caught the disc↔NGL colormap invert and the τ-vs-depth
majority-color paradox on 4OBE.

Usage::

    python -m experiments.diagnostics.viewer_channel_flipthrough \\
        checkpoints/v66/runs/.../viewers/4obe/4obe_split_viewer.html

    python -m experiments.diagnostics.viewer_channel_flipthrough \\
        --html .../4obe_split_viewer.html --json-out /tmp/4obe_channels.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

# Viewer Color-by keys (exclude expert bands — discrete route index).
CHANNELS = (
    "r",
    "cone_depth",
    "tau",
    "rho",
    "physics_investigation",
    "investigation",
    "aleatoric",
    "epistemic",
)

FOCUS_PAIRS = (
    ("cone_depth", "r"),
    ("cone_depth", "tau"),
    ("r", "tau"),
    ("rho", "tau"),
    ("rho", "cone_depth"),
    ("rho", "r"),
    ("physics_investigation", "tau"),
    ("physics_investigation", "rho"),
    ("investigation", "rho"),
    ("investigation", "tau"),
    ("investigation", "cone_depth"),
    ("aleatoric", "rho"),
    ("aleatoric", "tau"),
    ("aleatoric", "cone_depth"),
    ("epistemic", "rho"),
    ("epistemic", "tau"),
    ("epistemic", "cone_depth"),
    ("aleatoric", "epistemic"),
    ("investigation", "aleatoric"),
    ("investigation", "epistemic"),
)


def load_points_from_split_html(html_path: Path) -> list[dict[str, Any]]:
    text = html_path.read_text(encoding="utf-8")
    marker = "var POINTS = "
    idx = text.find(marker)
    if idx < 0:
        raise ValueError(f"POINTS payload not found in {html_path}")
    points, _ = json.JSONDecoder().raw_decode(text, idx + len(marker))
    if not points:
        raise ValueError(f"empty POINTS in {html_path}")
    return points


def channel_arrays(points: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key in CHANNELS:
        if key not in points[0] and key == "physics_investigation":
            # Older viewers may lack the physics default.
            continue
        if key not in points[0]:
            continue
        out[key] = np.asarray([float(p.get(key) or 0.0) for p in points], dtype=np.float64)
    return out


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or a.std() < 1e-15 or b.std() < 1e-15:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or a.std() < 1e-15 or b.std() < 1e-15:
        return float("nan")
    from scipy.stats import spearmanr

    return float(spearmanr(a, b).correlation)


def _minmax(v: np.ndarray) -> np.ndarray:
    return (v - v.min()) / (v.max() - v.min() + 1e-12)


def color_bin_fractions(v: np.ndarray) -> dict[str, float]:
    t = _minmax(v)
    return {
        "blue_lt_0.33": float((t < 0.33).mean()),
        "mid": float(((t >= 0.33) & (t <= 0.66)).mean()),
        "red_gt_0.66": float((t > 0.66).mean()),
    }


def analyze(points: list[dict[str, Any]]) -> dict[str, Any]:
    chans = channel_arrays(points)
    uni: dict[str, Any] = {}
    for k, v in chans.items():
        bins = color_bin_fractions(v)
        uni[k] = {
            "n": int(v.size),
            "mean": float(v.mean()),
            "std": float(v.std()),
            "min": float(v.min()),
            "max": float(v.max()),
            **bins,
        }

    focus: list[dict[str, Any]] = []
    for a, b in FOCUS_PAIRS:
        if a not in chans or b not in chans:
            continue
        focus.append(
            {
                "a": a,
                "b": b,
                "pearson": _pearson(chans[a], chans[b]),
                "spearman": _spearman(chans[a], chans[b]),
            }
        )

    names = list(chans.keys())
    matrix = [[_pearson(chans[i], chans[j]) for j in names] for i in names]

    tau_block: dict[str, Any] | None = None
    if "tau" in chans:
        tau = chans["tau"]
        tau_block = {"tau1_frac": float(tau.mean()), "n_tau1": int(tau.sum())}
        if "rho" in chans:
            tau_block["rho_mean_tau1"] = float(chans["rho"][tau == 1].mean()) if tau.any() else None
            tau_block["rho_mean_tau0"] = float(chans["rho"][tau == 0].mean()) if (1 - tau).any() else None
        if "cone_depth" in chans:
            d = chans["cone_depth"]
            dt = _minmax(d)
            tau_block["depth_mean_tau1"] = float(d[tau == 1].mean()) if tau.any() else None
            tau_block["depth_mean_tau0"] = float(d[tau == 0].mean()) if (1 - tau).any() else None
            if tau.any():
                tau_block["frac_tau1_depth_red"] = float((dt[tau == 1] > 0.66).mean())
                tau_block["frac_tau1_depth_blue"] = float((dt[tau == 1] < 0.33).mean())
                tau_block["mean_t_depth_tau1"] = float(dt[tau == 1].mean())
                tau_block["mean_t_depth_tau0"] = (
                    float(dt[tau == 0].mean()) if (1 - tau).any() else None
                )

    return {
        "n_residues": len(points),
        "channels": uni,
        "focus_pairs": focus,
        "pearson_matrix": {"names": names, "values": matrix},
        "tau_vs_depth_visual": tau_block,
        "note": (
            "Binary τ uses the same continuous RdYlBu+reverse path as depth "
            "(τ=0→blue, τ=1→red at full strength). Majority-color disagreement "
            "with continuous depth is expected under min-max when τ=1 is common "
            "but depth has a long right tail — check focus_pairs, not just the picture."
        ),
    }


def format_report(report: dict[str, Any], *, source: str) -> str:
    lines: list[str] = []
    lines.append(f"# viewer channel flip-through — {source}")
    lines.append(f"n_residues={report['n_residues']}")
    lines.append("")
    lines.append("## univariate + min-max color bins (blue <0.33 / mid / red >0.66)")
    lines.append(
        f"{'channel':22s} {'mean':>8s} {'std':>8s} {'blue':>6s} {'mid':>6s} {'red':>6s}"
    )
    for k, u in report["channels"].items():
        lines.append(
            f"{k:22s} {u['mean']:8.4f} {u['std']:8.4f} "
            f"{u['blue_lt_0.33']:6.2f} {u['mid']:6.2f} {u['red_gt_0.66']:6.2f}"
        )
    lines.append("")
    lines.append("## focus pairwise correlations")
    lines.append(f"{'pair':40s}  {'pearson':>8s}  {'spearman':>8s}")
    for row in report["focus_pairs"]:
        lines.append(
            f"{row['a'] + ' vs ' + row['b']:40s}  "
            f"{row['pearson']:+8.3f}  {row['spearman']:+8.3f}"
        )
    tb = report.get("tau_vs_depth_visual")
    if tb:
        lines.append("")
        lines.append("## τ vs depth visual paradox check")
        lines.append(
            f"tau1_frac={tb['tau1_frac']:.3f}  "
            f"depth_mean τ1/τ0={tb.get('depth_mean_tau1')} / {tb.get('depth_mean_tau0')}"
        )
        if "frac_tau1_depth_red" in tb:
            lines.append(
                f"among τ=1: frac depth-red={tb['frac_tau1_depth_red']:.3f}  "
                f"frac depth-blue={tb['frac_tau1_depth_blue']:.3f}  "
                f"mean t_depth={tb['mean_t_depth_tau1']:.3f}"
            )
        if tb.get("rho_mean_tau1") is not None:
            lines.append(
                f"rho_mean τ1/τ0={tb['rho_mean_tau1']:.2f} / {tb['rho_mean_tau0']:.2f}"
            )
    names = report["pearson_matrix"]["names"]
    mat = report["pearson_matrix"]["values"]
    lines.append("")
    lines.append("## full Pearson matrix")
    header = " ".join(f"{n[:10]:>10s}" for n in names)
    lines.append(f"{'':10s} {header}")
    for i, n in enumerate(names):
        row = " ".join(f"{mat[i][j]:+10.3f}" for j in range(len(names)))
        lines.append(f"{n[:10]:10s} {row}")
    lines.append("")
    lines.append(report["note"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "html",
        type=Path,
        nargs="?",
        help="Path to *_split_viewer.html",
    )
    parser.add_argument("--html", dest="html_opt", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    html_path = args.html_opt or args.html
    if html_path is None:
        parser.error("provide split-viewer HTML path")

    points = load_points_from_split_html(html_path)
    report = analyze(points)
    print(format_report(report, source=str(html_path)))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nJSON → {args.json_out}")


if __name__ == "__main__":
    main()
