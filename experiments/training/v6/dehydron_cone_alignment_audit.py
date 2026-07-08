"""Dehydron–cone alignment audit — does learned depth follow dehydron theory or SASA shell?

Compares per-residue MASTER features (ρ, τ, SASA from DB) against model outputs
(cone_depth, radial_depth, disc_r) without using GNN-derived leak/motif tables.

Theory (Tokyo Eye premise):
  - Dehydrons (τ=1, low ρ) → funnel rim → high hyperbolic depth / disc radius
  - Well-wrapped stable core → center → low depth

Current cone_loss target:
  - target_depth = ρ / 30  →  high ρ → high depth (opposite dehydron-rim story)

Usage:
  make audit-dehydron-topology CHECKPOINT=checkpoints/v6/tokyo_eyes_v6.pt

  python -m experiments.training.v6.dehydron_cone_alignment_audit \\
    --checkpoint checkpoints/v6/tokyo_eyes_v6.pt \\
    --corpus manifests/v6_corpus_disc_target.json \\
    --pdb-dir /tmp/dtie_pdb_cache \\
    --pdb-local \\
    --plot-dir checkpoints/v6/diagnostics/dehydron_topology \\
    --gate-exit
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.training.v6.corpus import load_training_proteins
from experiments.training.v6.train_loop import attach_v6_features
from science.training.corpus_governance import STAGE_A_MAX_RESIDUES
from science.training.dehydron_cone_gate import (
    GATE_NAME,
    MIN_R_CONE_DEPTH_TAU,
    dehydron_cone_gate_verdict,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parents[3]
RHO_SCALE = 30.0


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or b.size < 3:
        return float("nan")
    ac = a - a.mean()
    bc = b - b.mean()
    denom = np.linalg.norm(ac) * np.linalg.norm(bc)
    if denom < 1e-12:
        return float("nan")
    return float(np.dot(ac, bc) / denom)


def _load_model(checkpoint: Path, device: str) -> torch.nn.Module:
    from science.dtie.v6.gnn.model import GOSPConeMapperV6, infer_v6_model_kwargs, load_v6_state_dict

    raw = torch.load(checkpoint, map_location=device, weights_only=False)
    state = raw.get("model_state_dict", raw)
    kwargs = infer_v6_model_kwargs(state, raw.get("architecture"), raw.get("training_config"))
    bias = state.get("gate.expert_bias")
    if bias is not None:
        kwargs["num_experts"] = int(bias.shape[0])
    model = GOSPConeMapperV6(**kwargs)
    load_v6_state_dict(model, state)
    model.to(device)
    model.eval()
    return model


def _eval_structure(
    model: torch.nn.Module,
    prot: dict[str, Any],
    device: str,
    *,
    include_plot: bool = False,
) -> dict[str, Any]:
    pdb_id = str(prot["pdb_id"]).upper()
    data = attach_v6_features(prot["data"].to(device))
    with torch.no_grad():
        out = model(data)

    rho = prot["target_rho"].squeeze(-1).cpu().numpy().astype(np.float64)
    tau = prot["target_dehydron"].squeeze(-1).cpu().numpy().astype(np.float64)
    sasa_raw = prot["target_sasa"].squeeze(-1).cpu().numpy().astype(np.float64)
    # Training graph may store SASA as Å² or normalized; z-score per structure for correlation.
    sasa_z = (sasa_raw - sasa_raw.mean()) / (sasa_raw.std() + 1e-8)

    cone_depth = out["cone_depth"].squeeze(-1).cpu().numpy().astype(np.float64)
    radial = out.get("radial_features")
    if radial is None:
        radial = out.get("radial_depth")
    radial_depth = radial.squeeze(-1).cpu().numpy().astype(np.float64)

    disc_r = out["hyp_projections_2d"].norm(dim=-1).cpu().numpy().astype(np.float64)
    disc_xy = out["hyp_projections_2d"].detach().cpu().numpy().astype(np.float64)
    target_depth = np.clip(rho / RHO_SCALE, 0.0, 1.0)

    n_tau = int((tau > 0.5).sum())
    n_res = int(rho.size)

    correlations = {
        "r_cone_depth_tau": _pearson(cone_depth, tau),
        "r_cone_depth_rho": _pearson(cone_depth, rho),
        "r_cone_depth_sasa_z": _pearson(cone_depth, sasa_z),
        "r_cone_depth_sasa": _pearson(cone_depth, sasa_raw),
        "r_cone_depth_target_depth": _pearson(cone_depth, target_depth),
        "r_radial_depth_rho": _pearson(radial_depth, rho),
        "r_radial_depth_tau": _pearson(radial_depth, tau),
        "r_disc_r_tau": _pearson(disc_r, tau),
        "r_disc_r_rho": _pearson(disc_r, rho),
        "r_disc_r_sasa_z": _pearson(disc_r, sasa_z),
        "r_target_depth_rho": _pearson(target_depth, rho),  # tautology check ≈ 1
    }

    row: dict[str, Any] = {
        "pdb_id": pdb_id,
        "fold_id": str(prot.get("fold_id", "")),
        "n_residues": n_res,
        "tau_fraction": float(tau.mean()),
        "n_dehydron_residues": n_tau,
        "mean_rho": float(rho.mean()),
        "std_rho": float(rho.std()),
        "mean_cone_depth": float(cone_depth.mean()),
        "std_cone_depth": float(cone_depth.std()),
        "mean_disc_r": float(disc_r.mean()),
        "correlations": correlations,
    }
    if include_plot:
        row["plot"] = {
            "disc_xy": disc_xy,
            "tau": tau,
            "rho": rho,
            "cone_depth": cone_depth,
            "disc_r": disc_r,
            "sasa_z": sasa_z,
        }
    return row


def _gate_health_from_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    def mean_r(key: str) -> float:
        vals = [r["correlations"][key] for r in rows if np.isfinite(r["correlations"][key])]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "probe_r_depth_tau": mean_r("r_cone_depth_tau"),
        "probe_r_depth_rho": mean_r("r_cone_depth_rho"),
    }


def _render_topology_panels(
    per_structure: list[dict[str, Any]],
    plot_dir: Path,
    checkpoint_label: str,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for row in per_structure:
        plot = row.get("plot")
        if not plot:
            continue
        pdb_id = str(row["pdb_id"]).lower()
        disc_xy = plot["disc_xy"]
        tau = plot["tau"]
        rho = plot["rho"]
        cd = plot["cone_depth"]
        corr = row["correlations"]

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        fig.suptitle(
            f"{pdb_id.upper()} — {checkpoint_label}\n"
            f"r(depth,τ)={corr['r_cone_depth_tau']:.3f}  "
            f"r(depth,ρ)={corr['r_cone_depth_rho']:.3f}  "
            f"r(disc_r,τ)={corr['r_disc_r_tau']:.3f}",
            fontsize=11,
        )

        ax = axes[0, 0]
        sc = ax.scatter(disc_xy[:, 0], disc_xy[:, 1], c=tau, cmap="coolwarm", s=14, vmin=0, vmax=1)
        ax.set_title("disc_2d colored by τ (dehydron)")
        ax.set_aspect("equal")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

        ax = axes[0, 1]
        sc = ax.scatter(disc_xy[:, 0], disc_xy[:, 1], c=rho, cmap="viridis", s=14)
        ax.set_title("disc_2d colored by ρ (wrap density)")
        ax.set_aspect("equal")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

        ax = axes[1, 0]
        ax.scatter(tau, cd, c="#eca53a", s=12, alpha=0.75)
        ax.set_xlabel("τ (dehydron flag)")
        ax.set_ylabel("cone_depth")
        ax.set_title("depth vs dehydron")

        ax = axes[1, 1]
        ax.scatter(rho, cd, c="#4a9eff", s=12, alpha=0.75)
        ax.set_xlabel("ρ (wrap density)")
        ax.set_ylabel("cone_depth")
        ax.set_title("depth vs wrap (ρ/30 loss pulls here)")

        fig.tight_layout()
        out_path = plot_dir / f"{pdb_id}_topology.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(out_path)
        logger.info("Wrote panel %s", out_path)

    if per_structure:
        health = _gate_health_from_rows(per_structure)
        fig, ax = plt.subplots(figsize=(7, 4))
        keys = ["probe_r_depth_tau", "probe_r_depth_rho"]
        labels = ["r(depth,τ)", "r(depth,ρ)"]
        vals = [health[k] for k in keys]
        colors = ["#2ecc71" if k == "probe_r_depth_tau" and v >= MIN_R_CONE_DEPTH_TAU else "#e74c3c" if k == "probe_r_depth_tau" else "#3498db" for k, v in zip(keys, vals)]
        ax.bar(labels, vals, color=colors)
        ax.axhline(MIN_R_CONE_DEPTH_TAU, color="#2ecc71", linestyle="--", linewidth=1, label=f"P_DEHYDRON τ floor ({MIN_R_CONE_DEPTH_TAU})")
        ax.axhline(-MIN_R_CONE_DEPTH_TAU, color="#95a5a6", linestyle=":", linewidth=0.8)
        ax.set_ylabel("Pearson r (corpus mean)")
        ax.set_title(f"{GATE_NAME} probes — {checkpoint_label}")
        ax.legend(loc="best", fontsize=8)
        summary_path = plot_dir / f"{checkpoint_label}_gate_summary.png"
        fig.tight_layout()
        fig.savefig(summary_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(summary_path)
        logger.info("Wrote summary %s", summary_path)

    return written


def _print_gate_report(
    verdict: Any,
    health: dict[str, float],
    *,
    n_structures: int,
) -> None:
    print("=" * 78)
    print(f"GATE {GATE_NAME}  (n_structures={n_structures})")
    print(f"  PASS: {'YES' if verdict.passed else 'NO'}")
    print(f"  r(cone_depth, τ) = {health['probe_r_depth_tau']:.4f}  (need ≥ {MIN_R_CONE_DEPTH_TAU})")
    if "probe_r_depth_rho" in health:
        print(f"  r(cone_depth, ρ) = {health['probe_r_depth_rho']:.4f}")
    print(f"  reason: {verdict.reason}")
    print("=" * 78)


def _verdict(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean_r(key: str) -> float:
        vals = [r["correlations"][key] for r in rows if np.isfinite(r["correlations"][key])]
        return float(np.mean(vals)) if vals else float("nan")

    agg = {
        "mean_r_cone_depth_tau": mean_r("r_cone_depth_tau"),
        "mean_r_cone_depth_rho": mean_r("r_cone_depth_rho"),
        "mean_r_cone_depth_sasa_z": mean_r("r_cone_depth_sasa_z"),
        "mean_r_radial_depth_rho": mean_r("r_radial_depth_rho"),
        "mean_r_disc_r_tau": mean_r("r_disc_r_tau"),
        "mean_r_disc_r_sasa_z": mean_r("r_disc_r_sasa_z"),
    }

    theory_tau = agg["mean_r_cone_depth_tau"] > 0.15
    theory_rho = agg["mean_r_cone_depth_rho"] < -0.15
    cone_loss_rho = agg["mean_r_radial_depth_rho"] > 0.3

    if theory_tau and theory_rho:
        label = "dehydron_rim_aligned"
    elif cone_loss_rho and not theory_tau:
        label = "cone_loss_rho_aligned_not_dehydron_theory"
    else:
        label = "weak_or_mixed_alignment"

    return {
        "aggregate_correlations": agg,
        "theory_dehydron_rim": {
            "expects_r_cone_depth_tau_positive": True,
            "expects_r_cone_depth_rho_negative": True,
            "tau_positive": theory_tau,
            "rho_negative": theory_rho,
        },
        "cone_loss_contract": {
            "expects_r_radial_depth_rho_positive": True,
            "rho_positive": cone_loss_rho,
        },
        "verdict": label,
        "interpretation": (
            "dehydron_rim_aligned: dehydrons (τ) correlate with higher cone depth as theory predicts. "
            "cone_loss_rho_aligned_not_dehydron_theory: model follows ρ/30 supervision (high wrap → deep) "
            "but not τ→rim. weak_or_mixed_alignment: partial or inconsistent τ/ρ alignment."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Dehydron–cone alignment audit")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=_REPO / "manifests" / "v6_corpus_disc_target.json",
    )
    parser.add_argument("--pdb-dir", type=Path, default=_REPO / "pdb_cache")
    parser.add_argument(
        "--pdb-local",
        action="store_true",
        help="Build graphs from PDB (TRAINING_LOAD_FROM_PDB=1); no DB pool required",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Write per-structure matplotlib topology panels (+ gate summary bar chart)",
    )
    parser.add_argument(
        "--gate-exit",
        action="store_true",
        help=f"Exit 1 when {GATE_NAME} fails",
    )
    args = parser.parse_args()

    if args.pdb_local:
        os.environ["TRAINING_LOAD_FROM_PDB"] = "1"

    proteins, failed = load_training_proteins(
        args.pdb_dir,
        args.corpus,
        max_proteins=12,
        max_residues=STAGE_A_MAX_RESIDUES,
        use_cache=True,
    )
    if failed or len(proteins) < 1:
        raise SystemExit(f"Need >=1 proteins, got {len(proteins)} ({failed} failed)")

    model = _load_model(args.checkpoint, args.device)
    include_plot = args.plot_dir is not None
    per_structure = [
        _eval_structure(model, p, args.device, include_plot=include_plot) for p in proteins
    ]
    del model
    if args.device != "cpu" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Strip plot tensors before JSON serialization.
    per_structure_json = []
    for row in per_structure:
        slim = {k: v for k, v in row.items() if k != "plot"}
        per_structure_json.append(slim)

    summary = _verdict(per_structure_json)
    health = _gate_health_from_rows(per_structure_json)
    gate = dehydron_cone_gate_verdict(health)

    checkpoint_label = args.checkpoint.stem
    plot_paths: list[str] = []
    if args.plot_dir is not None:
        written = _render_topology_panels(per_structure, args.plot_dir, checkpoint_label)
        plot_paths = [str(p) for p in written]

    report: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "corpus": str(args.corpus),
        "n_structures": len(per_structure_json),
        "feature_source": "MASTER features (rho, tau, sasa) — no GNN leak/motif tables",
        "theory": {
            "dehydron_rim": "τ=1 (ρ<TAU) → high cone_depth / disc_r",
            "stable_core": "high ρ wrap → low depth (center)",
        },
        "cone_loss_target": f"target_depth = clip(ρ / {RHO_SCALE}, 0, 1)",
        "per_structure": per_structure_json,
        "summary": summary,
        "gate": {
            "name": GATE_NAME,
            "passed": gate.passed,
            "min_r_cone_depth_tau": MIN_R_CONE_DEPTH_TAU,
            "health": health,
            "reason": gate.reason,
        },
        "plot_paths": plot_paths,
    }

    out = args.output or Path(f"dehydron_cone_alignment_{checkpoint_label}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    logger.info("Wrote %s verdict=%s gate=%s", out, summary["verdict"], gate.passed)

    print(f"\nAlignment verdict: {summary['verdict']}")
    agg = summary["aggregate_correlations"]
    print(
        f"  mean r(depth,τ)={agg['mean_r_cone_depth_tau']:.3f}  "
        f"r(depth,ρ)={agg['mean_r_cone_depth_rho']:.3f}  "
        f"r(disc_r,τ)={agg['mean_r_disc_r_tau']:.3f}"
    )
    _print_gate_report(gate, health, n_structures=len(per_structure_json))

    if args.gate_exit and not gate.passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
