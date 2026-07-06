"""Dehydron–cone alignment audit — does learned depth follow dehydron theory or SASA shell?

Compares per-residue MASTER features (ρ, τ, SASA from DB) against model outputs
(cone_depth, radial_depth, disc_r) without using GNN-derived leak/motif tables.

Theory (Tokyo Eye premise):
  - Dehydrons (τ=1, low ρ) → funnel rim → high hyperbolic depth / disc radius
  - Well-wrapped stable core → center → low depth

Current cone_loss target:
  - target_depth = ρ / 30  →  high ρ → high depth (opposite dehydron-rim story)

Usage:
  docker compose run --rm --no-deps science python -m experiments.training.v6.dehydron_cone_alignment_audit \\
    --checkpoint /app/checkpoints/v6/runs/stage_a_small_master_cold_n6_v2/v6_phase2_12prot.pt \\
    --corpus /app/manifests/v6_corpus_stage_a_small_v1.json \\
    --pdb-dir /tmp/dtie_pdb_cache \\
    --output /tmp/dehydron_cone_alignment_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.training.v6.corpus import load_training_proteins
from experiments.training.v6.train_loop import attach_v6_features
from science.training.corpus_governance import STAGE_A_MAX_RESIDUES

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
    target_depth = np.clip(rho / RHO_SCALE, 0.0, 1.0)

    n_tau = int((tau > 0.5).sum())
    n_res = int(rho.size)

    correlations = {
        "r_cone_depth_tau": _pearson(cone_depth, tau),
        "r_cone_depth_rho": _pearson(cone_depth, rho),
        "r_cone_depth_sasa_z": _pearson(cone_depth, sasa_z),
        "r_cone_depth_target_depth": _pearson(cone_depth, target_depth),
        "r_radial_depth_rho": _pearson(radial_depth, rho),
        "r_radial_depth_tau": _pearson(radial_depth, tau),
        "r_disc_r_tau": _pearson(disc_r, tau),
        "r_disc_r_rho": _pearson(disc_r, rho),
        "r_disc_r_sasa_z": _pearson(disc_r, sasa_z),
        "r_target_depth_rho": _pearson(target_depth, rho),  # tautology check ≈ 1
    }

    return {
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
    sasa_wins = abs(agg["mean_r_cone_depth_sasa_z"]) > abs(agg["mean_r_cone_depth_tau"])

    if theory_tau and theory_rho:
        label = "dehydron_rim_aligned"
    elif cone_loss_rho and not theory_tau:
        label = "cone_loss_rho_aligned_not_dehydron_theory"
    elif sasa_wins:
        label = "sasa_shell_dominated"
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
        "sasa_dominance_over_tau": sasa_wins,
        "verdict": label,
        "interpretation": (
            "dehydron_rim_aligned: dehydrons (τ) correlate with higher cone depth as theory predicts. "
            "cone_loss_rho_aligned_not_dehydron_theory: model follows ρ/30 supervision (high wrap → deep) "
            "but not τ→rim. sasa_shell_dominated: depth tracks solvent exposure more than dehydron identity."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Dehydron–cone alignment audit")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=_REPO / "manifests" / "v6_corpus_stage_a_small_v1.json")
    parser.add_argument("--pdb-dir", type=Path, default=_REPO / "pdb_cache")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    proteins, failed = load_training_proteins(
        args.pdb_dir,
        args.corpus,
        max_proteins=12,
        max_residues=STAGE_A_MAX_RESIDUES,
        use_cache=True,
    )
    if failed or len(proteins) < 2:
        raise SystemExit(f"Need >=2 proteins, got {len(proteins)} ({failed} failed)")

    model = _load_model(args.checkpoint, args.device)
    per_structure = [_eval_structure(model, p, args.device) for p in proteins]
    del model
    if args.device != "cpu" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    report: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "corpus": str(args.corpus),
        "n_structures": len(per_structure),
        "feature_source": "MASTER DB (rho, tau, sasa) — no GNN leak/motif tables",
        "theory": {
            "dehydron_rim": "τ=1 (ρ<TAU) → high cone_depth / disc_r",
            "stable_core": "high ρ wrap → low depth (center)",
        },
        "cone_loss_target": f"target_depth = clip(ρ / {RHO_SCALE}, 0, 1)",
        "per_structure": per_structure,
        "summary": _verdict(per_structure),
    }

    out = args.output or Path("dehydron_cone_alignment_report.json")
    out.write_text(json.dumps(report, indent=2))
    logger.info("Wrote %s verdict=%s", out, report["summary"]["verdict"])


if __name__ == "__main__":
    main()
