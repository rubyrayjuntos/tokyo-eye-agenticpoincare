"""
assess_checkpoint.py — V6 checkpoint assessment harness (MoE + geometry + shell probes).

Usage:
    python -m experiments.training.v6.assess_checkpoint \\
        --checkpoint checkpoints/v6/runs/default/v6_best.pt \\
        --corpus manifests/v6_corpus_120.json \\
        --max-proteins 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from experiments.training.v6.corpus import load_training_proteins
from experiments.training.v6.train_loop import attach_v6_features, measure_geometry_health
from science.training.checkpoint_score import (
    PROJ_FRAC_MAX,
    ROUTING_ENTROPY_PROMOTE_MAX,
    disc_save_ineligibility_reasons,
    shell_probe_ineligibility_reasons,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("assess_v6")

MAX_ROUTING_ENTROPY = 1.386  # ln(4)


def _flat_uncertainty(uncertainty: dict[str, torch.Tensor] | torch.Tensor, key: str = "epistemic") -> np.ndarray:
    """Extract a flat numpy uncertainty vector from model output."""
    if isinstance(uncertainty, dict):
        tensor = uncertainty[key]
    else:
        tensor = uncertainty
    return tensor.detach().cpu().numpy().reshape(-1)


def _inference_mode(model: nn.Module) -> None:
    model.train(False)


def load_v6_model(
    checkpoint_path: Path,
    device: str,
    *,
    legacy_disc_projection: bool | None = None,
) -> nn.Module:
    from science.dtie.v6.gnn.model import (
        GOSPConeMapperV6,
        infer_legacy_disc_projection_from_checkpoint,
        infer_v6_model_kwargs,
        load_v6_state_dict,
    )

    raw = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = raw.get("model_state_dict", raw)
    kwargs = infer_v6_model_kwargs(
        state,
        raw.get("architecture"),
        raw.get("training_config"),
    )
    kwargs["legacy_disc_projection"] = infer_legacy_disc_projection_from_checkpoint(
        training_config=raw.get("training_config") if isinstance(raw, dict) else None,
        metrics=raw.get("metrics") if isinstance(raw, dict) else None,
        override=legacy_disc_projection,
    )
    model = GOSPConeMapperV6(**kwargs)
    load_v6_state_dict(model, state)
    model.to(device)
    _inference_mode(model)
    return model


def collect_protein_bundle(
    model: nn.Module,
    prot: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    """Single-protein inference bundle for per-structure diagnostics."""
    with torch.no_grad():
        data = attach_v6_features(prot["data"].to(device))
        out = model(data)
        sasa = data.x[:, 3].cpu().numpy()
        depth = out["cone_depth"].squeeze().cpu().numpy()
        epistemic = _flat_uncertainty(out["uncertainty"], "epistemic")
        aleatoric = _flat_uncertainty(out["uncertainty"], "aleatoric")
        hyp2d = out["hyp_projections_2d"].cpu().numpy()
    return {
        "pdb_id": prot.get("pdb_id", "?"),
        "n_residues": int(len(sasa)),
        "sasa": sasa,
        "cone_depth": depth,
        "epistemic": epistemic,
        "aleatoric": aleatoric,
        "hyp_projections_2d": hyp2d,
    }


def collect_inference_bundle(
    model: nn.Module,
    proteins: list[dict[str, Any]],
    device: str,
) -> dict[str, Any]:
    """Aggregate tensors for probe functions."""
    all_sasa, all_depth, all_epi, all_ale = [], [], [], []
    all_hyp2d = []
    expert_weights_all: list[np.ndarray] = []
    routing_entropies: list[float] = []
    expert_loads: list[np.ndarray] = []

    with torch.no_grad():
        for prot in proteins:
            data = attach_v6_features(prot["data"].to(device))
            out = model(data)
            sasa = data.x[:, 3].cpu().numpy()
            depth = out["cone_depth"].squeeze().cpu().numpy()
            epistemic = _flat_uncertainty(out["uncertainty"], "epistemic")
            aleatoric = _flat_uncertainty(out["uncertainty"], "aleatoric")
            hyp2d = out["hyp_projections_2d"].cpu().numpy()
            ew = out["expert_weights"].cpu().numpy()

            all_sasa.append(sasa)
            all_depth.append(depth)
            all_epi.append(epistemic)
            all_ale.append(aleatoric)
            all_hyp2d.append(hyp2d)
            expert_weights_all.append(ew)
            routing_entropies.append(float(out["routing_entropy"].item()))
            expert_loads.append(out["expert_load"].cpu().numpy())

    return {
        "sasa": np.concatenate(all_sasa),
        "cone_depth": np.concatenate(all_depth),
        "epistemic": np.concatenate(all_epi),
        "aleatoric": np.concatenate(all_ale),
        "uncertainty": np.concatenate(all_epi),
        "hyp_projections_2d": np.concatenate(all_hyp2d),
        "expert_weights": np.vstack(expert_weights_all),
        "routing_entropy_mean": float(np.mean(routing_entropies)),
        "expert_load_mean": np.mean(np.stack(expert_loads), axis=0),
    }


def moe_metrics(bundle: dict[str, Any]) -> dict[str, Any]:
    expert_load = bundle["expert_load_mean"]
    starvation = int(np.sum(expert_load < 0.05))
    return {
        "routing_entropy_mean": bundle["routing_entropy_mean"],
        "routing_entropy_max": MAX_ROUTING_ENTROPY,
        "expert_load": expert_load.tolist(),
        "expert_starvation_count": starvation,
        "expert_load_std": float(np.std(expert_load)),
    }


def expert_profiles(
    model: nn.Module,
    proteins: list[dict[str, Any]],
    device: str,
) -> list[dict[str, Any]]:
    """Per-expert mean degree, rho, clustering, SS distribution."""
    num_experts = len(model.experts)
    profiles: list[dict[str, Any]] = [
        {"expert": i, "degree": [], "rho": [], "clustering": [], "ss_helix": 0, "ss_sheet": 0, "ss_coil": 0}
        for i in range(num_experts)
    ]

    with torch.no_grad():
        for prot in proteins:
            data = attach_v6_features(prot["data"].to(device))
            out = model(data)
            weights = out["expert_weights"].cpu().numpy()
            assigned = weights.argmax(axis=1)
            deg = data.degree.cpu().numpy()
            rho = data.rho.cpu().numpy()
            clust = data.clustering.cpu().numpy()
            ss = data.ss_onehot.cpu().numpy()

            for i in range(num_experts):
                mask = assigned == i
                if not mask.any():
                    continue
                profiles[i]["degree"].extend(deg[mask].tolist())
                profiles[i]["rho"].extend(rho[mask].tolist())
                profiles[i]["clustering"].extend(clust[mask].tolist())
                profiles[i]["ss_helix"] += int(ss[mask, 0].sum())
                profiles[i]["ss_sheet"] += int(ss[mask, 1].sum())
                profiles[i]["ss_coil"] += int(ss[mask, 2].sum())

    for p in profiles:
        p["mean_degree"] = float(np.mean(p["degree"])) if p["degree"] else 0.0
        p["mean_rho"] = float(np.mean(p["rho"])) if p["rho"] else 0.0
        p["mean_clustering"] = float(np.mean(p["clustering"])) if p["clustering"] else 0.0
        del p["degree"]
        del p["rho"]
        del p["clustering"]
    return profiles


def run_shell_probes(bundle: dict[str, Any]) -> dict[str, Any]:
    """Reuse shell_signal_diagnostics probe logic on aggregated bundle."""
    from experiments.diagnostics.shell_signal_diagnostics import (
        probe1_uncertainty_sasa_correlation,
        probe3_surface_hotspot_alignment,
        probe4_radial_gradient,
    )

    out = {
        "sasa": bundle["sasa"],
        "cone_depth": bundle["cone_depth"],
        "epistemic": bundle["epistemic"],
        "aleatoric": bundle["aleatoric"],
        "uncertainty": bundle["epistemic"],
        "hyp_proj_2d": bundle["hyp_projections_2d"],
        "residue_ids": [str(i) for i in range(len(bundle["sasa"]))],
        "structure_id": "corpus_assess",
    }
    return {
        "probe1_uncertainty_sasa": probe1_uncertainty_sasa_correlation(out),
        "probe3_surface_hotspot": probe3_surface_hotspot_alignment(out),
        "probe4_radial_gradient": probe4_radial_gradient(out),
    }


def promotion_gate(
    report: dict[str, Any],
    *,
    disc_radial_source: str = "mobius",
) -> dict[str, Any]:
    """Determine pass/fail for candidate promotion (MoE + geometry + shell probes)."""
    from science.dtie.v6.gnn.model import uses_disc_radial_override

    moe = report.get("moe", {})
    geom = report.get("geometry", {})
    failures: list[str] = []
    radial_override = uses_disc_radial_override(disc_radial_source)

    if not radial_override and moe.get("routing_entropy_mean", 0) > ROUTING_ENTROPY_PROMOTE_MAX:
        failures.append(
            f"routing_entropy {moe['routing_entropy_mean']:.3f} > {ROUTING_ENTROPY_PROMOTE_MAX}"
        )
    if moe.get("expert_starvation_count", 0) > 0:
        failures.append(f"expert starvation: {moe['expert_starvation_count']} experts < 5%")
    if geom.get("proj_frac_mean", 1.0) > PROJ_FRAC_MAX:
        failures.append(f"boundary saturation proj_frac={geom['proj_frac_mean']:.3f}")

    failures.extend(shell_probe_ineligibility_reasons(geom))
    failures.extend(
        disc_save_ineligibility_reasons(
            geom,
            disc_radial_source=disc_radial_source,
        )
    )

    return {"passed": len(failures) == 0, "failures": failures}


def assess_checkpoint(
    checkpoint_path: Path,
    pdb_dir: Path,
    corpus_path: Path,
    *,
    device: str = "cpu",
    max_proteins: int | None = 5,
    max_residues: int = 800,
    legacy_disc_projection: bool | None = None,
) -> dict[str, Any]:
    proteins, failed = load_training_proteins(
        pdb_dir,
        corpus_path,
        max_proteins=max_proteins,
        max_residues=max_residues,
    )
    if not proteins:
        raise RuntimeError(f"No proteins loaded for assessment ({failed} failed)")

    model = load_v6_model(
        checkpoint_path,
        device,
        legacy_disc_projection=legacy_disc_projection,
    )
    geometry = measure_geometry_health(model, proteins, device)
    bundle = collect_inference_bundle(model, proteins, device)
    moe = moe_metrics(bundle)
    profiles = expert_profiles(model, proteins, device)
    probes = run_shell_probes(bundle)

    report: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "protein_count": len(proteins),
        "geometry": geometry,
        "moe": moe,
        "expert_profiles": profiles,
        "shell_probes": probes,
    }
    disc_radial_source = getattr(model, "disc_radial_source", "mobius")
    report["disc_radial_source"] = disc_radial_source
    report["promotion_gate"] = promotion_gate(
        report,
        disc_radial_source=disc_radial_source,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Assess v6 GNN checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=Path("manifests/v6_corpus_120.json"))
    parser.add_argument("--pdb-dir", type=Path, default=Path("/tmp/dtie_pdb_cache"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-proteins", type=int, default=5)
    parser.add_argument("--max-residues", type=int, default=600)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--mlflow-run-id", default=None)
    parser.add_argument(
        "--no-promotion-exit",
        action="store_true",
        help="Write report but do not exit 1 when promotion gate fails (benchmark/dev)",
    )
    parser.add_argument(
        "--legacy-disc-projection",
        action="store_true",
        help="Force legacy post-routing disc projection at inference",
    )
    parser.add_argument(
        "--no-legacy-disc-projection",
        action="store_true",
        help="Force new pre-routing disc projection at inference",
    )
    args = parser.parse_args()

    legacy_override: bool | None = None
    if args.legacy_disc_projection:
        legacy_override = True
    if args.no_legacy_disc_projection:
        legacy_override = False

    report = assess_checkpoint(
        args.checkpoint,
        args.pdb_dir,
        args.corpus,
        device=args.device,
        max_proteins=args.max_proteins,
        max_residues=args.max_residues,
        legacy_disc_projection=legacy_override,
    )

    out_path = args.output or args.checkpoint.parent / "assess_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    logger.info("Wrote %s", out_path)
    logger.info("Promotion gate: %s", report["promotion_gate"])

    if args.mlflow_run_id:
        try:
            import mlflow

            mlflow.set_tracking_uri("file:./mlruns")
            with mlflow.start_run(run_id=args.mlflow_run_id):
                mlflow.log_artifact(str(out_path), artifact_path="assess")
        except ImportError:
            logger.warning("mlflow not installed; skipping artifact upload")

    if not report["promotion_gate"]["passed"] and not args.no_promotion_exit:
        sys.exit(1)


if __name__ == "__main__":
    main()
