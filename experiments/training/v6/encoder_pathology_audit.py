"""Encoder pathology audit — fold separation across representation stages and checkpoints."""

from __future__ import annotations

import argparse
import json
import logging
from itertools import combinations
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


def _pairwise_cosine(vectors: np.ndarray) -> dict[str, float]:
    """Mean/std/min/max of off-diagonal cosine similarities."""
    if len(vectors) < 2:
        return {"pairwise_mean": 1.0, "pairwise_std": 0.0, "pairwise_min": 1.0, "pairwise_max": 1.0}
    norms = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
    sim = norms @ norms.T
    n = sim.shape[0]
    mask = ~np.eye(n, dtype=bool)
    vals = sim[mask]
    return {
        "pairwise_mean": float(vals.mean()),
        "pairwise_std": float(vals.std()),
        "pairwise_min": float(vals.min()),
        "pairwise_max": float(vals.max()),
    }


def _pc1_variance_frac(vectors: np.ndarray) -> float:
    if len(vectors) < 2:
        return 1.0
    x = vectors - vectors.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(x, full_matrices=False)
    var = s**2
    total = var.sum()
    return float(var[0] / total) if total > 1e-12 else 1.0


def _cath_separation(
    vectors: np.ndarray,
    fold_ids: list[str],
    *,
    level: int = 2,
) -> dict[str, Any]:
    """Within vs across CATH class separation (positive delta = better across separation)."""
    prefixes = [f.split(".")[:level] for f in fold_ids]
    prefix_str = [".".join(p) for p in prefixes]
    within: list[float] = []
    across: list[float] = []
    norms = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
    sim = norms @ norms.T
    for i, j in combinations(range(len(vectors)), 2):
        if prefix_str[i] == prefix_str[j]:
            within.append(float(sim[i, j]))
        else:
            across.append(float(sim[i, j]))
    out: dict[str, Any] = {
        "n_within_pairs": len(within),
        "n_across_pairs": len(across),
    }
    if within:
        out["within_mean_cos"] = float(np.mean(within))
    if across:
        out["across_mean_cos"] = float(np.mean(across))
    if within and across:
        out["separation_delta"] = out["within_mean_cos"] - out["across_mean_cos"]
    return out


def _structure_stats(prot: dict[str, Any]) -> dict[str, float]:
    """Per-structure scalar summaries from raw MASTER features."""
    x = prot["data"].x.detach().cpu().numpy()
    rho, tau, ss, sasa = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    return {
        "mean_rho": float(rho.mean()),
        "mean_tau": float(tau.mean()),
        "mean_ss": float(ss.mean()),
        "mean_sasa": float(sasa.mean()),
        "std_rho": float(rho.std()),
        "std_sasa": float(sasa.std()),
        "n_residues": int(prot.get("n_residues", len(rho))),
    }


def _extract_representations(
    model: torch.nn.Module,
    proteins: list[dict],
    device: str,
) -> dict[str, dict[str, np.ndarray]]:
    """Structure-level vectors at each stage of the encoder pipeline."""
    backbone_hook: list[torch.Tensor] = []

    def _capture(_module, _inp, out):
        backbone_hook.append(out.detach())

    handle = model.convs[-1].register_forward_hook(_capture)
    stages: dict[str, dict[str, np.ndarray]] = {
        "raw_input": {},
        "backbone": {},
        "radial_depth_mean": {},
        "angular_direction_mean": {},
        "disc_2d_mean": {},
        "gate_scores_mean": {},
        "depth_mean": {},
    }
    try:
        with torch.no_grad():
            for prot in proteins:
                pdb = str(prot["pdb_id"]).upper()
                data = attach_v6_features(prot["data"].to(device))
                raw = data.x.detach().cpu().numpy()
                stages["raw_input"][pdb] = raw.mean(axis=0)

                backbone_hook.clear()
                out = model(data)
                bb = backbone_hook[-1].mean(dim=0).cpu().numpy()
                stages["backbone"][pdb] = bb

                depth = out["cone_depth"].squeeze(-1).cpu().numpy()
                stages["radial_depth_mean"][pdb] = np.array([float(depth.mean()), float(depth.std())])
                stages["depth_mean"][pdb] = np.array([float(depth.mean())])

                ang = out.get("angular_features")
                if ang is not None:
                    stages["angular_direction_mean"][pdb] = ang.mean(dim=0).cpu().numpy()
                else:
                    stages["angular_direction_mean"][pdb] = bb  # fallback

                disc = out["hyp_projections_2d"].cpu().numpy()
                stages["disc_2d_mean"][pdb] = disc.mean(axis=0)

                scores = out["expert_weights"].mean(dim=0).cpu().numpy()
                stages["gate_scores_mean"][pdb] = scores
    finally:
        handle.remove()
    return stages


def _summarize_stage(
    stage_vecs: dict[str, np.ndarray],
    fold_ids: dict[str, str],
) -> dict[str, Any]:
    pdbs = sorted(stage_vecs.keys())
    vectors = np.stack([stage_vecs[p] for p in pdbs])
    folds = [fold_ids[p] for p in pdbs]
    summary = {
        "n_structures": len(pdbs),
        "pairwise_cosine": _pairwise_cosine(vectors),
        "pc1_variance_frac": _pc1_variance_frac(vectors),
        "cath_architecture_2": _cath_separation(vectors, folds, level=2),
        "cath_topology_3": _cath_separation(vectors, folds, level=3),
    }
    # Depth-norm correlation for 1D depth summaries
    if vectors.shape[1] == 1:
        norms = vectors[:, 0]
        summary["depth_spread"] = {
            "min": float(norms.min()),
            "max": float(norms.max()),
            "std": float(norms.std()),
        }
    return summary


def _loss_contribution_audit(metrics_path: Path) -> dict[str, Any]:
    """Estimate weighted loss contributions at phase boundaries."""
    metrics = json.loads(metrics_path.read_text())
    coeffs_p1 = {
        "cone_consistency": 0.30,
        "shell_correlation": 0.35,
        "neighborhood_consistency": 0.10,
        "angular_diversity": 0.0,
        "domain_separation_2d": 0.0,
        "domain_separation_3d": 0.0,
        "cone_depth_anticollapse": 0.75,
    }
    coeffs_p2 = {
        "cone_consistency": 0.25,
        "shell_correlation": 0.25,
        "neighborhood_consistency": 0.20,
        "angular_diversity": 0.20,
        "domain_separation_2d": 0.15,
        "domain_separation_3d": 0.15,
        "cone_depth_anticollapse": 0.50,
    }

    def _phase_tail(phase: int) -> dict[str, float]:
        rows = [m for m in metrics if m.get("phase") == phase]
        if not rows:
            return {}
        L = rows[-1]["losses"]
        coeffs = coeffs_p1 if phase == 1 else coeffs_p2
        weighted = {k: coeffs.get(k, 0.0) * float(L.get(k, 0.0)) for k in coeffs}
        weighted["evidential_raw"] = float(L.get("evidential", 0.0))
        weighted["grad_radial"] = float(L.get("grad_radial", 0.0))
        weighted["grad_angular"] = float(L.get("grad_angular", 0.0))
        weighted["grad_backbone"] = float(L.get("grad_backbone", 0.0))
        weighted["domain_sep_2d_active"] = float(L.get("domain_separation_2d", 0.0))
        return weighted

    return {
        "p1_final_weighted": _phase_tail(1),
        "p2_final_weighted": _phase_tail(2),
        "domain_labels_note": (
            "domain_separation only fires for structures with >=2 KRAS switch domains "
            "(4OBE only in 12-prot corpus); not CATH fold discrimination"
        ),
    }


def _load_model_for_checkpoint(checkpoint: Path, device: str) -> torch.nn.Module:
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


def audit_checkpoint(
    checkpoint: Path,
    proteins: list[dict],
    device: str,
    label: str,
) -> dict[str, Any]:
    model = _load_model_for_checkpoint(checkpoint, device)
    fold_ids = {str(p["pdb_id"]).upper(): str(p.get("fold_id", "")) for p in proteins}
    domain_label_count = sum(1 for p in proteins if p.get("domain_labels") is not None)

    stages = _extract_representations(model, proteins, device)
    stage_summaries = {
        name: _summarize_stage(vecs, fold_ids) for name, vecs in stages.items()
    }

    raw_stats = {str(p["pdb_id"]).upper(): _structure_stats(p) for p in proteins}
    depth_vs_bb = []
    for pdb, bb in stages["backbone"].items():
        d = stages["depth_mean"][pdb][0]
        depth_vs_bb.append((d, float(np.linalg.norm(bb))))
    if len(depth_vs_bb) >= 2:
        d_arr = np.array([x[0] for x in depth_vs_bb])
        n_arr = np.array([x[1] for x in depth_vs_bb])
        corr = float(np.corrcoef(d_arr, n_arr)[0, 1])
    else:
        corr = 0.0

    del model
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "label": label,
        "checkpoint": str(checkpoint),
        "domain_labelled_structures": domain_label_count,
        "depth_vs_backbone_norm_corr": corr,
        "stage_summaries": stage_summaries,
        "raw_input_stats": raw_stats,
    }


def _verdict(report: dict[str, Any]) -> str:
    """Classify encoder pathology from final checkpoint stage summaries."""
    final = report.get("checkpoints", [{}])[-1]
    stages = final.get("stage_summaries", {})
    raw_cos = stages.get("raw_input", {}).get("pairwise_cosine", {}).get("pairwise_mean", 0.0)
    bb_cos = stages.get("backbone", {}).get("pairwise_cosine", {}).get("pairwise_mean", 1.0)
    disc_cos = stages.get("disc_2d_mean", {}).get("pairwise_cosine", {}).get("pairwise_mean", 1.0)

    if raw_cos > 0.95 and bb_cos > 0.97:
        return "inputs_and_encoder_both_collapsed"
    if raw_cos < 0.90 and bb_cos > 0.97:
        return "encoder_collapses_discriminative_inputs"
    if bb_cos > 0.97 and disc_cos > 0.98:
        return "encoder_fold_blind_confirmed"
    return "mixed_signal_review"


def main() -> None:
    parser = argparse.ArgumentParser(description="Encoder pathology audit")
    parser.add_argument("--run-dir", type=Path, required=True)
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

    checkpoints: list[tuple[str, Path]] = []
    for name in ("v6_phase1_12prot.pt", "v6_phase2_12prot.pt", "phase_1.pt", "phase_2.pt"):
        p = args.run_dir / name
        if p.is_file():
            checkpoints.append((name, p))
    if not checkpoints:
        raise SystemExit(f"No checkpoints found in {args.run_dir}")

    report: dict[str, Any] = {
        "run_dir": str(args.run_dir),
        "n_structures": len(proteins),
        "structures": [
            {
                "pdb_id": p["pdb_id"],
                "fold_id": p.get("fold_id"),
                "has_domain_labels": p.get("domain_labels") is not None,
            }
            for p in proteins
        ],
        "checkpoints": [],
    }

    metrics_path = args.run_dir / "metrics.json"
    if metrics_path.is_file():
        report["loss_contribution_audit"] = _loss_contribution_audit(metrics_path)

    for label, ckpt in checkpoints:
        logger.info("Auditing %s", label)
        report["checkpoints"].append(audit_checkpoint(ckpt, proteins, args.device, label))

    report["verdict"] = _verdict(report)
    report["interpretation"] = (
        "Compare raw_input vs backbone pairwise cosine: if raw separates folds but backbone "
        "does not, the encoder is erasing fold signal. If both are high, inputs may be "
        "depth-dominated before the GNN. domain_separation loss is intra-protein (4OBE only), "
        "not cross-fold — absence of fold-discrimination objective is expected."
    )

    out = args.output or (args.run_dir / "encoder_pathology_report.json")
    out.write_text(json.dumps(report, indent=2))
    logger.info("Wrote %s (verdict=%s)", out, report["verdict"])
    print(json.dumps({"verdict": report["verdict"], "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
