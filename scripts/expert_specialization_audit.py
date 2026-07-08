#!/usr/bin/env python3
"""Per-protein / per-expert specialization audit for v6 checkpoints."""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.training.v6.assess_checkpoint import load_v6_model
from experiments.training.v6.corpus import load_training_proteins
from experiments.training.v6.train_loop import attach_v6_features, prepare_training_batch


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3:
        return None
    if np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _ss_label(ss: float) -> str:
    if ss < 0.25:
        return "helix"
    if ss < 0.75:
        return "sheet"
    return "coil"


def analyze_checkpoint(
    label: str,
    ckpt_path: Path,
    proteins: list[dict[str, Any]],
    device: str,
    *,
    structural_disc_frozen: bool = False,
) -> dict[str, Any]:
    model = load_v6_model(ckpt_path, device)
    n_exp = len(model.experts)
    global_assign = np.zeros(n_exp, dtype=int)
    global_n = 0

    per_protein: list[dict[str, Any]] = []
    expert_agg: dict[int, dict[str, list[float]]] = {
        e: {
            "depth": [],
            "disc_r": [],
            "tau": [],
            "rho": [],
            "epi": [],
            "soft_load": [],
        }
        for e in range(n_exp)
    }

    with torch.no_grad():
        for prot in proteins:
            if structural_disc_frozen:
                data = prepare_training_batch(
                    model, prot, device, structural_disc_frozen=True
                )
            else:
                data = attach_v6_features(prot["data"].clone().to(device))
            out = model(data)
            w = out["expert_weights"].detach().cpu().numpy()
            if w.ndim == 1:
                w = w.reshape(-1, 1)
            x = data.x.detach().cpu().numpy()
            rho, tau, ss = x[:, 0], x[:, 1], x[:, 2]
            cd = out["cone_depth"].squeeze().detach().cpu().numpy()
            hyp = out["hyp_projections_2d"].detach().cpu().numpy()
            disc_r = np.linalg.norm(hyp, axis=1) if hyp.ndim == 2 else np.abs(hyp)
            epi = out["uncertainty"]["epistemic"].squeeze().detach().cpu().numpy()
            assign = w.argmax(axis=1)

            load_mean = w.mean(axis=0)
            min_r = float(load_mean.min())
            ent = float(-(load_mean * np.log(load_mean + 1e-8)).sum())

            dom_counts = {int(e): int((assign == e).sum()) for e in range(n_exp)}
            global_assign += np.array([dom_counts[e] for e in range(n_exp)])
            global_n += len(assign)

            # per-expert on this protein (dominant-assignment stats)
            exp_stats = []
            for e in range(n_exp):
                mask = assign == e
                n = int(mask.sum())
                row: dict[str, Any] = {"expert": e, "n": n, "frac": n / len(assign)}
                if n >= 3:
                    row.update(
                        {
                            "depth_mean": float(cd[mask].mean()),
                            "disc_r_mean": float(disc_r[mask].mean()),
                            "tau_mean": float(tau[mask].mean()),
                            "rho_mean": float(rho[mask].mean()),
                            "epi_mean": float(epi[mask].mean()),
                            "r_depth_tau": _pearson(cd[mask], tau[mask]),
                            "ss_helix_frac": float((ss[mask] < 0.25).mean()),
                            "ss_sheet_frac": float(((ss[mask] >= 0.25) & (ss[mask] < 0.75)).mean()),
                            "ss_coil_frac": float((ss[mask] >= 0.75).mean()),
                        }
                    )
                    expert_agg[e]["depth"].extend(cd[mask].tolist())
                    expert_agg[e]["disc_r"].extend(disc_r[mask].tolist())
                    expert_agg[e]["tau"].extend(tau[mask].tolist())
                    expert_agg[e]["rho"].extend(rho[mask].tolist())
                    expert_agg[e]["epi"].extend(epi[mask].tolist())
                expert_agg[e]["soft_load"].append(float(load_mean[e]))
                exp_stats.append(row)

            per_protein.append(
                {
                    "pdb_id": prot["pdb_id"],
                    "gene": prot.get("gene"),
                    "fold_id": prot.get("fold_id"),
                    "n_res": len(assign),
                    "routing_entropy": ent,
                    "min_routing_fraction": min_r,
                    "load_mean": [float(x) for x in load_mean],
                    "dominant_counts": dom_counts,
                    "experts": exp_stats,
                }
            )

    experts_summary = []
    for e in range(n_exp):
        agg = expert_agg[e]
        depths = np.asarray(agg["depth"], dtype=float)
        taus = np.asarray(agg["tau"], dtype=float)
        rhos = np.asarray(agg["rho"], dtype=float)
        discs = np.asarray(agg["disc_r"], dtype=float)
        epis = np.asarray(agg["epi"], dtype=float)
        row: dict[str, Any] = {
            "expert": e,
            "global_dominant_frac": float(global_assign[e] / max(global_n, 1)),
            "mean_soft_load": float(np.mean(agg["soft_load"])) if agg["soft_load"] else 0.0,
        }
        if len(depths) >= 3:
            row.update(
                {
                    "depth_mean": float(depths.mean()),
                    "depth_std": float(depths.std()),
                    "disc_r_mean": float(discs.mean()),
                    "disc_r_std": float(discs.std()),
                    "tau_mean": float(taus.mean()),
                    "tau_std": float(taus.std()),
                    "rho_mean": float(rhos.mean()),
                    "epi_mean": float(epis.mean()),
                    "r_depth_tau": _pearson(depths, taus),
                    "r_depth_rho": _pearson(depths, rhos),
                    "n_assigned": int(len(depths)),
                }
            )
        experts_summary.append(row)

    return {
        "label": label,
        "checkpoint": str(ckpt_path),
        "structural_disc_frozen": structural_disc_frozen,
        "num_experts": n_exp,
        "experts": experts_summary,
        "proteins": per_protein,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=ROOT / "manifests/v6_corpus_stage_a_small_v1.json")
    parser.add_argument("--pdb-dir", type=Path, default=Path("/tmp/dtie_pdb_cache"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "checkpoints/v6/runs/expert_specialization_audit.json")
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=None,
        help="label=path (repeatable). Default: slim MoE cold + route checkpoints.",
    )
    parser.add_argument(
        "--structural-disc-frozen",
        action="store_true",
        help="Attach structural SSOT + hyperbolic graph before forward",
    )
    args = parser.parse_args()

    proteins, _ = load_training_proteins(
        args.pdb_dir,
        manifest_path=args.corpus,
        max_proteins=12,
        max_residues=650,
    )

    if args.checkpoint:
        checkpoints = []
        for spec in args.checkpoint:
            label, _, path = spec.partition("=")
            checkpoints.append((label, Path(path)))
    else:
        checkpoints = [
            (
                "slim_moe_structural_ssot_cold_v1",
                ROOT / "checkpoints/v6/runs/slim_moe_structural_ssot_cold_v1/v6_best.pt",
            ),
            (
                "slim_moe_route_v1",
                ROOT / "checkpoints/v6/runs/slim_moe_route_v1/v6_best.pt",
            ),
        ]

    results = []
    for label, path in checkpoints:
        if not path.is_file():
            print(f"skip {label}: missing {path}")
            continue
        print(f"analyzing {label} (structural_ssot={args.structural_disc_frozen})...")
        results.append(
            analyze_checkpoint(
                label,
                path,
                proteins,
                args.device,
                structural_disc_frozen=args.structural_disc_frozen,
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
