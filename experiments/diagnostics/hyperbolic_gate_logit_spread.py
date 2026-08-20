"""Pre-softmax HyperbolicPrototypeGate score spread (Fix-1 / IBU commitment diag).

Answers two questions on a feeler checkpoint:

1. Are hyp distance / raw logits narrowly compressed (geometry bottleneck),
   or already peaked enough that softmax/T would commit if rewarded?
2. Does per-residue logit range tighten near the disc origin vs rim
   (hyperbolic saturation asymmetry)?

Also reports a Euclidean tangent surrogate on the same fused trunk
(``-scale * ||logmap0(fused)-logmap0(proto)||``) for dynamic-range contrast.

Usage:
  python -m experiments.diagnostics.hyperbolic_gate_logit_spread \\
    --checkpoint checkpoints/v66/runs/fix1_s4_stage_a12_cold_v1/epochs/epoch_020.pt \\
    --corpus manifests/v6_corpus_stage_a_small_v1.json \\
    --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from geoopt.manifolds.stereographic import math as pmath

from experiments.training.v6.corpus import load_training_proteins
from experiments.training.v6.train_loop import prepare_training_batch
from science.training.gnn_lineage import load_model_from_checkpoint


def _match_x(model: torch.nn.Module, data: Any) -> Any:
    in_f = int(getattr(model.node_emb, "in_features", data.x.size(-1)))
    if data.x.size(-1) > in_f:
        data.x = data.x[:, :in_f].contiguous()
    return data


def _quantile_bins(x: np.ndarray, n_bins: int = 4) -> np.ndarray:
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(x, qs))
    if edges.size < 3:
        return np.zeros(x.shape[0], dtype=np.int64)
    return np.clip(np.digitize(x, edges[1:-1], right=False), 0, n_bins - 1)


def _spread_stats(logits: np.ndarray, soft: np.ndarray) -> dict[str, float]:
    """logits/soft: [N, E]."""
    row_range = logits.max(axis=1) - logits.min(axis=1)
    row_std = logits.std(axis=1)
    max_soft = soft.max(axis=1)
    return {
        "n": int(logits.shape[0]),
        "logit_std_global": float(logits.std()),
        "logit_row_range_mean": float(row_range.mean()),
        "logit_row_range_median": float(np.median(row_range)),
        "logit_row_range_p10": float(np.percentile(row_range, 10)),
        "logit_row_range_p90": float(np.percentile(row_range, 90)),
        "logit_row_std_mean": float(row_std.mean()),
        "frac_max_softmax_ge_0_40": float(np.mean(max_soft >= 0.40)),
        "frac_max_softmax_ge_0_50": float(np.mean(max_soft >= 0.50)),
        "frac_max_softmax_ge_0_60": float(np.mean(max_soft >= 0.60)),
        "mean_max_softmax": float(max_soft.mean()),
        "soft_load": [float(x) for x in soft.mean(axis=0)],
    }


def _bin_spread(
    logits: np.ndarray,
    soft: np.ndarray,
    radial: np.ndarray,
    *,
    label: str,
    n_bins: int = 4,
) -> dict[str, Any]:
    bins = _quantile_bins(radial, n_bins=n_bins)
    out: dict[str, Any] = {"radial": label, "n_bins": int(n_bins), "bins": []}
    for b in range(int(bins.max()) + 1 if bins.size else 0):
        mask = bins == b
        if int(mask.sum()) < 8:
            continue
        r = radial[mask]
        stats = _spread_stats(logits[mask], soft[mask])
        stats.update(
            {
                "bin": int(b),
                "n_residues": int(mask.sum()),
                "r_mean": float(r.mean()),
                "r_min": float(r.min()),
                "r_max": float(r.max()),
            }
        )
        out["bins"].append(stats)
    # Correlation: logit row-range vs radial (origin-tight → negative or near-zero).
    if logits.shape[0] > 8 and float(np.std(radial)) > 1e-8:
        row_range = logits.max(axis=1) - logits.min(axis=1)
        if float(np.std(row_range)) > 1e-8:
            out["corr_row_range_vs_radial"] = float(
                np.corrcoef(row_range, radial)[0, 1]
            )
        else:
            out["corr_row_range_vs_radial"] = float("nan")
    else:
        out["corr_row_range_vs_radial"] = float("nan")
    return out


def _eucl_tangent_logits(
    fused_hyp: torch.Tensor,
    proto_hyp: torch.Tensor,
    k: torch.Tensor,
    scale: float,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Same trunk, Euclidean distance in tangent at 0 (not a trained gate)."""
    fused_t = pmath.logmap0(fused_hyp, k=k)
    proto_t = pmath.logmap0(proto_hyp, k=k)
    # [N, E, D]
    diff = fused_t.unsqueeze(1) - proto_t.unsqueeze(0)
    dists = diff.norm(dim=-1)
    return -float(scale) * dists + bias.unsqueeze(0)


def _collect_protein(
    model: torch.nn.Module,
    prot: dict[str, Any],
    device: str,
) -> dict[str, np.ndarray] | None:
    gate = model.gate
    data = prepare_training_batch(
        model, prot, device, structural_disc_frozen=False
    )
    data = _match_x(model, data)
    with torch.no_grad():
        out = model(data)
    stash = getattr(gate, "_last_pre_softmax", None)
    if not isinstance(stash, dict) or "raw_logits" not in stash:
        return None

    raw = stash["raw_logits"]
    soft = F.softmax(raw, dim=-1)
    dists = stash["dists"]
    fused = stash["fused_hyp"]
    proto = stash["proto_hyp"]
    k = stash["k"]
    scale = float(stash["logit_scale"])
    bias = gate.expert_bias.detach()
    eucl = _eucl_tangent_logits(fused, proto, k, scale, bias)

    disc = out["hyp_projections_2d"].detach()
    disc_r = disc.norm(dim=-1)
    fused_r = fused.norm(dim=-1)
    depth = out["cone_depth"].detach().reshape(-1)

    return {
        "raw_logits": raw.cpu().numpy(),
        "soft": soft.cpu().numpy(),
        "dists": dists.cpu().numpy(),
        "eucl_logits": eucl.cpu().numpy(),
        "eucl_soft": F.softmax(eucl, dim=-1).cpu().numpy(),
        "disc_r": disc_r.cpu().numpy(),
        "fused_r": fused_r.cpu().numpy(),
        "depth": depth.cpu().numpy(),
    }


def _vstack(parts: list[dict[str, np.ndarray]], key: str) -> np.ndarray:
    return np.concatenate([p[key] for p in parts], axis=0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="HyperbolicPrototypeGate pre-softmax logit-spread diagnostic"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("manifests/v6_corpus_stage_a_small_v1.json"),
    )
    parser.add_argument("--pdb-dir", type=Path, default=Path("/tmp/dtie_pdb_cache"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-proteins", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/v66/diagnostics/hyperbolic_gate_logit_spread"),
    )
    args = parser.parse_args(argv)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = load_model_from_checkpoint(args.checkpoint, device, lineage_id="v6.6")
    model.eval()
    if not getattr(model, "hyperbolic_gate", False):
        raise SystemExit("checkpoint gate is not hyperbolic — this audit expects HyperbolicPrototypeGate")

    proteins, _failed = load_training_proteins(
        args.pdb_dir,
        args.corpus,
        max_proteins=args.max_proteins,
        max_residues=1200,
    )

    parts: list[dict[str, np.ndarray]] = []
    per_protein: list[dict[str, Any]] = []
    for prot in proteins:
        pdb = str(prot.get("pdb_id", "?")).upper()
        try:
            bundle = _collect_protein(model, prot, device)
        except Exception as exc:  # noqa: BLE001 — surface per-protein failure, continue
            per_protein.append({"pdb_id": pdb, "error": str(exc)})
            continue
        if bundle is None:
            per_protein.append({"pdb_id": pdb, "error": "missing _last_pre_softmax stash"})
            continue
        parts.append(bundle)
        hyp_s = _spread_stats(bundle["raw_logits"], bundle["soft"])
        eucl_s = _spread_stats(bundle["eucl_logits"], bundle["eucl_soft"])
        per_protein.append(
            {
                "pdb_id": pdb,
                "n": hyp_s["n"],
                "hyp_logit_row_range_mean": hyp_s["logit_row_range_mean"],
                "hyp_frac_max_softmax_ge_0_50": hyp_s["frac_max_softmax_ge_0_50"],
                "eucl_logit_row_range_mean": eucl_s["logit_row_range_mean"],
                "eucl_frac_max_softmax_ge_0_50": eucl_s["frac_max_softmax_ge_0_50"],
                "soft_load": hyp_s["soft_load"],
            }
        )

    if not parts:
        raise SystemExit("no proteins produced gate stash")

    raw = _vstack(parts, "raw_logits")
    soft = _vstack(parts, "soft")
    dists = _vstack(parts, "dists")
    eucl = _vstack(parts, "eucl_logits")
    eucl_soft = _vstack(parts, "eucl_soft")
    disc_r = _vstack(parts, "disc_r")
    fused_r = _vstack(parts, "fused_r")
    depth = _vstack(parts, "depth")

    gate = model.gate
    topo_only = bool(getattr(gate, "topology_only", False))
    hyp = _spread_stats(raw, soft)
    eucl_stats = _spread_stats(eucl, eucl_soft)
    dist_row_range = dists.max(axis=1) - dists.min(axis=1)

    report: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "corpus": str(args.corpus),
        "n_proteins": len(parts),
        "n_residues": int(raw.shape[0]),
        "topology_only_gate": topo_only,
        "logit_scale_softplus": float(
            getattr(gate, "_last_pre_softmax", {}).get("logit_scale", float("nan"))
        ),
        "formula": "logit_e = -softplus(scale) * d_H(fused, proto_e) + bias_e",
        "hyperbolic": hyp,
        "euclidean_tangent_surrogate": eucl_stats,
        "hyp_vs_eucl_row_range_ratio": float(
            hyp["logit_row_range_mean"] / max(eucl_stats["logit_row_range_mean"], 1e-12)
        ),
        "dist_stats": {
            "mean": float(dists.mean()),
            "std": float(dists.std()),
            "row_range_mean": float(dist_row_range.mean()),
            "row_range_median": float(np.median(dist_row_range)),
            "row_range_p90": float(np.percentile(dist_row_range, 90)),
        },
        "radial_asymmetry": {
            "vs_disc_r": _bin_spread(raw, soft, disc_r, label="disc_r"),
            "vs_fused_r": _bin_spread(raw, soft, fused_r, label="fused_hyp_r"),
            "vs_depth": _bin_spread(raw, soft, depth, label="cone_depth"),
        },
        "histogram_logit_row_range": {
            "edges": [float(x) for x in np.linspace(0.0, max(float(raw.max() - raw.min()), 1e-6), 11)],
            "counts": [
                int(c)
                for c in np.histogram(
                    raw.max(axis=1) - raw.min(axis=1),
                    bins=np.linspace(
                        0.0, max(float((raw.max(axis=1) - raw.min(axis=1)).max()), 1e-6), 11
                    ),
                )[0]
            ],
        },
        "per_protein": per_protein,
        "read_guide": {
            "narrow_hyp_flat_softmax": (
                "hyp logit_row_range_mean ≪ 1 and frac_max_softmax_ge_0_50 near 0 "
                "→ distances/logits compressed; commitment lever unlikely to fix alone "
                "unless distance is rescaled before softmax"
            ),
            "origin_rim_asymmetry": (
                "if inner disc_r/fused_r bins have smaller row_range than rim bins "
                "and corr_row_range_vs_radial > 0 → supports hyperbolic saturation story"
            ),
            "eucl_contrast": (
                "if eucl surrogate has much larger row_range on the same trunk, "
                "metric geometry (not feature poverty) is limiting dynamic range"
            ),
        },
    }

    # Plain-language verdicts (thresholds are diagnostic heuristics, not gates).
    rr = hyp["logit_row_range_mean"]
    frac50 = hyp["frac_max_softmax_ge_0_50"]
    corr_disc = report["radial_asymmetry"]["vs_disc_r"].get("corr_row_range_vs_radial")
    if rr < 0.5 and frac50 < 0.05:
        compression = "NARROW_HYPERBOLIC_SCORES"
    elif rr >= 1.5 and frac50 >= 0.15:
        compression = "REASONABLE_SCORE_SPREAD"
    else:
        compression = "AMBIGUOUS_SCORE_SPREAD"
    asym = "UNKNOWN"
    if corr_disc is not None and corr_disc == corr_disc:  # not NaN
        if float(corr_disc) >= 0.15:
            asym = "ORIGIN_TIGHTER_THAN_RIM"
        elif float(corr_disc) <= -0.15:
            asym = "RIM_TIGHTER_THAN_ORIGIN"
        else:
            asym = "NO_CLEAR_RADIAL_ASYMMETRY"
    report["verdict"] = {
        "score_compression": compression,
        "radial_asymmetry": asym,
        "hyp_logit_row_range_mean": rr,
        "hyp_frac_max_softmax_ge_0_50": frac50,
        "corr_row_range_vs_disc_r": corr_disc,
        "hyp_vs_eucl_row_range_ratio": report["hyp_vs_eucl_row_range_ratio"],
    }

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "report.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps(report["verdict"], indent=2))
    print(f"hyperbolic soft_load={hyp['soft_load']}")
    print(
        f"hyp row_range_mean={rr:.4f} frac_max≥0.5={frac50:.3f} | "
        f"eucl row_range_mean={eucl_stats['logit_row_range_mean']:.4f} | "
        f"ratio hyp/eucl={report['hyp_vs_eucl_row_range_ratio']:.3f}"
    )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
