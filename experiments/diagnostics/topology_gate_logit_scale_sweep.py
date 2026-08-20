"""Forward-only logit_scale sweep on frozen topology-only HyperbolicPrototypeGate.

Distinguishes:
  (2) SCALE_LIMITED — H drops when softplus(logit_scale) is raised (2×/5×/10×)
  (1) BOARD_CEILING — H stays ≈ ln(N) even at aggressive scale

No training; mutates ``gate.logit_scale`` in-memory only.

Usage:
  python -m experiments.diagnostics.topology_gate_logit_scale_sweep \\
    --checkpoint checkpoints/v66/runs/fix1_s4_t1a_znorm_stage_a12_cold_v1/epochs/epoch_020.pt
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from experiments.training.v6.corpus import load_training_proteins
from experiments.training.v6.train_loop import prepare_training_batch
from science.training.gnn_lineage import load_model_from_checkpoint


def _inv_softplus(y: float) -> float:
    """Inverse of softplus for y > 0."""
    y = float(max(y, 1e-8))
    if y > 20.0:
        return y
    return float(math.log(math.expm1(y)))


def _mi_discrete(x: np.ndarray, y: np.ndarray) -> float:
    """Mutual information in nats for integer label arrays."""
    x = np.asarray(x, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    n = int(x.shape[0])
    if n == 0:
        return float("nan")
    px: dict[int, int] = {}
    py: dict[int, int] = {}
    pxy: dict[tuple[int, int], int] = {}
    for a, b in zip(x.tolist(), y.tolist()):
        px[a] = px.get(a, 0) + 1
        py[b] = py.get(b, 0) + 1
        pxy[(a, b)] = pxy.get((a, b), 0) + 1
    mi = 0.0
    for (a, b), c in pxy.items():
        p_ab = c / n
        mi += p_ab * math.log(p_ab / ((px[a] / n) * (py[b] / n)))
    return float(mi)


def _pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _structure_effect_sizes(
    soft: np.ndarray,
    *,
    tau: np.ndarray,
    depth: np.ndarray,
    n_null: int = 64,
    seed: int = 0,
) -> dict[str, Any]:
    """Same family as IBU geom_prior_cold_to20_routing_null effect sizes."""
    hard = soft.argmax(axis=1).astype(np.int64)
    tau_i = (np.asarray(tau, dtype=np.float64) > 0.5).astype(np.int64)
    # Depth quartiles on pooled corpus (ties → digitize edges).
    d = np.asarray(depth, dtype=np.float64).reshape(-1)
    qs = np.quantile(d, [0.25, 0.5, 0.75])
    depth_q = np.digitize(d, qs, right=False).astype(np.int64)

    mi_tau = _mi_discrete(hard, tau_i)
    mi_depth = _mi_discrete(hard, depth_q)
    rng = np.random.default_rng(seed)
    null_tau = []
    null_depth = []
    for _ in range(n_null):
        null_tau.append(_mi_discrete(hard, rng.permutation(tau_i)))
        null_depth.append(_mi_discrete(hard, rng.permutation(depth_q)))
    null_tau_m = float(np.mean(null_tau))
    null_depth_m = float(np.mean(null_depth))

    soft_r: dict[str, float] = {}
    soft_r2: dict[str, float] = {}
    for e in range(soft.shape[1]):
        r_tau = _pearson_r(soft[:, e], tau_i.astype(np.float64))
        r_depth = _pearson_r(soft[:, e], d)
        soft_r[f"e{e}_vs_tau"] = r_tau
        soft_r[f"e{e}_vs_depth"] = r_depth
        soft_r2[f"e{e}_vs_tau"] = float(r_tau * r_tau) if np.isfinite(r_tau) else float("nan")
        soft_r2[f"e{e}_vs_depth"] = (
            float(r_depth * r_depth) if np.isfinite(r_depth) else float("nan")
        )
    abs_r_tau = [abs(soft_r[f"e{e}_vs_tau"]) for e in range(soft.shape[1])]
    abs_r_depth = [abs(soft_r[f"e{e}_vs_depth"]) for e in range(soft.shape[1])]
    peak_abs_r_tau = float(np.nanmax(abs_r_tau))
    peak_abs_r_depth = float(np.nanmax(abs_r_depth))
    peak_r2_tau = float(np.nanmax([soft_r2[f"e{e}_vs_tau"] for e in range(soft.shape[1])]))
    peak_r2_depth = float(
        np.nanmax([soft_r2[f"e{e}_vs_depth"] for e in range(soft.shape[1])])
    )

    p_hard = np.bincount(hard, minlength=soft.shape[1]).astype(np.float64)
    p_hard = p_hard / max(p_hard.sum(), 1.0)
    p_hard = p_hard[p_hard > 0]
    h_hard = float(-(p_hard * np.log(p_hard)).sum())

    return {
        "mi_hard_vs_tau_nats": mi_tau,
        "mi_hard_vs_depth_quartile_nats": mi_depth,
        "mi_null_tau_mean_nats": null_tau_m,
        "mi_null_depth_mean_nats": null_depth_m,
        "mi_tau_over_null": mi_tau / null_tau_m if null_tau_m > 1e-12 else float("inf"),
        "mi_depth_over_null": (
            mi_depth / null_depth_m if null_depth_m > 1e-12 else float("inf")
        ),
        "H_expert_hard_nats": h_hard,
        "U_expert_given_tau": mi_tau / h_hard if h_hard > 1e-12 else float("nan"),
        "soft_abs_r": soft_r,
        "soft_R2": soft_r2,
        "peak_abs_r_soft_vs_tau": peak_abs_r_tau,
        "peak_abs_r_soft_vs_depth": peak_abs_r_depth,
        "peak_R2_soft_vs_tau": peak_r2_tau,
        "peak_R2_soft_vs_depth": peak_r2_depth,
        # Frozen USAGE_MOVED feature-hold floor (from GNNV7_SUCCESS_CRITERIA).
        "feature_hold_pass": bool(peak_abs_r_tau >= 0.50 or mi_tau >= 0.40),
    }


def _routing_pack(soft: np.ndarray) -> dict[str, Any]:
    load = soft.mean(axis=0)
    h = float(-(load * np.log(load + 1e-12)).sum())
    max_share = float(load.max())
    min_share = float(load.min())
    return {
        "H": h,
        "H_minus_lnN": h - float(math.log(load.size)),
        "max_soft_share": max_share,
        "min_soft_share": min_share,
        "load_spread": max_share - min_share,
        "expert_load": [float(x) for x in load],
        "mean_max_softmax": float(soft.max(axis=1).mean()),
        "frac_max_softmax_ge_0_40": float(np.mean(soft.max(axis=1) >= 0.40)),
        "frac_max_softmax_ge_0_50": float(np.mean(soft.max(axis=1) >= 0.50)),
        "n_residues": int(soft.shape[0]),
    }


def _usage_moved_letter(pack: dict[str, Any], feature_hold: bool) -> dict[str, Any]:
    """Exact frozen USAGE_MOVED / IBU_HOLDS / AMBIGUOUS letter (usage side only + feature)."""
    mx = float(pack["max_soft_share"])
    sp = float(pack["load_spread"])
    h = float(pack["H"])
    usage_moved = mx >= 0.38 and sp >= 0.12 and h <= 1.30 and feature_hold
    ibu = mx < 0.34 and sp < 0.08 and h > 1.34
    if usage_moved:
        label = "USAGE_MOVED"
    elif ibu:
        label = "IBU_HOLDS"
    else:
        label = "AMBIGUOUS"
    return {
        "usage_gate_letter": label,
        "usage_moved_checks": {
            "max_ge_0_38": mx >= 0.38,
            "spread_ge_0_12": sp >= 0.12,
            "H_le_1_30": h <= 1.30,
            "feature_hold": feature_hold,
            "max_soft_share": mx,
            "load_spread": sp,
            "H": h,
        },
    }


def _collect_soft_and_structure(
    model: torch.nn.Module,
    proteins: list[dict[str, Any]],
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns soft scores [N,E], tau_flag [N], cone_depth [N]."""
    chunks: list[np.ndarray] = []
    taus: list[np.ndarray] = []
    depths: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for prot in proteins:
            structural_frozen = bool(getattr(model, "structural_disc_frozen", False))
            data = prepare_training_batch(
                model, prot, device, structural_disc_frozen=structural_frozen
            )
            out = model(data)
            scores = out.get("expert_weights")
            if scores is None:
                for key in ("moe_scores", "scores", "routing_weights"):
                    if key in out and torch.is_tensor(out[key]):
                        scores = out[key]
                        break
            if scores is None:
                raise RuntimeError(f"no routing scores in output keys={list(out)[:30]}")
            chunks.append(scores.detach().cpu().float().numpy())
            # Prefer model outputs; fall back to graph node features.
            if "cone_depth" in out and torch.is_tensor(out["cone_depth"]):
                depths.append(out["cone_depth"].detach().cpu().float().reshape(-1).numpy())
            elif hasattr(data, "cone_depth") and data.cone_depth is not None:
                depths.append(data.cone_depth.detach().cpu().float().reshape(-1).numpy())
            else:
                # topology depth often in data / radial — last resort: rho channel
                depths.append(data.x[:, 0].detach().cpu().float().numpy())
            if data.x.size(-1) > 1:
                taus.append(data.x[:, 1].detach().cpu().float().numpy())
            else:
                taus.append(np.zeros(scores.shape[0], dtype=np.float32))
    return (
        np.concatenate(chunks, axis=0),
        np.concatenate(taus, axis=0),
        np.concatenate(depths, axis=0),
    )


def _maybe_install_znorm(model: torch.nn.Module, stats_path: Path | None) -> None:
    if stats_path is None or not stats_path.is_file():
        return
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    mean = torch.tensor(stats["mean"], dtype=torch.float32)
    std = torch.tensor(stats["std"], dtype=torch.float32)
    model.input_feature_zscore = bool(stats.get("input_feature_zscore", True))
    model.replace_tau_with_abs_dist = bool(stats.get("replace_tau_with_abs_dist", False))
    if hasattr(model, "input_feat_mean"):
        model.input_feat_mean.copy_(mean.to(model.input_feat_mean.device))
        model.input_feat_std.copy_(std.to(model.input_feat_std.device))
    else:
        model.register_buffer("input_feat_mean", mean)
        model.register_buffer("input_feat_std", std)


def run_sweep(
    model: torch.nn.Module,
    proteins: list[dict[str, Any]],
    device: str,
    *,
    multipliers: list[float],
) -> dict[str, Any]:
    gate = model.gate
    if not hasattr(gate, "logit_scale"):
        raise RuntimeError(
            f"gate type {type(gate).__name__} has no logit_scale — expected HyperbolicPrototypeGate"
        )
    topology_only = bool(getattr(gate, "topology_only", False))
    base_param = float(gate.logit_scale.detach().cpu())
    base_scale = float(F.softplus(torch.tensor(base_param)))
    bias = gate.expert_bias.detach().cpu().float().tolist()

    rows: list[dict[str, Any]] = []
    for m in multipliers:
        target = base_scale * float(m)
        gate.logit_scale.data.fill_(_inv_softplus(target))
        soft, tau, depth = _collect_soft_and_structure(model, proteins, device)
        pack = _routing_pack(soft)
        effects = _structure_effect_sizes(soft, tau=tau, depth=depth)
        pack.update(_usage_moved_letter(pack, bool(effects["feature_hold_pass"])))
        pack["structure"] = effects
        pack.update(
            {
                "scale_multiplier": float(m),
                "logit_scale_param": float(gate.logit_scale.detach().cpu()),
                "softplus_scale": float(F.softplus(gate.logit_scale.detach()).cpu()),
            }
        )
        rows.append(pack)

    # restore
    gate.logit_scale.data.fill_(base_param)

    base = rows[0]
    best = min(rows, key=lambda r: r["H"])
    # Pre-registered-ish: meaningful move off max-entropy pin
    moved = any(
        r["H"] <= 1.32 or r["max_soft_share"] >= 0.33 or r["load_spread"] >= 0.08
        for r in rows
        if r["scale_multiplier"] > 1.0
    )
    # Stay pinned: H span across sweep < 0.02 and never clears band
    h_span = float(max(r["H"] for r in rows) - min(r["H"] for r in rows))
    if moved:
        verdict = "SCALE_LIMITED"
        read = (
            "Raising softplus(logit_scale) alone moves H / load off the ln(N) pin — "
            "usable separation already exists in the topology board; cheap scalar fix. "
            "Exact USAGE_MOVED letter still applies per row (near-miss ≠ pass)."
        )
    elif h_span < 0.02:
        verdict = "BOARD_CEILING"
        read = (
            "H stays pinned near ln(N) even at 10× scale — board information (not logit "
            "temperature) is the ceiling. Enrich board (|ρ−TAU|) or change what the gate sees."
        )
    else:
        verdict = "AMBIGUOUS"
        read = (
            f"H moves slightly (span={h_span:.4f}) but does not clear soft ROUTING_RESPONDED "
            "thresholds — inspect row table before choosing a lever."
        )

    # Structure alignment under sharpening vs baseline
    base_mi = float(base["structure"]["mi_hard_vs_tau_nats"])
    base_r2_tau = float(base["structure"]["peak_R2_soft_vs_tau"])
    base_r2_depth = float(base["structure"]["peak_R2_soft_vs_depth"])
    structure_read: dict[str, Any] = {}
    for r in rows:
        if r["scale_multiplier"] <= 1.0:
            continue
        s = r["structure"]
        mi = float(s["mi_hard_vs_tau_nats"])
        r2t = float(s["peak_R2_soft_vs_tau"])
        r2d = float(s["peak_R2_soft_vs_depth"])
        # Signal-preserving: MI and peak R² do not drop more than 10% relative vs ×1
        ok = (
            mi >= 0.90 * base_mi
            and r2t >= 0.90 * base_r2_tau
            and r2d >= 0.90 * base_r2_depth
            and bool(s["feature_hold_pass"])
        )
        structure_read[f"x{int(r['scale_multiplier'])}"] = {
            "signal_preserving": ok,
            "mi_hard_tau": mi,
            "peak_R2_tau": r2t,
            "peak_R2_depth": r2d,
            "delta_mi_vs_x1": mi - base_mi,
            "delta_peak_R2_tau_vs_x1": r2t - base_r2_tau,
            "delta_peak_R2_depth_vs_x1": r2d - base_r2_depth,
        }

    return {
        "gate_type": type(gate).__name__,
        "topology_only": topology_only,
        "baseline_logit_scale_param": base_param,
        "baseline_softplus_scale": base_scale,
        "expert_bias": bias,
        "multipliers": multipliers,
        "rows": rows,
        "h_span": h_span,
        "best_H": best["H"],
        "best_multiplier": best["scale_multiplier"],
        "verdict": verdict,
        "read": read,
        "baseline_H": base["H"],
        "baseline_max_share": base["max_soft_share"],
        "x10_usage_gate_letter": next(
            (
                r["usage_gate_letter"]
                for r in rows
                if r["scale_multiplier"] == 10.0
            ),
            None,
        ),
        "structure_under_sharpening": structure_read,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Topology-gate logit_scale forward sweep")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("manifests/v6_corpus_stage_a_small_v1.json"),
    )
    parser.add_argument("--pdb-dir", type=Path, default=Path("/tmp/dtie_pdb_cache"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-proteins", type=int, default=12)
    parser.add_argument(
        "--znorm-stats",
        type=Path,
        default=None,
        help="Optional t1a_input_feature_norm.json (needed if ckpt was trained with z-norm)",
    )
    parser.add_argument(
        "--multipliers",
        type=str,
        default="1,2,5,10",
        help="Comma-separated softplus(scale) multipliers vs checkpoint baseline",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    device = args.device
    if device != "cpu" and not torch.cuda.is_available():
        device = "cpu"

    proteins, failed = load_training_proteins(
        args.pdb_dir,
        args.corpus,
        max_proteins=args.max_proteins,
        use_cache=True,
    )
    if not proteins:
        raise SystemExit(f"no proteins loaded ({failed} failed)")

    model = load_model_from_checkpoint(args.checkpoint, device)
    stats_path = args.znorm_stats
    if stats_path is None:
        # Default: sibling run norm file if present
        cand = args.checkpoint.parents[1] / "t1a_input_feature_norm.json"
        if cand.is_file():
            stats_path = cand
    _maybe_install_znorm(model, stats_path)

    multipliers = [float(x.strip()) for x in args.multipliers.split(",") if x.strip()]
    report = run_sweep(model, proteins, device, multipliers=multipliers)
    report["checkpoint"] = str(args.checkpoint)
    report["corpus"] = str(args.corpus)
    report["n_proteins"] = len(proteins)
    report["znorm_stats"] = str(stats_path) if stats_path else None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "report.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(
        f"topology_only={report['topology_only']} softplus0={report['baseline_softplus_scale']:.4f} "
        f"| H0={report['baseline_H']:.4f} max0={report['baseline_max_share']:.3f}"
    )
    for row in report["rows"]:
        s = row.get("structure") or {}
        print(
            f"  ×{row['scale_multiplier']:.0f}: softplus={row['softplus_scale']:.3f} "
            f"H={row['H']:.4f} max={row['max_soft_share']:.3f} spread={row['load_spread']:.3f} "
            f"mean_max_p={row['mean_max_softmax']:.3f} | "
            f"letter={row.get('usage_gate_letter')} | "
            f"MI(τ)={s.get('mi_hard_vs_tau_nats', float('nan')):.3f} "
            f"R²τ={s.get('peak_R2_soft_vs_tau', float('nan')):.3f} "
            f"R²d={s.get('peak_R2_soft_vs_depth', float('nan')):.3f}"
        )
    print(f"{report['verdict']}: {report['read']}")
    print(f"x10 letter={report.get('x10_usage_gate_letter')} | structure={report.get('structure_under_sharpening')}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
