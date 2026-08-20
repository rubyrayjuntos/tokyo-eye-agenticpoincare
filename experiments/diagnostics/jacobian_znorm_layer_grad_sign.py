#!/usr/bin/env python3
"""Layer-by-layer activation vs gradient polarity: data.x → probe encoder_h.

Standing defect: Jacobian flow-influence anti-correlates under z-norm-trained
trunks while direct PC1 correlates. Hyperbolic ops are *off* this path (probe
hooks radial_head input). Ordered suspects: LayerNorm × N, SH normalize, input
z-score×LN interaction — see
``docs/specs/learned-flow-influence/JACOBIAN_ZNORM_DEFECT.md``.

Cheap check (no training): one structure, one z-norm-on checkpoint, forward +
one backward of ``s_B = (encoder_h[B]·û)²``. At each tap, compare activation
vs ``∂s/∂tap``.

Usage:
  GNN_INPUT_MODE=topology_three_vector python -m \\
    experiments.diagnostics.jacobian_znorm_layer_grad_sign \\
    --checkpoint checkpoints/v66/runs/chem_mvp_znorm_stage_a12_cold_v1/v66_best.pt \\
    --pdb-id 4OBE
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from experiments.diagnostics.jacobian_flow_influence import (
    EPS,
    jacobian_probe_node_count,
    load_proteins_from_cache,
    pc1_unit_direction,
    score_scalar,
    spearman_corr,
)
from experiments.training.v6.train_loop import prepare_training_batch
from science.dtie.common.classical_network_metrics import classical_network_metrics
from science.dtie.common.input_feature_norm import transform_node_features
from science.dtie.v66.gnn.model import resolve_message_passing_edges
from science.training.gnn_lineage import load_model_from_checkpoint

_REPO = Path(__file__).resolve().parents[2]
_DEFAULT_OUT = (
    _REPO
    / "checkpoints"
    / "v66"
    / "diagnostics"
    / "learned_flow_influence"
    / "jacobian_znorm_layer_grad_sign"
)


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).double()
    b = b.reshape(-1).double()
    if a.numel() == 0 or b.numel() == 0:
        return float("nan")
    na = torch.linalg.vector_norm(a)
    nb = torch.linalg.vector_norm(b)
    if float(na) < EPS or float(nb) < EPS:
        return float("nan")
    return float((a * b).sum() / (na * nb))


def _sign_agree(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).double()
    b = b.reshape(-1).double()
    mask = (a.abs() > EPS) & (b.abs() > EPS)
    if int(mask.sum()) == 0:
        return float("nan")
    return float((torch.sign(a[mask]) == torch.sign(b[mask])).float().mean())


def polarity_pack(act: torch.Tensor, grad: torch.Tensor | None) -> dict[str, Any]:
    if grad is None:
        return {
            "has_grad": False,
            "act_l2": float(torch.linalg.vector_norm(act.detach()).cpu()),
            "grad_l2": None,
            "cos_act_grad": None,
            "sign_agree_frac": None,
        }
    return {
        "has_grad": True,
        "act_l2": float(torch.linalg.vector_norm(act.detach()).cpu()),
        "grad_l2": float(torch.linalg.vector_norm(grad.detach()).cpu()),
        "cos_act_grad": _cos(act.detach(), grad.detach()),
        "sign_agree_frac": _sign_agree(act.detach(), grad.detach()),
    }


def per_residue_cos_act_grad(
    act: torch.Tensor,
    grad: torch.Tensor | None,
    *,
    n: int,
) -> np.ndarray:
    """Per-row cos(act[i], grad[i]) for the first ``n`` residues (NaN if missing)."""
    out = np.full(n, np.nan, dtype=np.float64)
    if grad is None or act.ndim != 2 or act.shape[0] < n:
        return out
    a = act[:n].detach()
    g = grad[:n].detach()
    for i in range(n):
        out[i] = _cos(a[i], g[i])
    return out


def cos_distribution_summary(cos: np.ndarray) -> dict[str, Any]:
    """Shape of per-residue act·grad cosines — attenuation vs incoherence."""
    arr = np.asarray(cos, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "median": None,
            "q10": None,
            "q90": None,
            "min": None,
            "max": None,
            "frac_positive": None,
            "frac_negative": None,
            "frac_abs_gt_0_5": None,
            "frac_abs_lt_0_1": None,
            "shape_read": "no_finite",
        }
    mean = float(np.mean(finite))
    std = float(np.std(finite))
    abs_lt_01 = float(np.mean(np.abs(finite) < 0.1))
    abs_gt_05 = float(np.mean(np.abs(finite) > 0.5))
    # Tight near mean≈0 → attenuation; wide ± → incoherence.
    # Compressed (most mass in [-0.25, +0.25], almost no |cos|>0.5) also counts
    # as attenuation even if frac_|cos|<0.1 is only ~half.
    q10 = float(np.quantile(finite, 0.10))
    q90 = float(np.quantile(finite, 0.90))
    if abs_lt_01 >= 0.7 and std < 0.25:
        shape = "attenuation_tight_near_zero"
    elif abs_gt_05 < 0.15 and abs(q10) < 0.3 and abs(q90) < 0.35 and std < 0.3:
        shape = "attenuation_compressed_near_zero"
    elif std >= 0.4 and abs_gt_05 >= 0.25:
        shape = "incoherence_wide_scatter"
    elif abs(mean) < 0.15 and std >= 0.25:
        shape = "mixed_near_zero_mean_with_spread"
    else:
        shape = "other"
    return {
        "n": int(finite.size),
        "mean": mean,
        "std": std,
        "median": float(np.median(finite)),
        "q10": q10,
        "q90": q90,
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "frac_positive": float(np.mean(finite > 0)),
        "frac_negative": float(np.mean(finite < 0)),
        "frac_abs_gt_0_5": abs_gt_05,
        "frac_abs_lt_0_1": abs_lt_01,
        "shape_read": shape,
    }


def forward_trunk_taps(
    model: torch.nn.Module,
    data: Any,
    *,
    device: str,
) -> tuple[torch.Tensor, list[dict[str, Any]], torch.Tensor]:
    """Replay Euclidean trunk (model steps 1–2); return encoder_h + tap list + x_raw.

    Each tap stores a live tensor (requires grad) named for the ordered op list.
    """
    model = model.double()
    x_raw = data.x.detach().to(device=device, dtype=torch.float64).requires_grad_(True)
    data.x = x_raw
    if hasattr(data, "edge_attr") and data.edge_attr is not None:
        data.edge_attr = data.edge_attr.detach().to(device=device, dtype=torch.float64)
    if hasattr(data, "edge_index") and data.edge_index is not None:
        data.edge_index = data.edge_index.to(device=device)

    taps: list[dict[str, Any]] = []
    taps.append({"name": "00_raw_x", "op": "data.x", "kind": "input", "tensor": x_raw})

    x_in = x_raw
    if bool(getattr(model, "input_feature_zscore", False)) or bool(
        getattr(model, "replace_tau_with_abs_dist", False)
    ):
        mean = getattr(model, "input_feat_mean", None)
        std = getattr(model, "input_feat_std", None)
        if mean is None or std is None:
            raise RuntimeError("z-norm flags set but input_feat_mean/std missing")
        x_in = transform_node_features(
            x_raw,
            mean=mean,
            std=std,
            zscore=bool(getattr(model, "input_feature_zscore", False))
            or bool(getattr(model, "replace_tau_with_abs_dist", False)),
            replace_tau_with_abs_dist=bool(
                getattr(model, "replace_tau_with_abs_dist", False)
            ),
        )
        taps.append(
            {
                "name": "01_after_input_zscore",
                "op": "transform_node_features (x−μ)/σ",
                "kind": "DIV",
                "tensor": x_in,
            }
        )

    x = model.node_emb(x_in)
    taps.append(
        {
            "name": "02_after_node_emb",
            "op": "node_emb Linear",
            "kind": "linear",
            "tensor": x,
        }
    )

    mp_edge_index, mp_edge_attr = resolve_message_passing_edges(data)
    for li, (conv, norm) in enumerate(zip(model.convs, model.norms)):
        x_res = x
        x_conv = conv(x, mp_edge_index, mp_edge_attr)
        taps.append(
            {
                "name": f"L{li}_a_after_conv",
                "op": f"EquivariantConvMultiRel[{li}] (+ SH normalize)",
                "kind": "NORM",
                "tensor": x_conv,
            }
        )
        x_ln = norm(x_conv)
        taps.append(
            {
                "name": f"L{li}_b_after_layernorm",
                "op": f"LayerNorm[{li}]",
                "kind": "DIV",
                "tensor": x_ln,
            }
        )
        x_act = F.silu(x_ln)
        taps.append(
            {
                "name": f"L{li}_c_after_silu",
                "op": f"SiLU[{li}]",
                "kind": "silu",
                "tensor": x_act,
            }
        )
        x = x_act + x_res
        taps.append(
            {
                "name": f"L{li}_d_after_residual",
                "op": f"residual add[{li}]",
                "kind": "residual",
                "tensor": x,
            }
        )

    taps.append(
        {
            "name": "99_encoder_h",
            "op": "encoder_h (= radial_head input)",
            "kind": "probe_site",
            "tensor": x,
        }
    )
    return x, taps, x_raw


def run_layer_grad_sign(
    *,
    model: torch.nn.Module,
    prot: dict[str, Any],
    device: str,
    target_b: int | None = None,
    high_btw_frac: float = 0.25,
) -> dict[str, Any]:
    model.eval()
    data = prepare_training_batch(model, prot, device)
    n = jacobian_probe_node_count(data, prot)
    ca = prot.get("ca_coords")
    ca_np = ca.detach().cpu().numpy() if torch.is_tensor(ca) else np.asarray(ca)
    classical = classical_network_metrics(ca_np[:n] if ca_np.shape[0] > n else ca_np)
    btw = classical["betweenness"]

    encoder_h, taps, _x_raw = forward_trunk_taps(model, data, device=device)
    encoder_h = encoder_h[:n]
    û = pc1_unit_direction(encoder_h)

    # Choose B: highest |PC1| among top-betweenness quartile (or explicit).
    proj = (encoder_h.detach() * û.unsqueeze(0)).sum(dim=-1).cpu().numpy()
    order = np.argsort(-btw)
    k = max(1, int(round(high_btw_frac * n)))
    high_idx = set(int(i) for i in order[:k])
    if target_b is None:
        # Prefer a high-btw residue with large |PC1| so s_B is informative.
        cand = sorted(high_idx, key=lambda i: abs(float(proj[i])), reverse=True)
        target_b = int(cand[0]) if cand else int(np.argmax(np.abs(proj)))
    if not (0 <= target_b < n):
        raise ValueError(f"target_b={target_b} out of range for n={n}")

    # Retain grads on every tap tensor.
    for tap in taps:
        t = tap["tensor"]
        if t.requires_grad:
            t.retain_grad()

    s_b = score_scalar(encoder_h[target_b], score_mode="pc1_sq", pc1_dir=û)
    s_b.backward()

    # Forward sanity: PC1 separates high vs low betweenness?
    low_idx = set(int(i) for i in order[-k:])
    high_proj = float(np.mean([proj[i] for i in high_idx]))
    low_proj = float(np.mean([proj[i] for i in low_idx]))
    # Align û so high-btw mean proj is positive for readability.
    pc1_sign = 1.0 if high_proj >= low_proj else -1.0
    forward_ok = abs(high_proj - low_proj) > 1e-8

    tap_reports: list[dict[str, Any]] = []
    prev_cos: float | None = None
    flip_events: list[dict[str, Any]] = []
    for tap in taps:
        t = tap["tensor"]
        # Per-residue row for target_b when 2D node features.
        if t.ndim == 2 and t.shape[0] >= n:
            act_b = t[target_b]
            g = t.grad
            grad_b = g[target_b] if g is not None else None
        else:
            act_b = t
            grad_b = t.grad
        pack = polarity_pack(act_b, grad_b)
        pack.update(
            {
                "name": tap["name"],
                "op": tap["op"],
                "kind": tap["kind"],
                "shape": list(t.shape),
            }
        )
        # Node-feature taps: also report cos of full residue pool vs ∂s/∂tap
        # restricted to target column influence isn't needed — keep B-local.
        if prev_cos is not None and pack["cos_act_grad"] is not None:
            # Flip = cosine changes sign across a tap boundary.
            if prev_cos * pack["cos_act_grad"] < 0 and abs(prev_cos) > 0.05 and abs(
                pack["cos_act_grad"]
            ) > 0.05:
                flip_events.append(
                    {
                        "after_tap": tap["name"],
                        "prev_cos": prev_cos,
                        "cos": pack["cos_act_grad"],
                        "kind": tap["kind"],
                    }
                )
        if pack["cos_act_grad"] is not None and np.isfinite(pack["cos_act_grad"]):
            prev_cos = pack["cos_act_grad"]
        tap_reports.append(pack)

    # Pool-level: Spearman of |PC1| vs betweenness (geometry still alive?)
    rho_pc1 = spearman_corr(pc1_sign * proj, btw)

    # Suspect ranking from this run: DIV/NORM taps with largest |Δcos| from previous.
    deltas: list[dict[str, Any]] = []
    for i in range(1, len(tap_reports)):
        a = tap_reports[i - 1].get("cos_act_grad")
        b = tap_reports[i].get("cos_act_grad")
        if a is None or b is None or not (np.isfinite(a) and np.isfinite(b)):
            continue
        deltas.append(
            {
                "from": tap_reports[i - 1]["name"],
                "to": tap_reports[i]["name"],
                "kind": tap_reports[i]["kind"],
                "delta_cos": float(b - a),
                "abs_delta_cos": float(abs(b - a)),
                "sign_flip": bool(a * b < 0 and abs(a) > 0.05 and abs(b) > 0.05),
            }
        )
    deltas_sorted = sorted(deltas, key=lambda d: d["abs_delta_cos"], reverse=True)

    # Per-residue act·grad cos distributions (same ∂s_B; cheap — one backward).
    # Distinguishes attenuation (tight near 0) vs incoherence (wide ± scatter).
    dist_names = (
        "00_raw_x",
        "01_after_input_zscore",
        "02_after_node_emb",
        "99_encoder_h",
    )
    tap_by_name = {t["name"]: t for t in taps}
    pool_distributions: dict[str, Any] = {}
    for name in dist_names:
        tap = tap_by_name.get(name)
        if tap is None:
            continue
        t = tap["tensor"]
        cos_i = per_residue_cos_act_grad(t, t.grad, n=n)
        summary = cos_distribution_summary(cos_i)
        # Also grad L2 magnitude spread (wash-out of signal strength).
        g = t.grad
        if g is not None and g.ndim == 2 and g.shape[0] >= n:
            g_l2 = torch.linalg.vector_norm(g[:n].detach(), dim=-1).cpu().numpy()
            summary["grad_l2_mean"] = float(np.mean(g_l2))
            summary["grad_l2_median"] = float(np.median(g_l2))
            summary["grad_l2_q90"] = float(np.quantile(g_l2, 0.90))
            summary["grad_l2_max"] = float(np.max(g_l2))
        pool_distributions[name] = summary

    return {
        "pdb_id": str(prot.get("pdb_id", "")).upper(),
        "n_residues": n,
        "target_b": target_b,
        "target_betweenness": float(btw[target_b]),
        "target_pc1_proj": float(proj[target_b]),
        "score_mode": "pc1_sq",
        "forward_pc1_separates_high_low_btw": bool(forward_ok),
        "forward_high_btw_mean_pc1": high_proj * pc1_sign,
        "forward_low_btw_mean_pc1": low_proj * pc1_sign,
        "spearman_pc1_vs_betweenness": rho_pc1,
        "flip_events": flip_events,
        "largest_cos_jumps": deltas_sorted[:8],
        "taps": tap_reports,
        "pool_act_grad_cos_distributions": pool_distributions,
        "hypothesis_note": (
            "B-local cos at 00_raw_x ≈+0.05 is wash-out, not a sign flip. "
            "Inspect pool_act_grad_cos_distributions['00_raw_x'].shape_read: "
            "attenuation_tight_near_zero vs incoherence_wide_scatter before "
            "chasing aggregation code. Trunk DIV (LayerNorm) already demoted."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "checkpoints/v66/runs/chem_mvp_znorm_stage_a12_cold_v1/v66_best.pt"
        ),
    )
    parser.add_argument(
        "--control-checkpoint",
        type=Path,
        default=Path(
            "checkpoints/v66/runs/chem_mvp_stage_a12_cold_v1/v66_best.pt"
        ),
        help="z-norm-off chem-MVP for matched comparison (optional skip with '')",
    )
    parser.add_argument(
        "--corpus-cache",
        type=Path,
        default=Path("pdb_cache/corpus_cache/graphs_38a6993d7a439aa4.pt"),
    )
    parser.add_argument("--pdb-id", default="4OBE")
    parser.add_argument("--target-b", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUT)
    args = parser.parse_args(argv)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    proteins = load_proteins_from_cache(args.corpus_cache, [args.pdb_id])
    prot = proteins[args.pdb_id.upper()]

    report: dict[str, Any] = {
        "schema_version": 1,
        "probe": "jacobian_znorm_layer_grad_sign",
        "defect_ssot": "docs/specs/learned-flow-influence/JACOBIAN_ZNORM_DEFECT.md",
        "corpus_cache": str(args.corpus_cache),
        "arms": {},
    }

    for label, ckpt in (
        ("znorm_on", args.checkpoint),
        ("znorm_off_control", args.control_checkpoint),
    ):
        if ckpt is None or str(ckpt) in ("", "none", "None"):
            continue
        if not Path(ckpt).is_file():
            report["arms"][label] = {"error": f"missing checkpoint {ckpt}"}
            continue
        print(f"running {label} on {ckpt} ...", flush=True)
        model = load_model_from_checkpoint(Path(ckpt), device)
        arm = run_layer_grad_sign(
            model=model,
            prot=prot,
            device=device,
            target_b=args.target_b,
        )
        arm["checkpoint"] = str(ckpt)
        arm["input_feature_zscore"] = bool(
            getattr(model, "input_feature_zscore", False)
        )
        report["arms"][label] = arm

    # Compact verdict: trunk flips + input pool shape (attenuation vs incoherence).
    z = report["arms"].get("znorm_on") or {}
    c = report["arms"].get("znorm_off_control") or {}
    flips = z.get("flip_events") or []
    div_flips = [f for f in flips if f.get("kind") == "DIV"]
    z_raw = (z.get("pool_act_grad_cos_distributions") or {}).get("00_raw_x") or {}
    c_raw = (c.get("pool_act_grad_cos_distributions") or {}).get("00_raw_x") or {}
    report["verdict"] = {
        "znorm_forward_pc1_alive": z.get("forward_pc1_separates_high_low_btw"),
        "znorm_spearman_pc1_btw": z.get("spearman_pc1_vs_betweenness"),
        "n_sign_flips_total": len(flips),
        "n_sign_flips_at_DIV": len(div_flips),
        "first_div_flip": div_flips[0] if div_flips else None,
        "top_cos_jump": (z.get("largest_cos_jumps") or [None])[0],
        "b_local_raw_x_cos": next(
            (
                t.get("cos_act_grad")
                for t in (z.get("taps") or [])
                if t.get("name") == "00_raw_x"
            ),
            None,
        ),
        "pool_raw_x_znorm": {
            "mean": z_raw.get("mean"),
            "std": z_raw.get("std"),
            "q10": z_raw.get("q10"),
            "q90": z_raw.get("q90"),
            "frac_abs_lt_0_1": z_raw.get("frac_abs_lt_0_1"),
            "frac_abs_gt_0_5": z_raw.get("frac_abs_gt_0_5"),
            "shape_read": z_raw.get("shape_read"),
        },
        "pool_raw_x_control": {
            "mean": c_raw.get("mean"),
            "std": c_raw.get("std"),
            "q10": c_raw.get("q10"),
            "q90": c_raw.get("q90"),
            "frac_abs_lt_0_1": c_raw.get("frac_abs_lt_0_1"),
            "frac_abs_gt_0_5": c_raw.get("frac_abs_gt_0_5"),
            "shape_read": c_raw.get("shape_read"),
        },
        "read": (
            "B-local raw_x cos ≈+0.05 is wash-out, not a flip. Use "
            "pool_raw_x_*.shape_read before aggregation surgery: "
            "attenuation_tight_near_zero vs incoherence_wide_scatter."
        ),
    }

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.pdb_id.upper()}_layer_grad_sign.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["verdict"], indent=2))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
