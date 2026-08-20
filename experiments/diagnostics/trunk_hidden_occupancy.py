"""S2 trunk / pre-gate hidden-state occupancy (not the 2D disc).

After the Fix-1+S4 logit-spread check closed *gate geometry* as the IBU
cause (Euclidean surrogate flatter than hyp), the remaining fork is:

  - low-rank encoder / pre-gate trunk → representation collapse still the
    bottleneck (Fix-1 / S4 only dressed the 2D projection);
  - high-rank trunk → flat routing is a readout / commitment problem.

Usage:
  python -m experiments.diagnostics.trunk_hidden_occupancy \\
    --checkpoint checkpoints/v66/runs/fix1_s4_stage_a12_cold_v1/epochs/epoch_020.pt \\
    --corpus manifests/v6_corpus_stage_a_small_v1.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from geoopt.manifolds.stereographic import math as pmath

from experiments.diagnostics.embedding_occupancy_audit import _effective_rank
from experiments.training.v6.corpus import load_training_proteins
from experiments.training.v6.train_loop import prepare_training_batch
from science.training.gnn_lineage import load_model_from_checkpoint


def _match_x(model: torch.nn.Module, data: Any) -> Any:
    in_f = int(getattr(model.node_emb, "in_features", data.x.size(-1)))
    if data.x.size(-1) > in_f:
        data.x = data.x[:, :in_f].contiguous()
    return data


def _svd_pack(pts: np.ndarray, *, name: str) -> dict[str, Any]:
    """Centered SVD occupancy + explained-variance / participation metrics."""
    X = np.asarray(pts, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 2:
        return {"layer": name, "n": int(X.shape[0]) if X.ndim == 2 else 0, "error": "too_few"}
    Xc = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    s = np.asarray(s, dtype=np.float64)
    energy = float(np.sum(s * s))
    if energy < 1e-18 or s[0] < 1e-15:
        return {
            "layer": name,
            "n": int(X.shape[0]),
            "dim": int(X.shape[1]),
            "effective_rank": 0.0,
            "sigma2_sigma1": float("nan"),
            "error": "degenerate",
        }
    frac = (s * s) / energy
    cum = np.cumsum(frac)
    # participation ratio (PR): (sum λ)^2 / sum(λ^2) with λ=σ²
    lam = s * s
    pr = float((lam.sum() ** 2) / (np.sum(lam * lam) + 1e-18))
    ks = [1, 2, 4, 8, 16]
    return {
        "layer": name,
        "n": int(X.shape[0]),
        "dim": int(X.shape[1]),
        "effective_rank": _effective_rank(s),
        "participation_ratio": pr,
        "sigma2_sigma1": float(s[1] / s[0]) if s.size > 1 else 0.0,
        "sigma": [float(x) for x in s[:8]],
        "sigma_ratio": [float(x / (s[0] + 1e-12)) for x in s[:8]],
        "explained_var_topk": {f"k{k}": float(cum[min(k, len(cum)) - 1]) for k in ks},
        "row_std_mean": float(Xc.std(axis=0).mean()),
        "row_l2_std": float(np.linalg.norm(Xc, axis=1).std()),
    }


def _verdict(encoder: dict[str, Any], pre_gate: dict[str, Any]) -> dict[str, Any]:
    """Heuristic fork: low-rank trunk vs reasonably diverse."""
    er = float(encoder.get("effective_rank", float("nan")))
    s2 = float(encoder.get("sigma2_sigma1", float("nan")))
    top1 = float((encoder.get("explained_var_topk") or {}).get("k1", float("nan")))
    # Historic collapse ~1.2 eff_rank on some representation.
    if er == er and (er < 2.5 or (s2 == s2 and s2 < 0.20) or (top1 == top1 and top1 > 0.85)):
        trunk = "LOW_RANK_COLLAPSED"
    elif er == er and er >= 8.0 and (s2 == s2 and s2 >= 0.35) and (top1 == top1 and top1 <= 0.55):
        trunk = "REASONABLY_HIGH_RANK"
    else:
        trunk = "AMBIGUOUS_RANK"
    return {
        "encoder_h": trunk,
        "encoder_effective_rank": er,
        "encoder_sigma2_sigma1": s2,
        "encoder_top1_explained_var": top1,
        "pre_gate_effective_rank": float(pre_gate.get("effective_rank", float("nan"))),
        "pre_gate_sigma2_sigma1": float(pre_gate.get("sigma2_sigma1", float("nan"))),
        "read": {
            "LOW_RANK_COLLAPSED": (
                "Gate/commitment levers secondary — Fix-1/S4 disc work improved a "
                "2D projection while the HD trunk that feeds routing stays thin. "
                "Next: trunk diversity / encoder occupancy, not gate temperature."
            ),
            "REASONABLY_HIGH_RANK": (
                "Trunk is diverse enough to route on — flat Softmax is a readout/"
                "commitment problem (logit scale, T, aux). Gate-geometry remains CLOSED."
            ),
            "AMBIGUOUS_RANK": (
                "Borderline occupancy — inspect explained_var_topk and compare "
                "encoder_h vs topo_tangent vs disc_2d before choosing a lever."
            ),
        }[trunk],
    }


def _collect(
    model: torch.nn.Module,
    prot: dict[str, Any],
    device: str,
) -> dict[str, np.ndarray] | None:
    captured: dict[str, torch.Tensor] = {}
    handles: list[Any] = []

    def _radial_hook(_mod: torch.nn.Module, inputs: tuple[torch.Tensor, ...], _out: Any) -> None:
        # radial_head(x) — x is post-MP Euclidean encoder state.
        if inputs and torch.is_tensor(inputs[0]):
            captured["encoder_h"] = inputs[0].detach()

    def _angular_hook(_mod: torch.nn.Module, inputs: tuple[torch.Tensor, ...], _out: Any) -> None:
        # SSOT / structural path may skip radial_head; angular_head still sees post-MP x.
        if "encoder_h" not in captured and inputs and torch.is_tensor(inputs[0]):
            captured["encoder_h"] = inputs[0].detach()

    def _emb_hook(_mod: torch.nn.Module, _inputs: Any, out: torch.Tensor) -> None:
        captured["pre_mp"] = out.detach()

    def _norm_hook(i: int):
        def _hook(_mod: torch.nn.Module, _inputs: Any, out: torch.Tensor) -> None:
            # After LayerNorm inside each MP residual block (pre-SiLU/residual).
            captured[f"mp_norm_{i}"] = out.detach()

        return _hook

    handles.append(model.radial_head.register_forward_hook(_radial_hook))
    if hasattr(model, "angular_head"):
        handles.append(model.angular_head.register_forward_hook(_angular_hook))
    if hasattr(model, "node_emb"):
        handles.append(model.node_emb.register_forward_hook(_emb_hook))
    for i, norm in enumerate(getattr(model, "norms", []) or []):
        handles.append(norm.register_forward_hook(_norm_hook(i)))
    try:
        structural_frozen = bool(getattr(model, "structural_disc_frozen", False))
        data = prepare_training_batch(
            model, prot, device, structural_disc_frozen=structural_frozen
        )
        data = _match_x(model, data)
        with torch.no_grad():
            out = model(data)
    finally:
        for h in handles:
            h.remove()

    if "encoder_h" not in captured:
        # Last resort: final MP norm (pre residual) when heads were skipped.
        norms = sorted(
            (k for k in captured if k.startswith("mp_norm_")),
            key=lambda s: int(s.rsplit("_", 1)[-1]),
        )
        if norms:
            captured["encoder_h"] = captured[norms[-1]]
    if "encoder_h" not in captured:
        return None

    k = -model.curvature
    x_hyp = out["x_hyp"].detach()
    x_routed = out["x_routed_hyp"].detach()
    disc = out["hyp_projections_2d"].detach()
    stash = getattr(model.gate, "_last_pre_softmax", None) or {}

    x_hyp_t = pmath.logmap0(x_hyp, k=k).detach()
    x_routed_t = pmath.logmap0(x_routed, k=k).detach()

    bundle: dict[str, np.ndarray] = {
        "encoder_h": captured["encoder_h"].cpu().numpy(),
        "x_hyp_ambient": x_hyp.cpu().numpy(),
        "x_hyp_tangent": x_hyp_t.cpu().numpy(),
        "x_routed_hyp_ambient": x_routed.cpu().numpy(),
        "x_routed_hyp_tangent": x_routed_t.cpu().numpy(),
        "disc_2d": disc.cpu().numpy(),
    }
    if "pre_mp" in captured:
        bundle["pre_mp"] = captured["pre_mp"].cpu().numpy()
    for key, tensor in captured.items():
        if key.startswith("mp_norm_"):
            bundle[key] = tensor.cpu().numpy()
    if "topo_tangent" in stash:
        bundle["topo_tangent"] = stash["topo_tangent"].cpu().numpy()
    if "fused_hyp" in stash:
        fused = stash["fused_hyp"]
        bundle["fused_hyp_ambient"] = fused.cpu().numpy()
        bundle["fused_hyp_tangent"] = pmath.logmap0(fused, k=k).detach().cpu().numpy()
    topo_feat = getattr(model.gate, "_last_topo_features", None)
    if topo_feat is not None:
        bundle["topo_features"] = topo_feat.cpu().numpy()
    return bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pre-gate / encoder HD occupancy (S2 trunk)")
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
        default=Path("checkpoints/v66/diagnostics/trunk_hidden_occupancy"),
    )
    args = parser.parse_args(argv)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = load_model_from_checkpoint(args.checkpoint, device)
    model.eval()
    raw_meta = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    tc_meta = raw_meta.get("training_config") if isinstance(raw_meta, dict) else None
    if isinstance(tc_meta, dict):
        ssot = bool(
            tc_meta.get("structural_disc_frozen", False)
            or tc_meta.get("slim_moe_structural_ssot", False)
        )
        model.structural_disc_frozen = ssot
    proteins, _ = load_training_proteins(
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
            bundle = _collect(model, prot, device)
        except Exception as exc:  # noqa: BLE001
            per_protein.append({"pdb_id": pdb, "error": str(exc)})
            continue
        if bundle is None:
            per_protein.append({"pdb_id": pdb, "error": "missing encoder_h hook"})
            continue
        parts.append(bundle)
        enc = _svd_pack(bundle["encoder_h"], name="encoder_h")
        disc = _svd_pack(bundle["disc_2d"], name="disc_2d")
        per_protein.append(
            {
                "pdb_id": pdb,
                "n": enc["n"],
                "encoder_effective_rank": enc.get("effective_rank"),
                "encoder_sigma2_sigma1": enc.get("sigma2_sigma1"),
                "disc_effective_rank": disc.get("effective_rank"),
                "disc_sigma2_sigma1": disc.get("sigma2_sigma1"),
            }
        )

    if not parts:
        raise SystemExit("no proteins collected")

    def stack(key: str) -> np.ndarray:
        return np.concatenate([p[key] for p in parts if key in p], axis=0)

    layer_keys = sorted({k for p in parts for k in p.keys()})
    layers = {key: _svd_pack(stack(key), name=key) for key in layer_keys}

    encoder = layers["encoder_h"]
    pre_gate = layers.get("topo_tangent") or layers.get("fused_hyp_tangent") or layers["x_hyp_tangent"]
    verdict = _verdict(encoder, pre_gate)

    # Origin fork: pre-MP vs post-MP vs raw gate features.
    pre_mp = layers.get("pre_mp")
    topo_feat = layers.get("topo_features")
    origin: dict[str, Any] = {"encoder_path": "UNKNOWN", "gate_topo_path": "UNKNOWN"}
    if pre_mp is not None:
        er0 = float(pre_mp.get("effective_rank", float("nan")))
        er1 = float(encoder.get("effective_rank", float("nan")))
        if er0 == er0 and er0 < 2.5:
            origin["encoder_path"] = "COLLAPSE_PRE_MP_OR_EMBED"
        elif er0 == er0 and er0 >= 8.0 and er1 == er1 and er1 < 2.5:
            origin["encoder_path"] = "COLLAPSE_IN_MESSAGE_PASSING"
        elif er0 == er0 and er1 == er1 and er0 < 8.0 and er1 < 2.5:
            origin["encoder_path"] = "COLLAPSE_WORSENS_IN_MP"
        else:
            origin["encoder_path"] = "ENCODER_NOT_CLEARLY_COLLAPSED"
        origin["pre_mp_effective_rank"] = er0
        origin["encoder_h_effective_rank"] = er1
    if topo_feat is not None:
        ert = float(topo_feat.get("effective_rank", float("nan")))
        ertan = float((layers.get("topo_tangent") or {}).get("effective_rank", float("nan")))
        if ert == ert and ert < 2.5:
            origin["gate_topo_path"] = "RAW_TOPO_FEATURES_LOW_RANK"
        elif ert == ert and ert >= 4.0 and ertan == ertan and ertan < 2.5:
            origin["gate_topo_path"] = "TOPO_ENCODER_MLP_COLLAPSES"
        else:
            origin["gate_topo_path"] = "GATE_TOPO_AMBIGUOUS"
        origin["topo_features_effective_rank"] = ert
        origin["topo_tangent_effective_rank"] = ertan
    verdict["origin"] = origin

    # Explicit: gate geometry already closed by logit-spread sibling.
    gate_geometry = {
        "status": "CLOSED",
        "do_not_reopen": True,
        "reason": (
            "fix1_s4 logit-spread: no origin/rim hyp asymmetry; Euclidean tangent "
            "surrogate on the same fused trunk was flatter than hyp — not a "
            "hyperbolic-vs-Euclidean gate choice problem."
        ),
        "artifact": "checkpoints/v66/diagnostics/fix1_s4_gate_logit_spread_seed1/",
    }

    report = {
        "checkpoint": str(args.checkpoint),
        "corpus": str(args.corpus),
        "n_proteins": len(parts),
        "n_residues": int(encoder["n"]),
        "topology_only_gate": bool(getattr(model.gate, "topology_only", False)),
        "note": (
            "encoder_h = post-MP Euclidean hidden (radial_head input). "
            "topo_tangent / fused_hyp = gate fused path (topology_only feeler). "
            "disc_2d listed only as contrast — Fix-1 targets this projection, not trunk rank."
        ),
        "gate_geometry_question": gate_geometry,
        "corpus_pooled": layers,
        "verdict": verdict,
        "per_protein": per_protein,
        "criteria_label": (
            "S2 trunk / hidden-state occupancy in Fix-1 sibling table "
            "(official S1–S6 table S2=disc occupancy; S3=topology depth — "
            "this check is the open 'post-MoE / trunk' item)"
        ),
    }

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps(verdict, indent=2))
    print(
        f"encoder_h ER={encoder['effective_rank']:.3f} σ2/σ1={encoder['sigma2_sigma1']:.3f} "
        f"top1={encoder['explained_var_topk']['k1']:.3f} | "
        f"disc_2d ER={layers['disc_2d']['effective_rank']:.3f} "
        f"σ2/σ1={layers['disc_2d']['sigma2_sigma1']:.3f}"
    )
    if "pre_mp" in layers:
        p = layers["pre_mp"]
        print(
            f"pre_mp ER={p['effective_rank']:.3f} top1={p['explained_var_topk']['k1']:.3f} | "
            f"origin.encoder={origin['encoder_path']}"
        )
    if "topo_features" in layers:
        tf = layers["topo_features"]
        print(
            f"topo_features ER={tf['effective_rank']:.3f} top1={tf['explained_var_topk']['k1']:.3f} | "
            f"origin.gate_topo={origin['gate_topo_path']}"
        )
    if "topo_tangent" in layers:
        t = layers["topo_tangent"]
        print(
            f"topo_tangent ER={t['effective_rank']:.3f} σ2/σ1={t['sigma2_sigma1']:.3f} "
            f"top1={t['explained_var_topk']['k1']:.3f}"
        )
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
