"""T1 origin split: raw x vs node_emb vs topo_features; init vs trained rank.

Scopes what ``pre_mp`` actually is (post-Linear embed, not raw inputs) and
separates two T1 lines:

  T1a — Euclidean embed / encoder path (raw x → node_emb → encoder_h)
  T1b — topology-only gate hand-features (topo_features → topo_tangent)

Also distinguishes init/capacity vs training dynamics via fresh init + early
epoch snapshots.

Usage:
  python -m experiments.diagnostics.t1_trunk_origin_audit \\
    --checkpoint-late checkpoints/v66/runs/fix1_s4_stage_a12_cold_v1/epochs/epoch_020.pt \\
    --checkpoint-early checkpoints/v66/runs/fix1_s4_stage_a12_cold_v1/epochs/epoch_001.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.diagnostics.trunk_hidden_occupancy import (
    _collect,
    _match_x,
    _svd_pack,
    _verdict,
)
from experiments.training.v6.corpus import load_training_proteins
from experiments.training.v6.train_loop import prepare_training_batch
from science.training.gnn_lineage import load_model_from_checkpoint


FEATURE_NAMES_4 = ("rho", "tau_flag", "ss_type", "sasa")


def _raw_feature_audit(proteins: list[dict[str, Any]]) -> dict[str, Any]:
    xs = []
    for prot in proteins:
        x = prot["data"].x.detach().cpu().numpy()
        if x.ndim != 2:
            continue
        xs.append(x[:, : min(4, x.shape[1])])
    X = np.concatenate(xs, axis=0)
    names = list(FEATURE_NAMES_4[: X.shape[1]])
    pack = _svd_pack(X, name="raw_x")
    Xz = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)
    pack_z = _svd_pack(Xz, name="raw_x_zscore")
    # Per-channel stats + pairwise |corr|
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    corr = np.corrcoef(X.T)
    abs_corr = np.abs(corr)
    np.fill_diagonal(abs_corr, 0.0)
    Xc = X - X.mean(axis=0, keepdims=True)
    var = Xc.var(axis=0)
    var_share = {n: float(v / max(float(var.sum()), 1e-12)) for n, v in zip(names, var)}
    return {
        "n_residues": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "feature_names": names,
        "mean": {n: float(m) for n, m in zip(names, means)},
        "std": {n: float(s) for n, s in zip(names, stds)},
        "variance_share": var_share,
        "std_ratio_max_min": float(stds.max() / max(float(stds.min()), 1e-12)),
        "max_abs_corr_offdiag": float(abs_corr.max()) if abs_corr.size else float("nan"),
        "corr": {
            f"{names[i]}__{names[j]}": float(corr[i, j])
            for i in range(len(names))
            for j in range(i + 1, len(names))
        },
        "svd_raw_unnormalized": pack,
        "svd_zscore": pack_z,
        "svd": pack,  # backward-compat alias (= unnormalized)
        "read": (
            "Unnormalized SVD is dominated by channel scale (esp. SASA). "
            "Always also report svd_zscore. Linear node_emb does not normalize — "
            "it sees raw scale. Intrinsic rank after Linear(4→128) ≤ 4; "
            "ER≥8 is only possible post-MP nonlinearity/neighborhood mixing."
        ),
    }


def _apply_ssot_flag(model: torch.nn.Module, checkpoint: Path) -> None:
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    tc = raw.get("training_config") if isinstance(raw, dict) else None
    if isinstance(tc, dict):
        model.structural_disc_frozen = bool(
            tc.get("structural_disc_frozen", False)
            or tc.get("slim_moe_structural_ssot", False)
        )


def _fresh_init_model(template_ckpt: Path, device: str) -> torch.nn.Module:
    """Same architecture as checkpoint, freshly initialized weights (true ep0)."""
    from science.training.gnn_lineage import (
        _import_model_module,
        get_lineage,
    )

    raw = torch.load(template_ckpt, map_location="cpu", weights_only=False)
    state = raw.get("model_state_dict", raw)
    arch = raw.get("architecture") if isinstance(raw, dict) else None
    tc = raw.get("training_config") if isinstance(raw, dict) else None
    lineage = "v6.6"
    if isinstance(arch, dict) and str(arch.get("version", "")).startswith("v6"):
        ver = str(arch.get("version"))
        lineage = "v6.6" if ver in {"v6.6", "v66"} else ("v6.5" if ver in {"v6.5", "v65"} else "v6")
    elif isinstance(tc, dict) and tc.get("gnn_lineage"):
        lineage = str(tc["gnn_lineage"])
    spec = get_lineage(lineage)
    module = _import_model_module(spec)
    infer = getattr(module, "infer_v6_model_kwargs", None) or getattr(
        module, f"infer_{spec.checkpoint_prefix}_model_kwargs", None
    )
    kwargs = infer(state, arch, tc)
    model = getattr(module, spec.model_class_name)(**kwargs)
    # Re-init carefully: brand-new module already randomly inited.
    model.to(device)
    model.eval()
    if isinstance(tc, dict):
        model.structural_disc_frozen = bool(
            tc.get("structural_disc_frozen", False)
            or tc.get("slim_moe_structural_ssot", False)
        )
        model.hyperbolic_mp_graph = bool(tc.get("hyperbolic_mp_graph", False))
    return model


def _pooled_layers(
    model: torch.nn.Module,
    proteins: list[dict[str, Any]],
    device: str,
) -> dict[str, Any]:
    parts: list[dict[str, np.ndarray]] = []
    raw_rows: list[np.ndarray] = []
    for prot in proteins:
        bundle = _collect(model, prot, device)
        if bundle is None:
            continue
        parts.append(bundle)
        # Raw input as fed to the model (matched width).
        structural_frozen = bool(getattr(model, "structural_disc_frozen", False))
        data = prepare_training_batch(
            model, prot, device, structural_disc_frozen=structural_frozen
        )
        data = _match_x(model, data)
        raw_rows.append(data.x.detach().cpu().numpy())
    if not parts:
        raise RuntimeError("no proteins collected under this model")

    def stack(key: str) -> np.ndarray:
        return np.concatenate([p[key] for p in parts if key in p], axis=0)

    keys = sorted({k for p in parts for k in p})
    layers = {k: _svd_pack(stack(k), name=k) for k in keys}
    X = np.concatenate(raw_rows, axis=0)
    layers["raw_x_model_input"] = _svd_pack(X, name="raw_x_model_input")
    # Linear embed weight rank (rows = hidden out).
    w = model.node_emb.weight.detach().cpu().numpy()
    layers["node_emb_weight"] = _svd_pack(w, name="node_emb_weight")
    return {
        "n_proteins": len(parts),
        "n_residues": int(layers["encoder_h"]["n"]),
        "layers": layers,
        "verdict": _verdict(
            layers["encoder_h"],
            layers.get("topo_tangent") or layers.get("fused_hyp_tangent") or layers["x_hyp_tangent"],
        ),
    }


def _snapshot_row(bundle: dict[str, Any]) -> dict[str, Any]:
    L = bundle["layers"]
    out = {}
    for k in (
        "raw_x_model_input",
        "pre_mp",
        "encoder_h",
        "topo_features",
        "topo_tangent",
        "disc_2d",
        "node_emb_weight",
    ):
        if k not in L:
            continue
        out[k] = {
            "effective_rank": L[k].get("effective_rank"),
            "sigma2_sigma1": L[k].get("sigma2_sigma1"),
            "top1": (L[k].get("explained_var_topk") or {}).get("k1"),
            "dim": L[k].get("dim"),
            "n": L[k].get("n"),
        }
    out["origin"] = (bundle.get("verdict") or {}).get("origin")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="T1a/T1b origin + init-vs-dynamics audit")
    parser.add_argument("--checkpoint-late", type=Path, required=True)
    parser.add_argument("--checkpoint-early", type=Path, default=None)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("manifests/v6_corpus_stage_a_small_v1.json"),
    )
    parser.add_argument("--pdb-dir", type=Path, default=Path("/tmp/dtie_pdb_cache"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/v66/diagnostics/t1_trunk_origin"),
    )
    parser.add_argument("--seed-init", type=int, default=0)
    args = parser.parse_args(argv)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    proteins, _ = load_training_proteins(
        args.pdb_dir, args.corpus, max_residues=1200
    )
    raw_audit = _raw_feature_audit(proteins)

    torch.manual_seed(int(args.seed_init))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed_init))

    snapshots: dict[str, Any] = {}
    # True ep0: fresh architecture init.
    init_model = _fresh_init_model(args.checkpoint_late, device)
    snapshots["init_ep0"] = _snapshot_row(_pooled_layers(init_model, proteins, device))
    snapshots["init_ep0"]["note"] = (
        "Fresh GOSPConeMapper weights (seeded); same kwargs as late ckpt — not a saved epoch_000"
    )

    if args.checkpoint_early is not None and args.checkpoint_early.is_file():
        early = load_model_from_checkpoint(args.checkpoint_early, device)
        early.eval()
        _apply_ssot_flag(early, args.checkpoint_early)
        # Match hyp-mp flag from late if early lacks it.
        late_raw = torch.load(args.checkpoint_late, map_location="cpu", weights_only=False)
        ltc = late_raw.get("training_config") if isinstance(late_raw, dict) else {}
        if isinstance(ltc, dict):
            early.hyperbolic_mp_graph = bool(
                getattr(early, "hyperbolic_mp_graph", False)
                or ltc.get("hyperbolic_mp_graph", False)
            )
            # Prefer early tc; fallback late geom flags already in early for geom runs.
        snapshots["epoch_early"] = _snapshot_row(_pooled_layers(early, proteins, device))
        snapshots["epoch_early"]["checkpoint"] = str(args.checkpoint_early)

    late = load_model_from_checkpoint(args.checkpoint_late, device)
    late.eval()
    _apply_ssot_flag(late, args.checkpoint_late)
    late_raw = torch.load(args.checkpoint_late, map_location="cpu", weights_only=False)
    ltc = late_raw.get("training_config") if isinstance(late_raw, dict) else {}
    if isinstance(ltc, dict):
        late.hyperbolic_mp_graph = bool(ltc.get("hyperbolic_mp_graph", False))
    snapshots["epoch_late"] = _snapshot_row(_pooled_layers(late, proteins, device))
    snapshots["epoch_late"]["checkpoint"] = str(args.checkpoint_late)

    # Init vs dynamics fork for T1a.
    er0 = float((snapshots["init_ep0"].get("pre_mp") or {}).get("effective_rank") or float("nan"))
    er_late = float(
        (snapshots["epoch_late"].get("pre_mp") or {}).get("effective_rank") or float("nan")
    )
    if er0 == er0 and er0 < 2.5:
        dynamics = "ALREADY_COLLAPSED_AT_INIT"
    elif er0 == er0 and er0 >= 8.0 and er_late == er_late and er_late < 2.5:
        dynamics = "COLLAPSES_DURING_TRAINING"
    elif er0 == er0 and er_late == er_late and er0 > er_late + 1.0:
        dynamics = "RANK_DROPS_WITH_TRAINING"
    else:
        dynamics = "AMBIGUOUS_DYNAMICS"
    raw_er = float((raw_audit.get("svd") or {}).get("effective_rank") or float("nan"))
    raw_z_er = float(
        (raw_audit.get("svd_zscore") or {}).get("effective_rank") or float("nan")
    )

    report = {
        "scope": {
            "pre_mp_definition": (
                "Output of nn.Linear(node_dim→hidden) a.k.a. node_emb — NOT raw input features"
            ),
            "T1a": "raw_x → node_emb(pre_mp) → encoder_h",
            "T1b": "topo_features → topo_tangent (topology-only gate board)",
            "rank_ceiling_note": (
                "Linear(4→128) cannot produce pre_mp ER>4; T1a 'healthy' for pre_mp "
                "means using most of the ≤4 input dims (after scaling), while encoder_h "
                "ER>4 would require MP nonlinear neighborhood mixing."
            ),
        },
        "raw_input_features_no_model": raw_audit,
        "snapshots": snapshots,
        "forks": {
            "T1a_feature_vs_embed": (
                "RAW_SCALE_DOMINATED"
                if raw_er == raw_er
                and raw_er < 2.5
                and raw_z_er == raw_z_er
                and raw_z_er >= 1.5
                else (
                    "RAW_FEATURES_INTRINSICALLY_LOW_RANK"
                    if raw_z_er == raw_z_er and raw_z_er < 2.5
                    else (
                        "EMBED_COLLAPSES_DIVERSE_INPUT"
                        if raw_z_er == raw_z_er and raw_z_er >= 2.5 and er_late < 2.5
                        else "AMBIGUOUS"
                    )
                )
            ),
            "T1a_init_vs_dynamics": dynamics,
            "pre_mp_er_init": er0,
            "pre_mp_er_late": er_late,
            "raw_x_er_unnormalized": raw_er,
            "raw_x_er_zscore": raw_z_er,
        },
    }

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    path = out / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps(report["forks"], indent=2))
    print(
        f"raw_x ER={raw_er:.3f} | pre_mp init={er0:.3f} late={er_late:.3f} | "
        f"T1a feature/embed={report['forks']['T1a_feature_vs_embed']} | "
        f"dynamics={dynamics}"
    )
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
