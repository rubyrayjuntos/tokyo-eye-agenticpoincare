"""T1a forward-only: z-norm inputs → pre_mp / encoder_h rank (no training).

Confirms the predicted ceiling (~z-scored input ER ≈ 1.7) and whether MP
preserves that gain into ``encoder_h``.

Usage:
  python -m experiments.diagnostics.t1a_znorm_forward_probe \\
    --checkpoint checkpoints/v66/runs/fix1_s4_stage_a12_cold_v1/epochs/epoch_020.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.diagnostics.trunk_hidden_occupancy import _collect, _svd_pack
from experiments.training.v6.corpus import load_training_proteins
from experiments.training.v6.train_loop import prepare_training_batch
from science.dtie.common.residue_features import TAU
from science.training.gnn_lineage import load_model_from_checkpoint


def _tau_definition_audit(X: np.ndarray) -> dict[str, Any]:
    """X columns: rho, tau_flag, ss, sasa."""
    rho = X[:, 0]
    tau = X[:, 1]
    # Official rule: tau_flag = 1 if rho < TAU else 0
    recon = (rho < TAU).astype(np.float64)
    exact = float(np.mean(np.isclose(tau, recon, atol=1e-5)))
    # Continuous alternatives carrying threshold distance.
    dist = np.abs(rho - TAU)
    soft = 1.0 / (1.0 + np.exp((rho - TAU) / 2.0))  # smooth sub-thresholdness
    z = (X - X.mean(0)) / (X.std(0) + 1e-8)
    packs = {
        "raw4_unnormalized": _svd_pack(X, name="raw4"),
        "raw4_zscore": _svd_pack(z, name="raw4_z"),
        "drop_tau_zscore": _svd_pack(z[:, [0, 2, 3]], name="no_tau_z"),
        "rho_ss_sasa_plus_abs_dist_z": _svd_pack(
            np.column_stack(
                [
                    (rho - rho.mean()) / (rho.std() + 1e-8),
                    (X[:, 2] - X[:, 2].mean()) / (X[:, 2].std() + 1e-8),
                    (X[:, 3] - X[:, 3].mean()) / (X[:, 3].std() + 1e-8),
                    (dist - dist.mean()) / (dist.std() + 1e-8),
                ]
            ),
            name="abs_dist_z",
        ),
        "rho_ss_sasa_plus_soft_tau_z": _svd_pack(
            np.column_stack(
                [
                    (rho - rho.mean()) / (rho.std() + 1e-8),
                    (X[:, 2] - X[:, 2].mean()) / (X[:, 2].std() + 1e-8),
                    (X[:, 3] - X[:, 3].mean()) / (X[:, 3].std() + 1e-8),
                    (soft - soft.mean()) / (soft.std() + 1e-8),
                ]
            ),
            name="soft_tau_z",
        ),
    }
    return {
        "definition": "tau_flag = 1.0 if rho < TAU else 0.0 (TAU=13.0) — deterministic of ρ",
        "TAU": float(TAU),
        "frac_exact_match_recon": exact,
        "corr_rho_tau_flag": float(np.corrcoef(rho, tau)[0, 1]),
        "feature_boards": {
            k: {
                "effective_rank": v.get("effective_rank"),
                "top1": (v.get("explained_var_topk") or {}).get("k1"),
                "sigma2_sigma1": v.get("sigma2_sigma1"),
            }
            for k, v in packs.items()
        },
        "expected_pre_mp_ceiling_after_znorm": float(
            packs["raw4_zscore"].get("effective_rank", float("nan"))
        ),
        "note": (
            "Linear is rank-non-increasing: z-norm alone should lift pre_mp toward "
            "~raw4_zscore ER (≈1.7), not toward 4. Dropping binary τ or replacing it "
            "with |ρ−TAU| / soft-τ is a separate anti-redundancy lever."
        ),
    }


def _forward_ranks(
    model: torch.nn.Module,
    proteins: list[dict[str, Any]],
    device: str,
    *,
    normalize: bool,
    drop_tau: bool = False,
    tau_mode: str = "flag",  # flag | abs_dist | soft
) -> dict[str, Any]:
    parts: list[dict[str, np.ndarray]] = []
    # Corpus means/stds for consistent z-norm across proteins.
    xs = []
    for prot in proteins:
        structural = bool(getattr(model, "structural_disc_frozen", False))
        data = prepare_training_batch(
            model, prot, device, structural_disc_frozen=structural
        )
        xs.append(data.x.detach().cpu().numpy()[:, :4])
    Xall = np.concatenate(xs, axis=0)
    mu = Xall.mean(axis=0)
    sd = Xall.std(axis=0) + 1e-8

    # Monkeypatch: wrap forward to rewrite x before node_emb via hook on node_emb input.
    def _emb_pre_hook(_mod: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        x = inputs[0]
        x4 = x[:, :4].clone()
        if normalize:
            x4 = (x4 - torch.as_tensor(mu, device=x.device, dtype=x.dtype)) / torch.as_tensor(
                sd, device=x.device, dtype=x.dtype
            )
        if drop_tau or tau_mode != "flag":
            rho = x[:, 0]
            if tau_mode == "abs_dist":
                alt = torch.abs(rho - float(TAU))
                alt_corpus = np.abs(Xall[:, 0] - TAU)
            elif tau_mode == "soft":
                alt = 1.0 / (1.0 + torch.exp((rho - float(TAU)) / 2.0))
                alt_corpus = 1.0 / (1.0 + np.exp((Xall[:, 0] - TAU) / 2.0))
            else:
                alt = None
                alt_corpus = None
            if drop_tau:
                x4 = x4.clone()
                x4[:, 1] = 0.0
            elif alt is not None and alt_corpus is not None:
                a_mu = float(alt_corpus.mean())
                a_sd = float(alt_corpus.std() + 1e-8)
                x4 = x4.clone()
                x4[:, 1] = (alt - a_mu) / a_sd
        if x.size(1) > 4:
            x = torch.cat([x4, x[:, 4:]], dim=1)
        else:
            x = x4
        return (x,)

    handle = model.node_emb.register_forward_pre_hook(_emb_pre_hook)
    try:
        for prot in proteins:
            bundle = _collect(model, prot, device)
            if bundle is not None:
                parts.append(bundle)
    finally:
        handle.remove()

    if not parts:
        raise RuntimeError("no proteins collected")

    def stack(key: str) -> np.ndarray:
        return np.concatenate([p[key] for p in parts if key in p], axis=0)

    layers = {}
    for key in ("pre_mp", "encoder_h", "topo_features", "topo_tangent", "disc_2d"):
        if any(key in p for p in parts):
            layers[key] = _svd_pack(stack(key), name=key)
    return {
        "normalize": normalize,
        "drop_tau": drop_tau,
        "tau_mode": tau_mode,
        "n_residues": int(layers["encoder_h"]["n"]),
        "layers": {
            k: {
                "effective_rank": v.get("effective_rank"),
                "top1": (v.get("explained_var_topk") or {}).get("k1"),
                "sigma2_sigma1": v.get("sigma2_sigma1"),
            }
            for k, v in layers.items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="T1a z-norm forward-only pre_mp/encoder_h probe")
    parser.add_argument("--checkpoint", type=Path, required=True)
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
        default=Path("checkpoints/v66/diagnostics/t1a_znorm_forward_probe"),
    )
    args = parser.parse_args(argv)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    proteins, _ = load_training_proteins(args.pdb_dir, args.corpus, max_residues=1200)
    X = np.concatenate([p["data"].x.detach().cpu().numpy()[:, :4] for p in proteins], 0)
    tau_audit = _tau_definition_audit(X)

    model = load_model_from_checkpoint(args.checkpoint, device)
    model.eval()
    raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    tc = raw.get("training_config") if isinstance(raw, dict) else {}
    if isinstance(tc, dict):
        model.structural_disc_frozen = bool(
            tc.get("structural_disc_frozen", False)
            or tc.get("slim_moe_structural_ssot", False)
        )
        model.hyperbolic_mp_graph = bool(tc.get("hyperbolic_mp_graph", False))

    baseline = _forward_ranks(model, proteins, device, normalize=False)
    znorm = _forward_ranks(model, proteins, device, normalize=True)
    znorm_drop_tau = _forward_ranks(
        model, proteins, device, normalize=True, drop_tau=True
    )
    znorm_abs = _forward_ranks(
        model, proteins, device, normalize=True, tau_mode="abs_dist"
    )

    pre_base = float(baseline["layers"]["pre_mp"]["effective_rank"])
    pre_z = float(znorm["layers"]["pre_mp"]["effective_rank"])
    enc_z = float(znorm["layers"]["encoder_h"]["effective_rank"])
    ceiling = float(tau_audit["expected_pre_mp_ceiling_after_znorm"])
    if abs(pre_z - ceiling) <= 0.35:
        pre_read = "PRE_MP_NEAR_PREDICTED_CEILING"
    elif pre_z > pre_base + 0.3:
        pre_read = "PRE_MP_IMPROVED_BUT_BELOW_CEILING"
    else:
        pre_read = "PRE_MP_LITTLE_CHANGE"
    if enc_z >= pre_z - 0.15:
        mp_read = "MP_PRESERVES_PRE_MP_RANK"
    elif enc_z < 1.25 and pre_z >= 1.4:
        mp_read = "MP_SQUEEZES_INDEPENDENTLY"
    else:
        mp_read = "MP_PARTIAL_LOSS"

    report = {
        "checkpoint": str(args.checkpoint),
        "tau_definition": tau_audit,
        "conditions": {
            "baseline_raw": baseline,
            "znorm": znorm,
            "znorm_drop_tau_channel": znorm_drop_tau,
            "znorm_replace_tau_abs_dist": znorm_abs,
        },
        "verdict": {
            "pre_mp_baseline": pre_base,
            "pre_mp_znorm": pre_z,
            "encoder_h_znorm": enc_z,
            "predicted_ceiling": ceiling,
            "pre_mp_read": pre_read,
            "mp_read": mp_read,
            "expected_narrative": (
                f"z-norm alone should move pre_mp from ~{pre_base:.2f} toward ~{ceiling:.2f}, "
                "not toward 4. If encoder_h stays ~1.0 while pre_mp ≈ ceiling, MP is an "
                "independent bottleneck."
            ),
        },
    }

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    path = out / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["verdict"], indent=2))
    print(
        f"τ exact-match to (ρ<13)={tau_audit['frac_exact_match_recon']:.3f} | "
        f"pre_mp {pre_base:.3f}→{pre_z:.3f} (ceil≈{ceiling:.3f}) | "
        f"encoder_h_znorm={enc_z:.3f} | {pre_read} / {mp_read}"
    )
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
