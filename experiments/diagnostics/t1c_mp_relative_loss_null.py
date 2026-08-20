"""T1c / MP null: relative rank loss of synthetic embeddings through MP.

Compares the z-norm natural path drop (~14% relative) to synthetic
pre-MP codes of controlled rank pushed through the *same* trained MP stack.

Usage:
  python -m experiments.diagnostics.t1c_mp_relative_loss_null \\
    --checkpoint checkpoints/v66/runs/fix1_s4_stage_a12_cold_v1/epochs/epoch_020.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.diagnostics.trunk_hidden_occupancy import _match_x, _svd_pack
from experiments.training.v6.corpus import load_training_proteins
from experiments.training.v6.train_loop import prepare_training_batch
from science.training.gnn_lineage import load_model_from_checkpoint


def _rel_drop(pre: float, post: float) -> float:
    if pre < 1e-8:
        return float("nan")
    return float((pre - post) / pre)


def _synth_embedding(n: int, hidden: int, rank: int, *, seed: int) -> np.ndarray:
    """Centered [N, H] cloud with target effective rank ~ ``rank`` (≤ min(N,H))."""
    rng = np.random.default_rng(seed)
    k = max(1, min(rank, n, hidden))
    # Orthonormal basis in R^H, Gaussian amplitudes per residue.
    basis = rng.standard_normal((hidden, k))
    basis, _ = np.linalg.qr(basis)
    coeffs = rng.standard_normal((n, k))
    # Equal energy across components → ER ≈ k
    X = coeffs @ basis.T
    X = X - X.mean(axis=0, keepdims=True)
    return X.astype(np.float64)


def _run_mp_from_pre(
    model: torch.nn.Module,
    prot: dict[str, Any],
    device: str,
    *,
    pre_mp_override: torch.Tensor | None,
    znorm_inputs: bool,
    mu: np.ndarray | None,
    sd: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (pre_mp, encoder_h) numpy arrays for one protein."""
    structural = bool(getattr(model, "structural_disc_frozen", False))
    data = prepare_training_batch(
        model, prot, device, structural_disc_frozen=structural
    )
    data = _match_x(model, data)

    captured: dict[str, torch.Tensor] = {}

    def _emb_pre(_m, inputs):
        x = inputs[0]
        if znorm_inputs and mu is not None and sd is not None:
            x4 = x[:, :4]
            x4 = (x4 - torch.as_tensor(mu, device=x.device, dtype=x.dtype)) / torch.as_tensor(
                sd, device=x.device, dtype=x.dtype
            )
            if x.size(1) > 4:
                x = torch.cat([x4, x[:, 4:]], dim=1)
            else:
                x = x4
        return (x,)

    def _emb_post(_m, _inp, out):
        if pre_mp_override is not None:
            captured["pre_mp"] = pre_mp_override
            return pre_mp_override
        captured["pre_mp"] = out.detach()
        return out

    def _radial_pre(_m, inputs):
        if inputs and torch.is_tensor(inputs[0]):
            captured["encoder_h"] = inputs[0].detach()
        return None

    def _angular_pre(_m, inputs):
        if "encoder_h" not in captured and inputs and torch.is_tensor(inputs[0]):
            captured["encoder_h"] = inputs[0].detach()
        return None

    h1 = model.node_emb.register_forward_pre_hook(_emb_pre)
    h2 = model.node_emb.register_forward_hook(_emb_post)
    h3 = model.radial_head.register_forward_pre_hook(_radial_pre)
    h4 = model.angular_head.register_forward_pre_hook(_angular_pre)
    try:
        with torch.no_grad():
            model(data)
    finally:
        h1.remove()
        h2.remove()
        h3.remove()
        h4.remove()

    if "pre_mp" not in captured or "encoder_h" not in captured:
        raise RuntimeError("failed to capture pre_mp/encoder_h")
    return (
        captured["pre_mp"].detach().cpu().numpy(),
        captured["encoder_h"].detach().cpu().numpy(),
    )


def _pooled(
    model: torch.nn.Module,
    proteins: list[dict[str, Any]],
    device: str,
    *,
    mode: str,
    rank: int | None = None,
    seed: int = 0,
    mu: np.ndarray | None = None,
    sd: np.ndarray | None = None,
) -> dict[str, Any]:
    """Aggregate **per-protein** relative drops (pooled SVD would inflate synth ER)."""
    pre_ers: list[float] = []
    enc_ers: list[float] = []
    rel_drops: list[float] = []
    hidden = int(model.hidden)
    n_res = 0
    for i, prot in enumerate(proteins):
        n = int(prot["data"].x.size(0))
        n_res += n
        override = None
        znorm = False
        if mode == "baseline":
            pass
        elif mode == "znorm":
            znorm = True
        elif mode.startswith("synth_rank_"):
            synth = _synth_embedding(n, hidden, int(rank or 4), seed=seed + 17 * i)
            # Scale to typical node_emb magnitude so norms/convs aren't weirdly saturated.
            synth = synth * 0.5
            override = torch.as_tensor(synth, device=device, dtype=torch.float32)
        else:
            raise ValueError(mode)
        pre, enc = _run_mp_from_pre(
            model,
            prot,
            device,
            pre_mp_override=override,
            znorm_inputs=znorm,
            mu=mu,
            sd=sd,
        )
        pre_s = _svd_pack(pre, name="pre_mp")
        enc_s = _svd_pack(enc, name="encoder_h")
        pre_er = float(pre_s["effective_rank"])
        enc_er = float(enc_s["effective_rank"])
        pre_ers.append(pre_er)
        enc_ers.append(enc_er)
        rel_drops.append(_rel_drop(pre_er, enc_er))

    pre_mean = float(np.mean(pre_ers))
    enc_mean = float(np.mean(enc_ers))
    rel_mean = float(np.mean(rel_drops))
    return {
        "mode": mode,
        "n_residues": int(n_res),
        "n_proteins": len(proteins),
        "pre_mp": {
            "effective_rank_mean": pre_mean,
            "effective_rank_std": float(np.std(pre_ers)),
        },
        "encoder_h": {
            "effective_rank_mean": enc_mean,
            "effective_rank_std": float(np.std(enc_ers)),
        },
        "relative_rank_drop": rel_mean,
        "relative_rank_drop_std": float(np.std(rel_drops)),
        "absolute_rank_drop": float(pre_mean - enc_mean),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MP relative rank-loss null (T1c gate)")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("manifests/v6_corpus_stage_a_small_v1.json"),
    )
    parser.add_argument("--pdb-dir", type=Path, default=Path("/tmp/dtie_pdb_cache"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/v66/diagnostics/t1c_mp_relative_loss_null"),
    )
    args = parser.parse_args(argv)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    proteins, _ = load_training_proteins(args.pdb_dir, args.corpus, max_residues=1200)
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

    # Corpus input stats for z-norm condition.
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

    baseline = _pooled(model, proteins, device, mode="baseline", seed=args.seed)
    znorm = _pooled(
        model, proteins, device, mode="znorm", seed=args.seed, mu=mu, sd=sd
    )
    # Match z-norm's achieved pre_mp rank (~1.7) and a fuller rank-4 control.
    synth_matched = _pooled(
        model,
        proteins,
        device,
        mode="synth_rank_2",
        rank=2,
        seed=args.seed,
    )
    synth_r4 = _pooled(
        model, proteins, device, mode="synth_rank_4", rank=4, seed=args.seed
    )

    z_drop = float(znorm["relative_rank_drop"])
    # Null band: average relative drop of synth controls near matched / R4.
    null_drops = [
        float(synth_matched["relative_rank_drop"]),
        float(synth_r4["relative_rank_drop"]),
    ]
    null_mean = float(np.mean(null_drops))
    # If natural z-norm drop is within +10pp of null mean → healthy MP mix.
    # If ≥ +20pp worse than null → independent MP squeeze (T1c).
    excess = z_drop - null_mean
    if excess <= 0.10:
        mp_verdict = "MP_LOSS_CONSISTENT_WITH_NULL"
    elif excess >= 0.20:
        mp_verdict = "MP_INDEPENDENT_SQUEEZE_T1C"
    else:
        mp_verdict = "MP_LOSS_BORDERLINE"

    report = {
        "checkpoint": str(args.checkpoint),
        "conditions": {
            "baseline": baseline,
            "znorm": znorm,
            "synth_rank_approx_2": synth_matched,
            "synth_rank_4": synth_r4,
        },
        "verdict": {
            "znorm_relative_drop": z_drop,
            "null_relative_drop_mean": null_mean,
            "excess_over_null": excess,
            "mp_verdict": mp_verdict,
            "read": {
                "MP_LOSS_CONSISTENT_WITH_NULL": (
                    "MP relative rank loss on z-normed real inputs matches synthetic "
                    "controls — expected aggregation, not a separate MP bug."
                ),
                "MP_INDEPENDENT_SQUEEZE_T1C": (
                    "MP destroys more rank than synthetic controls — open T1c line."
                ),
                "MP_LOSS_BORDERLINE": (
                    "Mild excess over null — watch under z-norm retrain; no separate "
                    "T1c commitment yet."
                ),
            }[mp_verdict],
        },
    }

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    path = out / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["verdict"], indent=2))
    print(
        f"znorm drop={z_drop:.3f} null≈{null_mean:.3f} excess={excess:.3f} → {mp_verdict}"
    )
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
