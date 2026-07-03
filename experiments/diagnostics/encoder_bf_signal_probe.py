"""
encoder_bf_signal_probe.py
==========================
Pre-v3 gate: does uncertainty_head input carry B-factor signal orthogonal to
SASA and depth?

If max partial_corr(encoder_hidden_i, bf | depth, sasa) < 0.10 across the
cohort, no λ schedule can align epistemic with B-factor residuals — the MoE
tangent features do not encode the referent. Step 4 needs encoder changes.

If > 0.20, λ adjustment (Option B staged ramp) is viable — v2 was gradient
balance, not missing signal.

Usage:
  python -m experiments.diagnostics.encoder_bf_signal_probe --run
  make probe-encoder-bf-signal CHECKPOINT=checkpoints/v6/runs/lever_a_clean_slate_v1/v6_best_disc.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from experiments.diagnostics.epistemic_decoupling_baseline import (
    DEFAULT_CHECKPOINT,
    DEFAULT_PDB_DIR,
    DEFAULT_STRUCTURES,
    _bfactors_for_training_graph,
    _ols_residual,
    zscore,
)

PROBE_GO_THRESHOLD = 0.20
PROBE_NO_GO_THRESHOLD = 0.10


def partial_corr_multi(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """corr(x, y | z) with z as one or more covariates (columns)."""
    z = np.asarray(z, dtype=float)
    if z.ndim == 1:
        z = z[:, None]
    mask = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(z), axis=1)
    if mask.sum() < 5:
        return float("nan")
    rx = _ols_residual(x[mask], z[mask])
    ry = _ols_residual(y[mask], z[mask])
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def max_feature_partial_corr(
    features: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> tuple[float, int]:
    """Best single hidden dimension by |partial corr| with y given z."""
    best_r = float("nan")
    best_i = -1
    for i in range(features.shape[1]):
        r = partial_corr_multi(features[:, i], y, z)
        if not np.isfinite(r):
            continue
        if not np.isfinite(best_r) or abs(r) > abs(best_r):
            best_r = r
            best_i = i
    return best_r, best_i


@dataclass
class StructureProbe:
    structure_id: str
    n_residues: int
    hidden_dim: int
    max_dim_partial_corr_bf: float
    max_dim_index: int
    tangent_norm_partial_corr_bf: float
    mean_abs_partial_corr_bf: float
    r_epi_bf_partial_depth_only: float
    r_epi_bf_partial_depth_sasa: float
    partial_epi_sasa_given_depth: float
    corr_bf_resid_sasa_resid_given_depth: float


@torch.inference_mode()
def load_structure_probe_arrays(
    structure_id: str,
    *,
    checkpoint: Path,
    pdb_dir: Path,
    chain: str = "A",
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    from experiments.diagnostics.embedding_occupancy_audit import load_audit_model
    from experiments.training.v6._data import load_protein_graph
    from experiments.training.v6.train_loop import attach_v6_features

    pdb_id = structure_id.upper()
    prot = load_protein_graph(pdb_id, chain, pdb_dir)
    if prot is None:
        raise RuntimeError(f"Could not load protein graph for {pdb_id}:{chain}")

    data = attach_v6_features(prot["data"])
    model, version = load_audit_model(checkpoint, device)
    if version != "v6":
        raise RuntimeError(f"Expected v6 checkpoint, got {version}")

    captured: dict[str, torch.Tensor] = {}

    def _hook(_module, inputs):
        captured["x_for_unc"] = inputs[0].detach()

    handle = model.uncertainty_head.register_forward_pre_hook(_hook)
    try:
        out = model(data.to(device))
    finally:
        handle.remove()

    x_for_unc = captured["x_for_unc"].cpu().numpy()
    # x_for_unc = [routed_tangent (hidden), depth (1), cone_width (1)]
    depth_col = x_for_unc[:, -2]
    tangent = x_for_unc[:, :-2]

    epi = out["uncertainty"]["epistemic"].detach().cpu().numpy().reshape(-1)
    depth = out["cone_depth"].detach().cpu().numpy().reshape(-1)
    sasa = data.x[:, 3].detach().cpu().numpy()

    bfactor_raw, present_mask = _bfactors_for_training_graph(pdb_id, chain, pdb_dir)
    if bfactor_raw.shape[0] != tangent.shape[0]:
        raise RuntimeError(
            f"{pdb_id}:{chain} B-factor count {bfactor_raw.shape[0]} != nodes {tangent.shape[0]}"
        )

    bf_z = zscore(np.where(present_mask, bfactor_raw, np.nan))
    use = present_mask & np.isfinite(bf_z) & np.isfinite(sasa) & np.isfinite(depth)
    z_cov = np.column_stack([depth[use], sasa[use]])

    return {
        "tangent": tangent[use],
        "bf_z": bf_z[use],
        "depth": depth[use],
        "sasa": sasa[use],
        "epi": epi[use],
        "z_cov": z_cov,
        "hidden_dim": tangent.shape[1],
    }


def compute_structure_probe(structure_id: str, arrays: dict[str, np.ndarray]) -> StructureProbe:
    tangent = arrays["tangent"]
    bf_z = arrays["bf_z"]
    depth = arrays["depth"]
    sasa = arrays["sasa"]
    epi = arrays["epi"]
    z_cov = arrays["z_cov"]

    max_r, max_i = max_feature_partial_corr(tangent, bf_z, z_cov)
    tangent_norm = np.linalg.norm(tangent, axis=1)
    norm_r = partial_corr_multi(tangent_norm, bf_z, z_cov)

    abs_rs = []
    for i in range(tangent.shape[1]):
        r = partial_corr_multi(tangent[:, i], bf_z, z_cov)
        if np.isfinite(r):
            abs_rs.append(abs(r))
    mean_abs = float(np.mean(abs_rs)) if abs_rs else float("nan")

    bf_sasa_partial = partial_corr_multi(bf_z, sasa, depth.reshape(-1, 1))

    return StructureProbe(
        structure_id=structure_id,
        n_residues=int(tangent.shape[0]),
        hidden_dim=int(arrays["hidden_dim"]),
        max_dim_partial_corr_bf=max_r,
        max_dim_index=max_i,
        tangent_norm_partial_corr_bf=norm_r,
        mean_abs_partial_corr_bf=mean_abs,
        r_epi_bf_partial_depth_only=partial_corr_multi(epi, bf_z, depth.reshape(-1, 1)),
        r_epi_bf_partial_depth_sasa=partial_corr_multi(epi, bf_z, z_cov),
        partial_epi_sasa_given_depth=partial_corr_multi(epi, sasa, depth.reshape(-1, 1)),
        corr_bf_resid_sasa_resid_given_depth=bf_sasa_partial,
    )


def cohort_verdict(probes: list[StructureProbe]) -> dict[str, float | int | str]:
    max_dims = [p.max_dim_partial_corr_bf for p in probes if np.isfinite(p.max_dim_partial_corr_bf)]
    norm_rs = [
        p.tangent_norm_partial_corr_bf
        for p in probes
        if np.isfinite(p.tangent_norm_partial_corr_bf)
    ]
    mean_max = float(np.mean(max_dims)) if max_dims else float("nan")
    mean_norm = float(np.mean(norm_rs)) if norm_rs else float("nan")

    if not np.isfinite(mean_max):
        verdict = "INDETERMINATE"
        rationale = "No finite per-structure encoder partial correlations."
    elif mean_max < PROBE_NO_GO_THRESHOLD:
        verdict = "NO-GO"
        rationale = (
            f"mean max_dim_partial_corr={mean_max:.3f} < {PROBE_NO_GO_THRESHOLD}: "
            "encoder hidden lacks B-factor signal beyond depth+SASA — revise Step 4 "
            "(encoder must learn flexibility features) before any λ run."
        )
    elif mean_max >= PROBE_GO_THRESHOLD:
        verdict = "GO"
        rationale = (
            f"mean max_dim_partial_corr={mean_max:.3f} >= {PROBE_GO_THRESHOLD}: "
            "B-factor signal exists in uncertainty_head inputs — λ schedule (Option B) viable."
        )
    else:
        verdict = "BORDERLINE"
        rationale = (
            f"mean max_dim_partial_corr={mean_max:.3f} in "
            f"[{PROBE_NO_GO_THRESHOLD}, {PROBE_GO_THRESHOLD}): proceed with caution; "
            "Option B epoch-5 rising gate is decisive."
        )

    return {
        "n_structures": len(probes),
        "mean_max_dim_partial_corr_bf": mean_max,
        "mean_tangent_norm_partial_corr_bf": mean_norm,
        "PROBE_GO_THRESHOLD": PROBE_GO_THRESHOLD,
        "PROBE_NO_GO_THRESHOLD": PROBE_NO_GO_THRESHOLD,
        "VERDICT": verdict,
        "rationale": rationale,
    }


def run(
    structure_ids: list[str],
    *,
    checkpoint: Path,
    pdb_dir: Path,
    chain: str = "A",
    device: str = "cpu",
) -> dict[str, object]:
    probes: list[StructureProbe] = []
    for sid in structure_ids:
        arrays = load_structure_probe_arrays(
            sid,
            checkpoint=checkpoint,
            pdb_dir=pdb_dir,
            chain=chain,
            device=device,
        )
        probes.append(compute_structure_probe(sid, arrays))
    return {
        "checkpoint": str(checkpoint),
        "pdb_dir": str(pdb_dir),
        "chain": chain,
        "structures": structure_ids,
        "per_structure": [asdict(p) for p in probes],
        "cohort": cohort_verdict(probes),
        "question": (
            "partial_corr(encoder_hidden_i, zscore(bfactor) | depth, sasa) — "
            "does MoE tangent output carry B-factor signal the uncertainty head can use?"
        ),
    }


def _selftest() -> int:
    rng = np.random.default_rng(0)
    n, h = 500, 8
    depth = rng.normal(size=n)
    sasa = 0.5 * depth + rng.normal(size=n)
    bf_latent = rng.normal(size=n)
    bf_z = bf_latent + 0.3 * rng.normal(size=n)
    z_cov = np.column_stack([depth, sasa])

    # Hidden dim 0 encodes bf_latent beyond depth+sasa
    hidden = np.column_stack(
        [bf_latent + 0.1 * rng.normal(size=n)] + [rng.normal(size=n) for _ in range(h - 1)]
    )
    max_r, _ = max_feature_partial_corr(hidden, bf_z, z_cov)
    assert max_r > 0.5, f"expected strong signal, got {max_r}"

    # No signal in hidden
    hidden_null = rng.normal(size=(n, h))
    max_null, _ = max_feature_partial_corr(hidden_null, bf_z, z_cov)
    assert abs(max_null) < 0.15, f"expected weak signal, got {max_null}"

    print("self-tests PASSED: encoder_bf_signal_probe statistics")
    return 0


def _parse_structures(raw: str) -> list[str]:
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true", help="Run probe on default battery")
    ap.add_argument("--structures", default=",".join(DEFAULT_STRUCTURES))
    ap.add_argument("--chain", default="A")
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--pdb-dir", type=Path, default=DEFAULT_PDB_DIR)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("checkpoints/v6/diagnostics/encoder_bf_signal_probe.json"),
    )
    args = ap.parse_args()
    if args.selftest:
        sys.exit(_selftest())
    structure_ids = _parse_structures(args.structures)
    if not structure_ids:
        ap.error("No structures specified")
    report = run(
        structure_ids,
        checkpoint=args.checkpoint,
        pdb_dir=args.pdb_dir,
        chain=args.chain,
        device=args.device,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report["cohort"], indent=2))
    if report["cohort"]["VERDICT"] == "NO-GO":
        sys.exit(2)
    sys.exit(0)
