"""Investigation metrics on frozen Θ — AlleleSens and Epistasis (not NIG ale/epi).

See ``docs/specs/tokyo-eye-v7/investigation-allele-epistasis-metrics.md``.
"""

from __future__ import annotations

import torch
from geoopt.manifolds.stereographic import math as pmath


def _as_k(curvature: float | torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(curvature, torch.Tensor):
        k = curvature.to(device=device, dtype=dtype)
    else:
        k = torch.tensor(float(curvature), device=device, dtype=dtype)
    # geoopt stereographic uses k = -c for Poincaré ball (negative curvature).
    if float(k.reshape(-1)[0].item()) > 0:
        k = -k.abs()
    return k


def allele_sens(
    x_hyp_wt: torch.Tensor,
    x_hyp_mut: torch.Tensor,
    neighborhood_idx: torch.Tensor,
    *,
    curvature: float | torch.Tensor,
) -> torch.Tensor:
    """Mean Poincaré geodesic displacement WT↔mut over neighborhood indices.

    Args:
        x_hyp_wt / x_hyp_mut: ``[N, D]`` ball embeddings (aligned residues).
        neighborhood_idx: 1D long tensor of residue indices in ``N``.
        curvature: learned ``c > 0`` or geoopt ``k < 0``.

    Returns:
        Scalar mean distance over ``|neighborhood_idx|``.
    """
    if x_hyp_wt.shape != x_hyp_mut.shape:
        raise ValueError(f"shape mismatch wt={tuple(x_hyp_wt.shape)} mut={tuple(x_hyp_mut.shape)}")
    idx = neighborhood_idx.long().reshape(-1)
    if idx.numel() == 0:
        raise ValueError("neighborhood_idx is empty")
    wt = x_hyp_wt.index_select(0, idx)
    mut = x_hyp_mut.index_select(0, idx)
    k = _as_k(curvature, device=wt.device, dtype=wt.dtype)
    d = pmath.dist(wt, mut, k=k)
    return d.mean()


def epistasis_coupling(
    x_hyp_wt: torch.Tensor,
    x_hyp_a: torch.Tensor,
    x_hyp_b: torch.Tensor,
    x_hyp_ab: torch.Tensor,
    *,
    curvature: float | torch.Tensor,
    focus_idx: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Tangent-chart non-additivity: ‖Δx_AB − (Δx_A + Δx_B)‖ via logmap₀.

    Args:
        x_hyp_*: ``[N, D]`` aligned ball embeddings.
        focus_idx: optional residue subset; default all residues.
        reduction: ``mean`` (default) or ``sum`` over focus rows' L2 norms.

    Returns:
        Scalar residual after reduction.
    """
    for name, t in (
        ("wt", x_hyp_wt),
        ("a", x_hyp_a),
        ("b", x_hyp_b),
        ("ab", x_hyp_ab),
    ):
        if t.shape != x_hyp_wt.shape:
            raise ValueError(f"{name} shape {tuple(t.shape)} != wt {tuple(x_hyp_wt.shape)}")
    k = _as_k(curvature, device=x_hyp_wt.device, dtype=x_hyp_wt.dtype)

    def _delta(x: torch.Tensor) -> torch.Tensor:
        return pmath.logmap0(x, k=k) - pmath.logmap0(x_hyp_wt, k=k)

    dx_a = _delta(x_hyp_a)
    dx_b = _delta(x_hyp_b)
    dx_ab = _delta(x_hyp_ab)
    residual = dx_ab - (dx_a + dx_b)
    if focus_idx is not None:
        residual = residual.index_select(0, focus_idx.long().reshape(-1))
    row_norm = residual.norm(dim=-1)
    if reduction == "mean":
        return row_norm.mean()
    if reduction == "sum":
        return row_norm.sum()
    raise ValueError(f"unknown reduction={reduction!r}")


def ball_distance_pair(
    x_a: torch.Tensor,
    x_b: torch.Tensor,
    *,
    curvature: float | torch.Tensor,
) -> torch.Tensor:
    """Pairwise Poincaré distance for aligned ``[N, D]`` rows → ``[N]``."""
    if x_a.shape != x_b.shape:
        raise ValueError(f"shape mismatch {tuple(x_a.shape)} vs {tuple(x_b.shape)}")
    k = _as_k(curvature, device=x_a.device, dtype=x_a.dtype)
    return pmath.dist(x_a, x_b, k=k)


def singleton_ball_distance(
    x_hyp_wt: torch.Tensor,
    x_hyp_mut: torch.Tensor,
    residue_idx: int,
    *,
    curvature: float | torch.Tensor,
) -> torch.Tensor:
    """Scalar Poincaré ball distance at one graph row."""
    i = int(residue_idx)
    return ball_distance_pair(
        x_hyp_wt[i : i + 1],
        x_hyp_mut[i : i + 1],
        curvature=curvature,
    ).reshape(())
