"""Unit tests for AlleleSens / Epistasis investigation metrics (not NIG)."""

from __future__ import annotations

import torch
from geoopt.manifolds.stereographic import math as pmath

from science.tokyo_eye.investigation_metrics import allele_sens, epistasis_coupling


def _point_on_ball(tangents: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    k = torch.tensor(-abs(c), dtype=tangents.dtype)
    return pmath.expmap0(tangents, k=k)


def test_allele_sens_zero_when_identical() -> None:
    c = 1.0
    tang = 0.05 * torch.randn(8, 4)
    x = _point_on_ball(tang, c=c)
    idx = torch.tensor([0, 2, 5])
    val = allele_sens(x, x.clone(), idx, curvature=c)
    assert float(val) < 1e-5


def test_allele_sens_positive_when_displaced() -> None:
    c = 1.0
    tang = 0.05 * torch.randn(6, 3)
    wt = _point_on_ball(tang, c=c)
    mut_tang = tang.clone()
    mut_tang[1] = mut_tang[1] + 0.2
    mut = _point_on_ball(mut_tang, c=c)
    idx = torch.tensor([1, 2])
    val = allele_sens(wt, mut, idx, curvature=c)
    assert float(val) > 1e-3


def test_epistasis_zero_when_additive() -> None:
    c = 1.0
    # Small tangents so chart additivity ≈ ball composition for this test.
    wt_t = torch.zeros(5, 3)
    a_t = wt_t.clone()
    a_t[0, 0] = 0.05
    b_t = wt_t.clone()
    b_t[0, 1] = 0.05
    ab_t = wt_t.clone()
    ab_t[0, 0] = 0.05
    ab_t[0, 1] = 0.05
    wt = _point_on_ball(wt_t, c=c)
    xa = _point_on_ball(a_t, c=c)
    xb = _point_on_ball(b_t, c=c)
    xab = _point_on_ball(ab_t, c=c)
    # Exact additivity in tangent-of-origin chart for deltas from wt=0.
    val = epistasis_coupling(wt, xa, xb, xab, curvature=c, focus_idx=torch.tensor([0]))
    assert float(val) < 1e-4


def test_epistasis_positive_when_nonadditive() -> None:
    c = 1.0
    wt_t = torch.zeros(4, 3)
    a_t = wt_t.clone()
    a_t[0, 0] = 0.08
    b_t = wt_t.clone()
    b_t[0, 1] = 0.08
    ab_t = wt_t.clone()
    ab_t[0, 0] = 0.20  # not a+b in tangent
    ab_t[0, 1] = 0.20
    wt = _point_on_ball(wt_t, c=c)
    val = epistasis_coupling(
        wt,
        _point_on_ball(a_t, c=c),
        _point_on_ball(b_t, c=c),
        _point_on_ball(ab_t, c=c),
        curvature=c,
        focus_idx=torch.tensor([0]),
    )
    assert float(val) > 1e-3
