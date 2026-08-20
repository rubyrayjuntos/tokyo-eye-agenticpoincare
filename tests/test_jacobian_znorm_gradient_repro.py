"""Minimal repro: affine z-score does not flip gradient sign by itself.

Standing defect (docs/specs/learned-flow-influence/JACOBIAN_ZNORM_DEFECT.md):
on *trained* z-norm GNN trunks, Jacobian flow-centrality anti-correlates with
classical GT while direct PC1 correlates. This test only locks the negative
control — standardization alone is not a sign bug — so future fixes are not
misdirected at autograd through (x−μ)/σ.
"""

from __future__ import annotations

import torch


def test_affine_zscore_preserves_grad_sign_for_linear_score() -> None:
    torch.manual_seed(0)
    n, d = 8, 3
    x = torch.randn(n, d, dtype=torch.float64) * torch.tensor(
        [7.0, 0.5, 0.5], dtype=torch.float64
    )
    mean = x.mean(dim=0)
    std = x.std(dim=0).clamp_min(1e-8)

    x_raw = x.detach().clone().requires_grad_(True)
    z = (x_raw - mean) / std
    # Linear map then squared projection on row 0 (toy stand-in for s_B).
    w = torch.randn(d, dtype=torch.float64)
    s = (z[0] * w).sum().pow(2)
    (g_raw,) = torch.autograd.grad(s, x_raw)

    x_z = ((x - mean) / std).detach().clone().requires_grad_(True)
    s2 = (x_z[0] * w).sum().pow(2)
    (g_z,) = torch.autograd.grad(s2, x_z)

    # Chain rule: ∂s/∂x_raw = (∂s/∂z) / std  — positive channel scales only.
    expected = g_z / std
    assert torch.allclose(g_raw, expected, rtol=1e-10, atol=1e-12)
    # No global sign flip vs differentiating in z-space.
    cos = torch.nn.functional.cosine_similarity(
        g_raw.reshape(1, -1), expected.reshape(1, -1)
    )
    assert float(cos) > 0.999


def test_zscore_channel_scales_are_positive() -> None:
    std = torch.tensor([6.95, 0.50, 0.47], dtype=torch.float64)
    inv = 1.0 / std
    assert torch.all(inv > 0)
