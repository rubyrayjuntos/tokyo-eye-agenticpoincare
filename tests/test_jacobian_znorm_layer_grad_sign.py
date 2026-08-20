"""Unit checks for Jacobian z-norm layer grad-sign helpers."""

from __future__ import annotations

import numpy as np
import torch

from experiments.diagnostics.jacobian_znorm_layer_grad_sign import (
    _cos,
    _sign_agree,
    cos_distribution_summary,
    per_residue_cos_act_grad,
    polarity_pack,
)


def test_cos_aligned_and_anti() -> None:
    a = torch.tensor([1.0, 2.0, 3.0])
    assert abs(_cos(a, a) - 1.0) < 1e-6
    assert abs(_cos(a, -a) + 1.0) < 1e-6


def test_sign_agree() -> None:
    a = torch.tensor([1.0, -2.0, 3.0])
    b = torch.tensor([0.5, -1.0, 9.0])
    assert _sign_agree(a, b) == 1.0
    assert _sign_agree(a, -b) == 0.0


def test_polarity_pack_none_grad() -> None:
    p = polarity_pack(torch.ones(3), None)
    assert p["has_grad"] is False
    assert p["cos_act_grad"] is None


def test_per_residue_cos_and_distribution_shapes() -> None:
    act = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    grad_aligned = torch.tensor([[0.01, 0.0], [0.0, 0.02], [0.03, 0.03]])
    cos = per_residue_cos_act_grad(act, grad_aligned, n=3)
    assert cos.shape == (3,)
    assert float(np.min(cos)) > 0.99

    tight = np.array([0.04, 0.05, -0.02, 0.01, 0.06, -0.03, 0.0, 0.02])
    assert cos_distribution_summary(tight)["shape_read"] == "attenuation_tight_near_zero"

    compressed = np.array(
        [0.05, 0.12, -0.08, 0.20, 0.02, -0.05, 0.15, 0.08, -0.02, 0.22]
    )
    assert (
        cos_distribution_summary(compressed)["shape_read"]
        == "attenuation_compressed_near_zero"
    )

    wide = np.array([-0.9, 0.85, -0.7, 0.95, -0.6, 0.8, -0.95, 0.7])
    assert cos_distribution_summary(wide)["shape_read"] == "incoherence_wide_scatter"
