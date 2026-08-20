"""Unit tests for T1a input feature z-score / |ρ−TAU| transforms."""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from science.dtie.common.input_feature_norm import (
    fit_and_install_input_feature_norm,
    fit_input_feature_stats,
    rewrite_tau_channel,
    transform_node_features,
)
from science.dtie.common.residue_features import TAU


def test_rewrite_tau_abs_dist() -> None:
    x = np.array([[10.0, 1.0, 0.0, 50.0], [20.0, 0.0, 1.0, 80.0]], dtype=np.float64)
    out = rewrite_tau_channel(x, mode="abs_dist")
    assert np.allclose(out[:, 1], np.abs(x[:, 0] - TAU))
    assert np.allclose(out[:, [0, 2, 3]], x[:, [0, 2, 3]])


def test_zscore_centers_and_scales() -> None:
    mean = torch.tensor([0.0, 0.0, 0.0, 100.0])
    std = torch.tensor([1.0, 1.0, 1.0, 10.0])
    x = torch.tensor([[0.0, 0.0, 0.0, 110.0], [1.0, -1.0, 0.5, 90.0]])
    out = transform_node_features(x, mean=mean, std=std, zscore=True)
    assert torch.allclose(out[:, 3], torch.tensor([1.0, -1.0]))
    assert torch.allclose(out[:, 0], torch.tensor([0.0, 1.0]))


def test_zscore_topology_three_vector() -> None:
    """T1a must accept topology_three_vector [ρ, τ, ss] — not hard-require width 4."""
    mean = torch.tensor([10.0, 0.5, 1.0])
    std = torch.tensor([2.0, 0.5, 0.5])
    x = torch.tensor([[12.0, 1.0, 1.5], [8.0, 0.0, 0.5]])
    out = transform_node_features(x, mean=mean, std=std, zscore=True)
    assert out.shape == (2, 3)
    assert torch.allclose(out[:, 0], torch.tensor([1.0, -1.0]))
    assert torch.allclose(out[:, 1], torch.tensor([1.0, -1.0]))


def test_fit_and_install_on_tiny_model() -> None:
    class _M(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = torch.nn.Linear(4, 4)

    rng = np.random.default_rng(0)
    proteins = []
    for _ in range(3):
        x = rng.normal(size=(8, 4)).astype(np.float32)
        x[:, 3] = x[:, 3] * 100 + 200  # SASA-scale channel
        proteins.append({"data": Data(x=torch.from_numpy(x))})

    model = _M()
    stats = fit_and_install_input_feature_norm(model, proteins, zscore=True)
    assert stats["input_feature_zscore"] is True
    assert hasattr(model, "input_feat_mean")
    assert model.input_feat_std[3].item() > 10.0

    mean, std = fit_input_feature_stats(proteins)
    assert np.allclose(mean, stats["mean"])
    assert np.allclose(std, stats["std"])
