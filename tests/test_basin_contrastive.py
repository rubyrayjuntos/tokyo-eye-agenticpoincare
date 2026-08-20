"""Unit tests for Phase A basin contrastive helpers."""

from __future__ import annotations

import torch

from science.tokyo_eye.basin_contrastive import (
    basin_contrastive_margin_loss,
    r_star_indices,
)


def test_r_star_includes_switch_and_n12() -> None:
    present = set(range(1, 100))
    rs = r_star_indices(present, [12, 32])
    assert 25 in rs and 40 in rs and 57 in rs and 12 in rs and 32 in rs


def test_basin_margin_loss_positive_when_close() -> None:
    z0 = torch.zeros(8)
    z1 = torch.zeros(8)
    loss = basin_contrastive_margin_loss(z0, z1, curvature=1.0, margin=0.5)
    assert float(loss.item()) > 0.0
