"""Unit tests for core_radial_floor_loss (e1 origin lift)."""

from __future__ import annotations

import torch

from science.training.disc_occupancy import core_radial_floor_loss


def test_core_radial_floor_penalizes_high_rho_low_tau_near_origin() -> None:
    xy = torch.tensor([[0.02, 0.0], [0.40, 0.0]], dtype=torch.float32)
    rho = torch.tensor([25.0, 5.0], dtype=torch.float32)
    tau = torch.tensor([0.0, 1.0], dtype=torch.float32)
    out = core_radial_floor_loss(xy, rho, tau, min_r=0.15)
    assert float(out["core_radial_floor"]) > 0.0
    # Weight mass on core residue only → weighted r near first point.
    assert float(out["core_radial_floor_r_weighted"]) < 0.1


def test_core_radial_floor_zero_when_above_min_r() -> None:
    xy = torch.tensor([[0.20, 0.0], [0.30, 0.0]], dtype=torch.float32)
    rho = torch.tensor([25.0, 20.0], dtype=torch.float32)
    tau = torch.tensor([0.0, 0.0], dtype=torch.float32)
    out = core_radial_floor_loss(xy, rho, tau, min_r=0.15)
    assert float(out["core_radial_floor"]) == 0.0
