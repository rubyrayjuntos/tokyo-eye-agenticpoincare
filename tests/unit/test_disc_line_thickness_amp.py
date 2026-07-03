"""SVD thickness floors must work under AMP / fp16 activations."""

from __future__ import annotations

import pytest
import torch

from science.training.disc_occupancy import (
    ball_line_thickness_floor_loss,
    disc_line_thickness_from_tensor,
)


def test_disc_line_thickness_fp16_input() -> None:
    x = torch.randn(80, 64, dtype=torch.float16)
    thick = disc_line_thickness_from_tensor(x)
    assert thick.dtype == torch.float16
    assert float(thick) > 0.05


def test_ball_line_thickness_floor_loss_grad_fp32() -> None:
    x = torch.randn(120, 32, requires_grad=True)
    x.data[:, 1:] *= 0.001  # thin wedge along PC1
    losses = ball_line_thickness_floor_loss(x, min_thickness=0.18)
    assert float(losses["x_hyp_line_thickness_rms"].detach()) > 0.0
    assert float(losses["x_hyp_thickness_floor"].detach()) > 0.0
    losses["x_hyp_thickness_floor"].backward()
    assert x.grad is not None
    assert float(x.grad.abs().sum()) > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for AMP test")
def test_ball_line_thickness_floor_loss_under_cuda_autocast() -> None:
    x = torch.randn(168, 128, device="cuda", dtype=torch.float16, requires_grad=True)
    with torch.autocast("cuda"):
        losses = ball_line_thickness_floor_loss(x, min_thickness=0.18)
        total = losses["x_hyp_thickness_floor"]
    assert float(losses["x_hyp_line_thickness_rms"]) > 0.05
    assert float(total) > 0.0
    total.backward()
    assert x.grad is not None
    assert float(x.grad.abs().sum()) > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for AMP test")
def test_gosp_loss_x_hyp_thickness_under_autocast() -> None:
    from science.dtie.v6.loss import gosp_loss_v6

    device = "cuda"
    n, hidden = 168, 128
    # Thin collinear ball activations (simulates wedge collapse).
    base = torch.randn(n, 1, device=device, dtype=torch.float16)
    x_hyp = torch.cat([base, torch.randn(n, hidden - 1, device=device) * 0.001], dim=1)
    x_hyp.requires_grad_(True)
    output = {
        "x_hyp": x_hyp,
        "hyp_projections_2d": torch.randn(n, 2, device=device) * 0.1,
        "hyp_projections_3d": torch.randn(n, 3, device=device) * 0.1,
        "cone_depth": torch.rand(n, 1, device=device),
        "radial_features": torch.rand(n, 1, device=device),
        "angular_features": torch.randn(n, hidden, device=device),
        "uncertainty": {"epistemic": torch.rand(n, 1, device=device)},
        "routing_entropy": torch.tensor(1.2, device=device),
        "expert_load": torch.ones(4, device=device) / 4,
    }
    with torch.autocast("cuda"):
        losses = gosp_loss_v6(
            output=output,
            target_rho=torch.rand(n, device=device),
            ca_coords=torch.randn(n, 3, device=device),
            x_hyp_thickness_floor_coeff=6.0,
            x_hyp_thickness_floor_min=0.18,
            evidential_coeff=0.0,
            balance_coeff=0.0,
            cone_coeff=0.0,
            neighborhood_coeff=0.0,
            angular_coeff=0.0,
            shell_corr_coeff=0.0,
            proj_violation_coeff=0.0,
        )
        total = losses["total"]
    assert float(losses["x_hyp_line_thickness_rms"]) > 0.0
    assert float(losses["x_hyp_thickness_floor"]) > 0.0
    total.backward()
    assert output["x_hyp"].grad is not None
    assert float(output["x_hyp"].grad.abs().sum()) > 0.0
