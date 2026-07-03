"""Tests for Step 4 epistemic-depth decoupling loss."""

from __future__ import annotations

import torch

from science.dtie.v6.loss import epistemic_decoupling_loss, gosp_loss_v6


def test_epistemic_decoupling_prefers_bf_residual_alignment() -> None:
    n = 64
    depth = torch.linspace(0.2, 1.0, n)
    latent = torch.randn(n)
    bf_z = 0.5 * depth + 0.8 * latent
    sasa = 0.3 * depth + 0.2 * torch.randn(n)

    epi_good = depth + 0.9 * latent + 0.05 * torch.randn(n)
    epi_bad = depth + 0.02 * torch.randn(n)

    good = epistemic_decoupling_loss(
        epi_good.unsqueeze(1), depth.unsqueeze(1), sasa, bf_z.unsqueeze(1)
    )
    bad = epistemic_decoupling_loss(
        epi_bad.unsqueeze(1), depth.unsqueeze(1), sasa, bf_z.unsqueeze(1)
    )

    assert good["r_epi_bf_resid"].item() > bad["r_epi_bf_resid"].item()
    assert good["epistemic_bf_align"].item() < bad["epistemic_bf_align"].item()


def test_epistemic_decoupling_sasa_penalty_rises_with_partial_corr() -> None:
    n = 80
    depth = torch.linspace(0.1, 0.9, n)
    sasa = depth + 0.15 * torch.randn(n)
    bf = depth + 0.2 * torch.randn(n)

    epi_coupled = depth + 0.85 * sasa
    epi_decoupled = depth + 0.1 * torch.randn(n)

    coupled = epistemic_decoupling_loss(
        epi_coupled.unsqueeze(1), depth.unsqueeze(1), sasa, bf.unsqueeze(1)
    )
    decoupled = epistemic_decoupling_loss(
        epi_decoupled.unsqueeze(1), depth.unsqueeze(1), sasa, bf.unsqueeze(1)
    )

    assert coupled["epistemic_sasa_pen"].item() > decoupled["epistemic_sasa_pen"].item()


def test_gosp_loss_v6_epistemic_decoupling_terms() -> None:
    n = 24
    device = torch.device("cpu")
    depth = torch.linspace(0.3, 1.2, n, device=device).unsqueeze(1)
    sasa = torch.linspace(0.0, 1.0, n, device=device)
    bf = (depth.squeeze() * 10 + torch.randn(n, device=device)).unsqueeze(1)
    output = {
        "x_hyp": torch.randn(n, 2, device=device),
        "x_routed_hyp": torch.randn(n, 2, device=device),
        "radial_features": torch.linspace(0.1, 0.9, n, device=device).unsqueeze(1),
        "cone_depth": depth,
        "hyp_projections_2d": torch.randn(n, 2, device=device) * 0.25,
        "hyp_projections_3d": torch.randn(n, 3, device=device) * 0.3,
        "capacity_loss": torch.tensor(0.1, device=device),
        "routing_entropy": torch.tensor(1.0, device=device),
        "expert_load": torch.ones(4, device=device) * 0.25,
        "evidence": {
            "mu": torch.zeros(n, device=device),
            "nu": torch.ones(n, device=device),
            "alpha": torch.ones(n, device=device) * 2.0,
            "beta": torch.ones(n, device=device),
        },
        "uncertainty": {"epistemic": torch.linspace(0.2, 1.0, n, device=device).unsqueeze(1)},
        "audit_trail": {"curvature_value": -1.0},
    }
    target_rho = torch.linspace(1.0, 20.0, n, device=device)
    ca_coords = torch.randn(n, 3, device=device)

    with_dec = gosp_loss_v6(
        output,
        target_rho,
        ca_coords,
        sasa=sasa,
        b_factor_ca=bf,
        epistemic_decoupling_coeff=1.0,
        epistemic_bf_align_coeff=0.22,
        epistemic_sasa_pen_coeff=0.246,
    )
    without = gosp_loss_v6(
        output,
        target_rho,
        ca_coords,
        sasa=sasa,
        epistemic_decoupling_coeff=0.0,
    )

    assert "epistemic_decoupling" in with_dec
    assert with_dec["epistemic_decoupling"].item() > 0.0
    assert without["epistemic_decoupling"].item() == 0.0
