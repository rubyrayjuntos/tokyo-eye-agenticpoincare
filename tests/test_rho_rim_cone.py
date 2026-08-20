"""Continuous rho_rim cone target (no binary τ depth)."""

from __future__ import annotations

import torch

from science.dtie.v6.loss import cone_alignment_loss, resolve_cone_target_depth


def test_rho_rim_inverts_polarity_vs_rho_wrap() -> None:
    rho = torch.tensor([5.0, 15.0, 25.0])
    wrap = resolve_cone_target_depth(target_rho=rho, mode="rho_wrap")
    rim = resolve_cone_target_depth(target_rho=rho, mode="rho_rim")
    assert torch.allclose(rim, 1.0 - wrap)
    # Low ρ → higher depth (rim)
    assert float(rim[0]) > float(rim[2])


def test_rho_rim_cone_loss_prefers_matching_depth() -> None:
    rho = torch.tensor([5.0, 15.0, 25.0])
    target = resolve_cone_target_depth(target_rho=rho, mode="rho_rim")
    good = cone_alignment_loss(
        target.clone().unsqueeze(-1), target_rho=rho.unsqueeze(-1), mode="rho_rim"
    )
    bad = cone_alignment_loss(
        (1.0 - target).unsqueeze(-1), target_rho=rho.unsqueeze(-1), mode="rho_rim"
    )
    assert float(good) < float(bad)


def test_tau_mode_unchanged() -> None:
    tau = torch.tensor([0.0, 1.0, 1.0])
    d = resolve_cone_target_depth(target_dehydron=tau, mode="tau_dehydron_rim")
    assert torch.allclose(d, tau)
