"""Residual radial×angular fusion ablation."""

from __future__ import annotations

import torch

from science.dtie.v6.gnn.model import GOSPConeMapperV6


def test_angular_lift_zero_init_matches_angular_only() -> None:
    angular_only = GOSPConeMapperV6(radial_angular_recombine="multiply", legacy_disc_projection=False)
    lift = GOSPConeMapperV6(radial_angular_recombine="angular_lift", legacy_disc_projection=False)
    lift.load_state_dict(angular_only.state_dict(), strict=False)

    x = torch.randn(12, angular_only.hidden)
    radial = angular_only.radial_head(x)
    angular = angular_only.angular_head(x)
    t_ang = angular
    t_lift = lift._tangent_from_radial_angular(radial, angular)
    assert torch.allclose(t_ang, t_lift, atol=1e-6)


def test_mlp_fusion_zero_init_matches_multiply() -> None:
    multiply = GOSPConeMapperV6(radial_angular_recombine="multiply", legacy_disc_projection=False)
    fusion = GOSPConeMapperV6(radial_angular_recombine="mlp_fusion", legacy_disc_projection=False)
    fusion.load_state_dict(multiply.state_dict(), strict=False)

    x = torch.randn(12, multiply.hidden)
    radial = multiply.radial_head(x)
    angular = multiply.angular_head(x)
    t_mul = multiply._tangent_from_radial_angular(radial, angular)
    t_fus = fusion._tangent_from_radial_angular(radial, angular)
    assert torch.allclose(t_mul, t_fus, atol=1e-6)
