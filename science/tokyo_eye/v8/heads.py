"""v8-local downstream heads (SDRP dual-space + evidential) and losses."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from science.tokyo_eye.v8.attention import (
    GyroOrthogonalMap,
    exp_map_zero,
    gyroscalar_mul,
    mobius_add,
    poincare_dist,
    project_to_ball,
    tau_relative_residual_step,
)


class DualSpaceSDRPHead(nn.Module):
    """Class logits from Poincaré distances to learned prototypes.

    Euclidean skip is lifted via ``exp₀`` (allowed: Linear on Euclidean ``h``).
    Hyperbolic queries stay on-manifold via gyro maps + Möbius mix.
    """

    def __init__(self, dim: int, num_classes: int, *, c: float = 1.0) -> None:
        super().__init__()
        self.c = float(c)
        self.num_classes = int(num_classes)
        self.R_z = GyroOrthogonalMap(dim)
        self.euc_to_tangent = nn.Linear(dim, dim, bias=False)
        self.prototypes = nn.Parameter(0.05 * torch.randn(num_classes, dim))
        # Undamped q⊕he is the same residual-inflation class as attn/MoE.
        self.residual_logit = nn.Parameter(torch.tensor(-1.0986122886681098))

    def forward(
        self,
        z_hyp: torch.Tensor,
        h_euc: torch.Tensor,
        *,
        tau_ceiling: float = 0.70,
    ) -> torch.Tensor:
        z = project_to_ball(z_hyp, c=self.c)
        q = self.R_z(z, c=self.c, eps=1e-5)
        he = exp_map_zero(self.euc_to_tangent(h_euc), c=self.c)
        r = torch.linalg.vector_norm(q, dim=-1)
        step = tau_relative_residual_step(
            self.residual_logit, tau_ceiling=float(tau_ceiling), radius=r
        )
        he_step = gyroscalar_mul(step, he, c=self.c, eps=1e-5)
        fused = mobius_add(q, he_step, c=self.c)
        proto = exp_map_zero(self.prototypes.to(dtype=fused.dtype), c=self.c)
        n, d = fused.shape
        k = proto.shape[0]
        fused_rep = fused.unsqueeze(1).expand(n, k, d).reshape(n * k, d)
        proto_rep = proto.unsqueeze(0).expand(n, k, d).reshape(n * k, d)
        dist = poincare_dist(fused_rep, proto_rep, c=self.c)
        return -dist.view(n, k)


class EvidentialHead(nn.Module):
    """Deep evidential 4-tuple ``(γ, ν, α, β)`` with radius vacuity."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Linear(dim, 4)

    def forward(
        self, h: torch.Tensor, *, radius: torch.Tensor | None = None
    ) -> torch.Tensor:
        raw = self.net(h)
        gamma = raw[..., 0]
        nu = F.softplus(raw[..., 1]) + 1e-4
        alpha = F.softplus(raw[..., 2]) + 1.0
        beta = F.softplus(raw[..., 3]) + 1e-4
        if radius is not None:
            r = radius.reshape_as(nu)
            # High Poincaré radius → more epistemic pressure (vacuity).
            nu = nu / (1.0 + r)
            beta = beta * (1.0 + r)
        return torch.stack([gamma, nu, alpha, beta], dim=-1)


class MechanismScoreHead(nn.Module):
    """Scalar mechanism score for margin hinge supervision."""

    def __init__(self, dim: int, *, hidden: int | None = None) -> None:
        super().__init__()
        hid = int(dim if hidden is None else hidden)
        self.net = nn.Sequential(nn.Linear(dim, hid), nn.SiLU(), nn.Linear(hid, 1))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).squeeze(-1)


def mechanism_margin_loss_v2(
    score: torch.Tensor,
    pos: torch.Tensor,
    neg: torch.Tensor,
    *,
    margin: float = 0.1,
) -> torch.Tensor:
    """Stabilized L1-hinge: ``mean(relu(margin - (score_pos - score_neg)))``."""
    target_gap = (pos - neg).detach()
    pred_gap = score - neg
    return F.relu(margin + target_gap * 0.0 - (pred_gap - target_gap)).abs().mean()


def sdrp_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return F.cross_entropy(logits, target)


__all__ = [
    "DualSpaceSDRPHead",
    "EvidentialHead",
    "MechanismScoreHead",
    "mechanism_margin_loss_v2",
    "sdrp_cross_entropy",
]
