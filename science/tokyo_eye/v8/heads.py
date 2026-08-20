"""v8-local downstream heads (SDRP dual-space + evidential) and losses."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from science.tokyo_eye.v8.attention import log_map_zero


class DualSpaceSDRPHead(nn.Module):
    """Dual-space cross-attention → per-node SDRP state logits.

    Queries from hyperbolic logmap₀(z); keys/values from Euclidean hidden ``h``.
    """

    def __init__(self, dim: int, num_classes: int, *, c: float = 1.0) -> None:
        super().__init__()
        self.c = float(c)
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, num_classes)

    def forward(self, z_hyp: torch.Tensor, h_euc: torch.Tensor) -> torch.Tensor:
        q = self.q(log_map_zero(z_hyp, c=self.c))
        k = self.k(h_euc)
        v = self.v(h_euc)
        # Node-wise dual fusion (not dense N×N board): gated residual mix
        attn = torch.sigmoid((q * k).sum(dim=-1, keepdim=True) / (q.shape[-1] ** 0.5))
        fused = attn * v + (1.0 - attn) * q
        return self.out(fused)


class EvidentialHead(nn.Module):
    """Deep evidential regression-style 4-tuple ``(γ, ν, α, β)`` per node."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Linear(dim, 4)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        raw = self.net(h)
        gamma = raw[..., 0]
        nu = F.softplus(raw[..., 1]) + 1e-4
        alpha = F.softplus(raw[..., 2]) + 1.0
        beta = F.softplus(raw[..., 3]) + 1e-4
        return torch.stack([gamma, nu, alpha, beta], dim=-1)


class MechanismScoreHead(nn.Module):
    """Scalar mechanism score for margin hinge supervision."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 1))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).squeeze(-1)


def mechanism_margin_loss_v2(
    score: torch.Tensor,
    pos: torch.Tensor,
    neg: torch.Tensor,
    *,
    margin: float = 0.1,
) -> torch.Tensor:
    """Stabilized L1-hinge: ``mean(relu(margin - (score_pos - score_neg)))``.

    ``pos`` / ``neg`` are target mechanism strengths; we form a pairwise gap
    proxy ``score - neg`` vs ``pos - score`` style hinge on ``score`` ranking.
    """
    # Encourage score to sit above midpoint toward positives
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
