"""RadialAngularProjector — Option B lift into Poincaré ball (v8-local)."""

from __future__ import annotations

import torch
import torch.nn as nn

from science.tokyo_eye.v8.attention import exp_map_zero, project_to_ball


class RadialAngularProjector(nn.Module):
    """Option B: ``v_lifted = α · â`` then ``exp₀``, clamped by ``tau_ceiling``.

    ``α = σ(MLP(s))`` scales the unit angular direction of Equiformer vectors.
    """

    def __init__(
        self,
        scalar_dim: int,
        vector_dim: int,
        hidden_dim: int,
        *,
        c: float = 1.0,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.c = float(c)
        self.eps = float(eps)
        self.radial_mlp = nn.Sequential(
            nn.Linear(scalar_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.angular_proj = nn.Linear(vector_dim, hidden_dim, bias=False)
        self.scalar_proj = nn.Linear(scalar_dim, hidden_dim)

    def forward(
        self,
        s: torch.Tensor,
        v: torch.Tensor,
        *,
        tau_ceiling: float = 0.995,
    ) -> torch.Tensor:
        """Lift Equiformer scalars/vectors to ball points ``[N, hidden_dim]``."""
        alpha = torch.sigmoid(self.radial_mlp(s))  # [N, 1] in (0,1)
        # Scale into curriculum radius: α_eff ∈ (0, tau_ceiling)
        alpha = alpha * float(tau_ceiling)
        v_h = self.angular_proj(v)
        v_norm = torch.linalg.vector_norm(v_h, dim=-1, keepdim=True).clamp_min(self.eps)
        a_hat = v_h / v_norm
        # Keep tangent small so exp₀ does not saturate → flat radial histogram
        scalar_t = 0.01 * torch.tanh(self.scalar_proj(s))
        tangent = alpha * a_hat + scalar_t
        t_norm = torch.linalg.vector_norm(tangent, dim=-1, keepdim=True).clamp_min(self.eps)
        max_t = 2.0  # artanh(tanh(2)) stays far from boundary under exp₀
        tangent = tangent * torch.clamp(max_t / t_norm, max=1.0)
        z = exp_map_zero(tangent, c=self.c, eps=self.eps)
        return project_to_ball(z, c=self.c, eps=self.eps)
