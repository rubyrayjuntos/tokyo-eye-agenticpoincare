"""TokyoEyesHyperbolicV8 — top-level forward spine (Sprint 4).

Accepts EquiformerV3-style residue scalars ``s`` and vectors ``v`` (or any
compatible front-end). Does not import v7 ``TokyoEye`` / v66.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from science.tokyo_eye.v8.attention import (
    HyperbolicGraphAttention,
    clamp_ball_radius,
    project_to_ball,
)
from science.tokyo_eye.v8.heads import (
    DualSpaceSDRPHead,
    EvidentialHead,
    MechanismScoreHead,
)
from science.tokyo_eye.v8.moe import TopologyAwareHardMoE
from science.tokyo_eye.v8.projector import RadialAngularProjector


class TokyoEyesHyperbolicV8(nn.Module):
    """Lift → HypGraphAttention×L → Hard MoE → SDRP / evidential / mechanism."""

    def __init__(
        self,
        *,
        scalar_dim: int,
        vector_dim: int,
        hidden_dim: int,
        num_attn_layers: int = 2,
        num_relations: int = 6,
        num_sdrp_classes: int = 8,
        gate_hidden: int = 16,
        c: float = 1.0,
        eps: float = 1e-5,
        moe_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if int(num_attn_layers) not in (2, 3):
            raise ValueError("num_attn_layers must be in {2, 3} (freeze §7.3)")
        self.hidden_dim = int(hidden_dim)
        self.eps = float(eps)
        # Learned curvature (freeze §6). softplus(log_c) > 0.
        init_c = max(float(c), 1e-3)
        self._log_c = nn.Parameter(torch.log(torch.expm1(torch.tensor(init_c))))
        self.tau_clamp_mode = "per_stage"

        self.projector = RadialAngularProjector(
            scalar_dim, vector_dim, hidden_dim, c=float(c), eps=eps
        )
        self.attn_layers = nn.ModuleList(
            [
                HyperbolicGraphAttention(
                    hidden_dim, num_relations=num_relations, c=float(c), eps=eps
                )
                for _ in range(num_attn_layers)
            ]
        )
        self.moe = TopologyAwareHardMoE(
            hidden_dim,
            gate_hidden=gate_hidden,
            temperature=moe_temperature,
            c=float(c),
            eps=eps,
        )
        self.euc_skip = nn.Linear(scalar_dim, hidden_dim)
        self.sdrp_head = DualSpaceSDRPHead(hidden_dim, num_sdrp_classes, c=float(c))
        self.evidential_head = EvidentialHead(hidden_dim)
        self.mechanism_head = MechanismScoreHead(hidden_dim)

    @property
    def c(self) -> float:
        return float(self.curvature().detach())

    def curvature(self) -> torch.Tensor:
        return F.softplus(self._log_c) + 1e-4

    def _bind_c(self, c_val: float) -> None:
        self.projector.c = c_val
        self.moe.c = c_val
        self.sdrp_head.c = c_val
        for layer in self.attn_layers:
            layer.c = c_val

    def set_moe_temperature(self, tau: float) -> None:
        self.moe.set_temperature(tau)

    def set_moe_explore_epsilon(self, epsilon: float) -> None:
        self.moe.set_explore_epsilon(epsilon)

    def set_moe_mode(self, mode: str) -> None:
        self.moe.set_moe_mode(mode)

    def forward(
        self,
        s: torch.Tensor,
        v: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        *,
        tau_ceiling: float = 0.995,
        edge_attr: torch.Tensor | None = None,
        chem: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Full v8 spine."""

        c_val = float(self.curvature().detach().to(device=s.device, dtype=s.dtype))
        self._bind_c(c_val)

        def _tau_clamp(z_in: torch.Tensor, *, force: bool = False) -> torch.Tensor:
            z_out = project_to_ball(z_in, c=c_val, eps=self.eps)
            mode = getattr(self, "tau_clamp_mode", "per_stage")
            if mode == "final_only" and not force:
                return z_out
            return clamp_ball_radius(z_out, max_r=float(tau_ceiling), eps=self.eps)

        z = self.projector(s, v, tau_ceiling=tau_ceiling)
        z = _tau_clamp(z, force=False)
        z_lift = z

        for layer in self.attn_layers:
            z = layer(
                z, edge_index, edge_type, edge_attr, tau_ceiling=float(tau_ceiling)
            )
            z = _tau_clamp(z, force=False)
        z_attn = z

        h_euc = self.euc_skip(s) + 2.0
        z_moe, moe_aux = self.moe(
            z, edge_index, h=h_euc, chem=chem, tau_ceiling=float(tau_ceiling)
        )
        z_moe = _tau_clamp(z_moe, force=True)

        sdrp_logits = self.sdrp_head(z_moe, h_euc, tau_ceiling=float(tau_ceiling))
        radius = torch.linalg.vector_norm(z_moe, dim=-1)
        evidence = self.evidential_head(h_euc, radius=radius)
        mechanism_score = self.mechanism_head(h_euc)

        return {
            "z_hyp": z_moe,
            "z_lift": z_lift,
            "z_attn": z_attn,
            "h_euc": h_euc,
            "sdrp_logits": sdrp_logits,
            "evidence": evidence,
            "mechanism_score": mechanism_score,
            "moe_aux": moe_aux,
            "tau_ceiling": float(tau_ceiling),
            "curvature": c_val,
        }


__all__ = ["TokyoEyesHyperbolicV8"]
