"""TokyoEyesHyperbolicV8 — top-level forward spine (Sprint 4).

Accepts EquiformerV3-style residue scalars ``s`` and vectors ``v`` (or any
compatible front-end). Does not import v7 ``TokyoEye`` / v66.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from science.tokyo_eye.v8.attention import HyperbolicGraphAttention, project_to_ball
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
        if num_attn_layers < 1:
            raise ValueError("num_attn_layers must be >= 1")
        self.hidden_dim = int(hidden_dim)
        self.c = float(c)
        self.eps = float(eps)

        self.projector = RadialAngularProjector(
            scalar_dim, vector_dim, hidden_dim, c=c, eps=eps
        )
        self.attn_layers = nn.ModuleList(
            [
                HyperbolicGraphAttention(
                    hidden_dim, num_relations=num_relations, c=c, eps=eps
                )
                for _ in range(num_attn_layers)
            ]
        )
        self.moe = TopologyAwareHardMoE(
            hidden_dim,
            gate_hidden=gate_hidden,
            temperature=moe_temperature,
            c=c,
            eps=eps,
        )
        self.euc_skip = nn.Linear(scalar_dim, hidden_dim)
        self.sdrp_head = DualSpaceSDRPHead(hidden_dim, num_sdrp_classes, c=c)
        self.evidential_head = EvidentialHead(hidden_dim)
        self.mechanism_head = MechanismScoreHead(hidden_dim)

    def set_moe_temperature(self, tau: float) -> None:
        self.moe.set_temperature(tau)

    def forward(
        self,
        s: torch.Tensor,
        v: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        *,
        tau_ceiling: float = 0.995,
        edge_attr: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Full v8 spine.

        Args:
            s: Equiformer scalar features ``[N, scalar_dim]``.
            v: Equiformer vector features ``[N, vector_dim]`` (pooled).
            edge_index / edge_type: R0–R5 sparse graph.
            tau_ceiling: curriculum ball radius clamp.
        """
        z = self.projector(s, v, tau_ceiling=tau_ceiling)
        z = project_to_ball(z, c=self.c, eps=self.eps)
        # Enforce tau ceiling explicitly (stricter than 1/sqrt(c)-eps)
        r = torch.linalg.vector_norm(z, dim=-1, keepdim=True).clamp_min(self.eps)
        z = z * torch.clamp(float(tau_ceiling) / r, max=1.0)
        z_lift = z

        for layer in self.attn_layers:
            z = layer(z, edge_index, edge_type, edge_attr)
            z = project_to_ball(z, c=self.c, eps=self.eps)
            r = torch.linalg.vector_norm(z, dim=-1, keepdim=True).clamp_min(self.eps)
            z = z * torch.clamp(float(tau_ceiling) / r, max=1.0)
        z_attn = z

        h_euc = self.euc_skip(s)
        z_moe, moe_aux = self.moe(z, edge_index, h=h_euc)
        z_moe = project_to_ball(z_moe, c=self.c, eps=self.eps)
        r = torch.linalg.vector_norm(z_moe, dim=-1, keepdim=True).clamp_min(self.eps)
        z_moe = z_moe * torch.clamp(float(tau_ceiling) / r, max=1.0)

        sdrp_logits = self.sdrp_head(z_moe, h_euc)
        evidence = self.evidential_head(h_euc)
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
        }


__all__ = ["TokyoEyesHyperbolicV8"]
