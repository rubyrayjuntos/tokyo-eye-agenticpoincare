"""Sparse relation-aware Hyperbolic Graph Attention (TokyoEye-v8 Sprint 1 (science.tokyo_eye.v8)).

FROZEN contract:
``docs/superpowers/specs/2026-07-22-tokyo-eye-v8-equiformer-hyp-design.md`` §7.

* Sparse ``edge_index`` / ``edge_type`` only — no dense ``N×N`` logit board.
* Logits: ``(-d_H(Q_i, K_j) * γ_R + β_R) / sqrt(d_h)`` on each directed edge.
* Segmented softmax + Einstein midpoint via Klein coordinates.
* Option A isolates: self-transport ``exp₀(W_o(log₀(z)))`` — no empty barycenter.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.utils import scatter, softmax

# R0–R5
NUM_RELATIONS_DEFAULT = 6


def project_to_ball(x: torch.Tensor, *, c: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
    """Clamp points strictly inside the open ball of radius ``1/sqrt(c)``."""
    max_r = (1.0 / math.sqrt(c)) - eps
    # Replace non-finite coords before projecting (prevents NaN contagion).
    x = torch.nan_to_num(x, nan=0.0, posinf=max_r, neginf=-max_r)
    r = torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(eps)
    scale = torch.clamp(max_r / r, max=1.0)
    return x * scale


def poincare_dist(
    x: torch.Tensor,
    y: torch.Tensor,
    c: float = 1.0,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Strict Poincaré geodesic distance with stable gradients near x≈y.

    ``acosh`` has singular derivative at 1; we keep the argument strictly above
    ``1+eps`` and never allow zero pairwise Euclidean separation.
    """
    x = project_to_ball(x, c=c, eps=eps)
    y = project_to_ball(y, c=c, eps=eps)
    sq_norm_x = torch.sum(x * x, dim=-1).clamp(max=1.0 / c - eps)
    sq_norm_y = torch.sum(y * y, dim=-1).clamp(max=1.0 / c - eps)
    sq_dist = torch.sum((x - y) * (x - y), dim=-1).clamp_min(eps)
    denom = (1.0 - c * sq_norm_x) * (1.0 - c * sq_norm_y)
    arg = 1.0 + 2.0 * c * sq_dist / denom.clamp(min=eps)
    # Keep acosh' = 1/sqrt(arg^2-1) finite
    arg = arg.clamp(min=1.0 + 1e-4, max=1.0e6)
    return torch.acosh(arg) / math.sqrt(c)


def lorentz_factor(x: torch.Tensor, *, c: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
    """γ = 1 / sqrt(1 - c ||x||²) for Poincaré points."""
    sq = torch.sum(x * x, dim=-1, keepdim=True).clamp(max=1.0 / c - eps)
    return 1.0 / torch.sqrt((1.0 - c * sq).clamp(min=eps))


def poincare_to_klein(x: torch.Tensor, *, c: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
    """V_K = γ V_H / (1 + γ)  (= V_H / (1 + sqrt(1 - c||V_H||²)))."""
    x = project_to_ball(x, c=c, eps=eps)
    gamma = lorentz_factor(x, c=c, eps=eps)
    return (gamma * x) / (1.0 + gamma)


def klein_to_poincare(k: torch.Tensor, *, c: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
    """Inverse of ``poincare_to_klein``: V_H = 2 V_K / (1 + c ||V_K||²)."""
    sq = torch.sum(k * k, dim=-1, keepdim=True)
    x = (2.0 * k) / (1.0 + c * sq).clamp(min=eps)
    return project_to_ball(x, c=c, eps=eps)


def exp_map_zero(v: torch.Tensor, *, c: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
    """Exponential map at the origin of the Poincaré ball."""
    sqrt_c = math.sqrt(c)
    v_norm = torch.linalg.vector_norm(v, dim=-1, keepdim=True).clamp_min(eps)
    return project_to_ball(
        torch.tanh(sqrt_c * v_norm) * v / (sqrt_c * v_norm),
        c=c,
        eps=eps,
    )


def log_map_zero(x: torch.Tensor, *, c: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
    """Logarithmic map at the origin of the Poincaré ball."""
    x = project_to_ball(x, c=c, eps=eps)
    sqrt_c = math.sqrt(c)
    x_norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(eps)
    # Stay well inside (-1,1) for atanh
    u = (sqrt_c * x_norm).clamp(max=1.0 - 1e-4)
    return (torch.atanh(u) / (sqrt_c * x_norm)) * x


class HyperbolicGraphAttention(nn.Module):
    """Relation-aware sparse hyp attention with Einstein midpoint aggregation.

    Aggregation index follows Sprint-1 blueprint: for directed edge ``(i→j)``
    logits use ``(Q_i, K_j)`` and segmented softmax / scatter reduce on
    ``edge_index[0]`` (source / query node). Bidirectional R0–R5 graphs from
    Sprint 2 make this symmetric in practice.
    """

    def __init__(
        self,
        dim: int,
        *,
        num_relations: int = NUM_RELATIONS_DEFAULT,
        c: float = 1.0,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError("dim must be >= 1")
        if num_relations < 1:
            raise ValueError("num_relations must be >= 1")
        self.dim = int(dim)
        self.num_relations = int(num_relations)
        self.c = float(c)
        self.eps = float(eps)

        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)
        self.W_o = nn.Linear(dim, dim, bias=False)

        self.gamma = nn.Embedding(num_relations, 1)
        self.beta = nn.Embedding(num_relations, 1)
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        for lin in (self.W_q, self.W_k, self.W_v, self.W_o):
            nn.init.xavier_uniform_(lin.weight, gain=0.1)

    def _tangent_linear(self, z: torch.Tensor, lin: nn.Linear) -> torch.Tensor:
        return exp_map_zero(lin(log_map_zero(z, c=self.c, eps=self.eps)), c=self.c, eps=self.eps)

    def _self_transport(self, z: torch.Tensor) -> torch.Tensor:
        """Option A: ``exp₀(W_o(log₀(z)))``."""
        return exp_map_zero(
            self.W_o(log_map_zero(z, c=self.c, eps=self.eps)),
            c=self.c,
            eps=self.eps,
        )

    def forward(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sparse hyp attention.

        Args:
            z: ``[N, d]`` Poincaré ball features.
            edge_index: ``[2, E]`` directed sparse edges.
            edge_type: ``[E]`` relation ids in ``{0..num_relations-1}``.
            edge_attr: unused (Sprint-2 attrs reserved for later bias).
        """
        del edge_attr  # reserved
        if z.ndim != 2:
            raise ValueError(f"z must be [N, d], got {tuple(z.shape)}")
        n, d = z.shape
        if d != self.dim:
            raise ValueError(f"z dim {d} != layer dim {self.dim}")

        z = project_to_ball(z, c=self.c, eps=self.eps)
        self_out = self._self_transport(z)

        if edge_index.numel() == 0:
            return self_out

        if edge_index.shape[0] != 2:
            raise ValueError("edge_index must be [2, E]")
        e = int(edge_index.shape[1])
        if edge_type.numel() != e:
            raise ValueError("edge_type length must match E")
        if int(edge_type.min()) < 0 or int(edge_type.max()) >= self.num_relations:
            raise ValueError(
                f"edge_type out of range for num_relations={self.num_relations}"
            )

        src = edge_index[0].long()
        dst = edge_index[1].long()
        rel = edge_type.long()

        q = self._tangent_linear(z, self.W_q)
        k = self._tangent_linear(z, self.W_k)
        v = self._tangent_linear(z, self.W_v)

        # Edge (i→j): distance between Q_i and K_j
        d_ij = poincare_dist(q[src], k[dst], c=self.c, eps=self.eps)
        gamma_r = self.gamma(rel).squeeze(-1)
        beta_r = self.beta(rel).squeeze(-1)
        logits = (-d_ij * gamma_r + beta_r) / math.sqrt(self.dim)

        # Segmented softmax over each source node's sparse out-neighborhood
        attn = softmax(logits, src, num_nodes=n)

        v_k = poincare_to_klein(v, c=self.c, eps=self.eps)
        msg = attn.unsqueeze(-1) * v_k[dst]
        agg_k = scatter(msg, src, dim=0, dim_size=n, reduce="sum")
        agg_h = klein_to_poincare(agg_k, c=self.c, eps=self.eps)
        attended = exp_map_zero(
            self.W_o(log_map_zero(agg_h, c=self.c, eps=self.eps)),
            c=self.c,
            eps=self.eps,
        )

        # Option A: nodes with no outgoing edges never enter Einstein midpoint
        deg = scatter(
            torch.ones(e, device=z.device, dtype=z.dtype),
            src,
            dim=0,
            dim_size=n,
            reduce="sum",
        )
        isolate = deg <= 0
        # Tangent residual: keep node identity; pure replace→oversmooth collapses the board.
        u_self = log_map_zero(self_out, c=self.c, eps=self.eps)
        u_att = log_map_zero(attended, c=self.c, eps=self.eps)
        mixed = exp_map_zero(u_self + u_att, c=self.c, eps=self.eps)
        out = torch.where(isolate.unsqueeze(-1), self_out, mixed)
        return project_to_ball(out, c=self.c, eps=self.eps)


__all__ = [
    "NUM_RELATIONS_DEFAULT",
    "HyperbolicGraphAttention",
    "exp_map_zero",
    "klein_to_poincare",
    "log_map_zero",
    "lorentz_factor",
    "poincare_dist",
    "poincare_to_klein",
    "project_to_ball",
]
