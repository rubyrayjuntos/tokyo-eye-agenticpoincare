"""Topology-aware hard-commitment MoE guilds (TokyoEye-v8 Sprint 3).

Isolated under ``science.tokyo_eye.v8``. Post-transport specialization into
E0–E3 with train-time ``gumbel_softmax(..., hard=True)`` (STE) and eval argmax.

FROZEN: ``docs/superpowers/specs/2026-07-22-tokyo-eye-v8-equiformer-hyp-design.md`` §8.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter

from science.tokyo_eye.v8.attention import poincare_dist, project_to_ball

# E0 wrapped core | E1 dehydron rim | E2 interface/spoke | E3 coil/solvent
NUM_EXPERTS = 4


def topology_gate_features(
    z: torch.Tensor,
    h_proj: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    c: float = 1.0,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Build structural gate vector ``[r, density, deg | h_proj]``.

    * ``r``: Poincaré radius ``||z||``
    * ``density``: mean ``-d_H(z_i, z_j)`` over sparse neighbors (0 if isolate)
    * ``deg``: incoming edge count (``edge_index[1]``)
    * ``h_proj``: invariant scalar projection of the hidden state
    """
    if z.ndim != 2 or h_proj.ndim != 2:
        raise ValueError("z and h_proj must be [N, *]")
    n = z.shape[0]
    if h_proj.shape[0] != n:
        raise ValueError("h_proj batch must match z")

    z = project_to_ball(z, c=c, eps=eps)
    radius = torch.linalg.vector_norm(z, dim=-1)

    if edge_index.numel() == 0:
        density = torch.zeros(n, device=z.device, dtype=z.dtype)
        deg = torch.zeros(n, device=z.device, dtype=z.dtype)
    else:
        src = edge_index[0].long()
        dst = edge_index[1].long()
        # Incoming degree at destination
        deg = scatter(
            torch.ones(dst.shape[0], device=z.device, dtype=z.dtype),
            dst,
            dim=0,
            dim_size=n,
            reduce="sum",
        )
        # Density is a structural feature — detach hyp geometry to avoid
        # singular acosh grads through the MoE gate.
        with torch.no_grad():
            d_e = poincare_dist(z[src], z[dst], c=c, eps=eps)
            neg_d = -d_e
            sum_i = scatter(neg_d, src, dim=0, dim_size=n, reduce="sum")
            sum_j = scatter(neg_d, dst, dim=0, dim_size=n, reduce="sum")
            cnt_i = scatter(
                torch.ones_like(neg_d), src, dim=0, dim_size=n, reduce="sum"
            )
            cnt_j = scatter(
                torch.ones_like(neg_d), dst, dim=0, dim_size=n, reduce="sum"
            )
            neigh_sum = sum_i + sum_j
            neigh_cnt = (cnt_i + cnt_j).clamp_min(1.0)
            density = neigh_sum / neigh_cnt
            touched = (cnt_i + cnt_j) > 0
            density = torch.where(touched, density, torch.zeros_like(density))

    return torch.cat(
        [
            radius.unsqueeze(-1),
            density.unsqueeze(-1),
            deg.unsqueeze(-1),
            h_proj,
        ],
        dim=-1,
    )


def cv_load_balance_loss(routing: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Coefficient-of-variation penalty over mean expert load fractions.

    ``routing``: ``[N, E]`` one-hot (or soft) assignment rows.
    Returns **unscaled** CV — harness applies ``cv_coeff``.
    """
    # Fraction of nodes assigned to each expert
    load = routing.mean(dim=0)  # [E]
    mean = load.mean()
    std = load.std(unbiased=False)
    return std / (mean.abs() + eps)


def min_load_quota_loss(
    routing: torch.Tensor,
    *,
    floor: float = 0.05,
) -> torch.Tensor:
    """Soft minimum-load hinge: ``Σ_e ReLU(floor − load_e)²``.

    ``routing``: ``[N, E]`` hard STE (train) or one-hot (eval) rows.
    Returns **unscaled** quota — harness applies ``moe_quota_coeff``.
    """
    if floor < 0.0 or floor > 1.0:
        raise ValueError("floor must be in [0, 1]")
    load = routing.mean(dim=0)  # [E]
    return torch.sum(F.relu(float(floor) - load) ** 2)


class TopologyAwareHardMoE(nn.Module):
    """E0–E3 guilds with topology gate + hard Gumbel (train) / argmax (eval)."""

    def __init__(
        self,
        dim: int,
        *,
        num_experts: int = NUM_EXPERTS,
        gate_hidden: int = 16,
        temperature: float = 1.0,
        c: float = 1.0,
        eps: float = 1e-5,
        cv_coeff: float = 1.0,
        moe_quota_floor: float = 0.05,
    ) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError("dim must be >= 1")
        if num_experts < 2:
            raise ValueError("num_experts must be >= 2")
        self.dim = int(dim)
        self.num_experts = int(num_experts)
        self.gate_hidden = int(gate_hidden)
        self.temperature = float(temperature)
        self.c = float(c)
        self.eps = float(eps)
        # Deprecated: scaling is harness SSOT (Sprint 9). Kept for ctor compat.
        self.cv_coeff = float(cv_coeff)
        self.moe_quota_floor = float(moe_quota_floor)

        self.h_proj = nn.Linear(dim, gate_hidden)
        gate_in = 3 + gate_hidden
        self.gate = nn.Sequential(
            nn.Linear(gate_in, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, num_experts),
        )
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.SiLU(),
                    nn.Linear(dim, dim),
                )
                for _ in range(num_experts)
            ]
        )

    def set_temperature(self, tau: float) -> None:
        """Curriculum cool-down for Gumbel logits (tau > 0)."""
        if tau <= 0:
            raise ValueError("temperature must be > 0")
        self.temperature = float(tau)

    def _route(self, logits: torch.Tensor) -> torch.Tensor:
        """Return ``[N, E]`` hard assignment (STE in train, argmax in eval)."""
        if self.training:
            # Frozen contract: hard=True required — soft mush forbidden.
            return F.gumbel_softmax(
                logits, tau=self.temperature, hard=True, dim=-1
            )
        idx = logits.argmax(dim=-1)
        return F.one_hot(idx, num_classes=self.num_experts).to(
            dtype=logits.dtype
        )

    def forward(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        h: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Route post-transport states through a single hard-selected expert.

        Args:
            z: ``[N, d]`` manifold (or ambient) node states after hyp attention.
            edge_index: ``[2, E]`` sparse R0–R5 graph (degree / density).
            h: optional Euclidean hidden for invariant projection; defaults to ``z``.

        Returns:
            ``(out, aux)`` with unscaled ``cv_loss`` / ``quota_loss`` and ``load``.
        """
        if z.ndim != 2 or z.shape[1] != self.dim:
            raise ValueError(f"z must be [N, {self.dim}]")
        if h is None:
            h = z
        if h.shape != z.shape:
            raise ValueError("h must match z shape")

        h_proj = self.h_proj(h)
        feats = topology_gate_features(
            z, h_proj, edge_index, c=self.c, eps=self.eps
        )
        logits = self.gate(feats)
        routing = self._route(logits)  # [N, E] one-hot (STE)

        # Expert delta in ambient space, residual on manifold input (preserves pre-MoE geometry).
        stacked = torch.stack([expert(z) for expert in self.experts], dim=1)
        delta = (routing.unsqueeze(-1) * stacked).sum(dim=1)
        out = project_to_ball(z + delta, c=self.c, eps=self.eps)

        load = routing.mean(dim=0)
        cv_loss = cv_load_balance_loss(routing)  # unscaled — harness applies λ_cv
        quota_loss = min_load_quota_loss(
            routing, floor=self.moe_quota_floor
        )  # unscaled — harness applies λ_quota
        aux: dict[str, Any] = {
            "routing": routing,
            "logits": logits,
            "load": load,
            "cv_loss": cv_loss,
            "quota_loss": quota_loss,
            "temperature": self.temperature,
            "hard": True if self.training else "argmax",
        }
        return out, aux


__all__ = [
    "NUM_EXPERTS",
    "TopologyAwareHardMoE",
    "cv_load_balance_loss",
    "min_load_quota_loss",
    "topology_gate_features",
]
