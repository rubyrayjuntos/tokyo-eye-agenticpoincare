"""Hyperbolic neighbor message passing (Möbius aggregation) — Tokyo Eye v7 SSOT."""

from __future__ import annotations

import torch
import torch.nn as nn
from geoopt.manifolds.stereographic import math as pmath
from torch_geometric.nn import MessagePassing


class MobiusGraphConv(MessagePassing):
    """Aggregate neighbors in the tangent space at each node, then expmap back.

    For edge (j → i): message = logmap_{x_i}(x_j); mean-aggregate; MLP; expmap_{x_i}.
    This is primary transport for Tokyo Eye v7 — not Euclidean SE(3) on hyp edges.
    """

    def __init__(self, dim: int, *, aggr: str = "mean") -> None:
        super().__init__(aggr=aggr)
        self.lin = nn.Linear(dim, dim)
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(
        self,
        x_hyp: torch.Tensor,
        edge_index: torch.Tensor,
        k: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return x_hyp
        out = self.propagate(edge_index, x=x_hyp, k=k)
        return out

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        k: torch.Tensor,
    ) -> torch.Tensor:
        # Tangent at receiver i pointing toward sender j.
        return pmath.logmap(x_i, x_j, k=k)

    def update(
        self,
        aggr_out: torch.Tensor,
        x: torch.Tensor,
        k: torch.Tensor,
    ) -> torch.Tensor:
        h = self.lin(aggr_out) + self.bias
        return pmath.project(pmath.expmap(x, h, k=k), k=k)


class HyperbolicMessagePassingStack(nn.Module):
    """Stack of Möbius graph convs with residual in the ball via geodesic mix."""

    def __init__(self, dim: int, num_layers: int = 3) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.layers = nn.ModuleList([MobiusGraphConv(dim) for _ in range(num_layers)])

    def forward(
        self,
        x_hyp: torch.Tensor,
        edge_index: torch.Tensor,
        k: torch.Tensor,
    ) -> torch.Tensor:
        x = x_hyp
        for layer in self.layers:
            y = layer(x, edge_index, k)
            # Residual: geodesic midpoint toward update (stable hierarchy transport).
            t = torch.tensor(0.5, device=x.device, dtype=x.dtype)
            x = pmath.project(pmath.geodesic(t, x, y, k=k), k=k)
        return x


def _ca_fallback_allowed(data, allow_ca_fallback: bool | None) -> bool:
    """Resolve whether Cα ``edge_index`` may be used for Hyp MP."""
    if allow_ca_fallback is not None:
        return bool(allow_ca_fallback)
    if hasattr(data, "allow_ca_fallback"):
        return bool(getattr(data, "allow_ca_fallback"))
    if bool(getattr(data, "hyp_biology_mp", False)):
        return False
    return True


def resolve_hyp_mp_edges(
    data,
    *,
    allow_ca_fallback: bool | None = None,
) -> torch.Tensor:
    """Prefer explicit hyperbolic graph; else Cα ``edge_index``.

    When ``allow_ca_fallback`` is false (or ``data.hyp_biology_mp`` /
    ``data.allow_ca_fallback=False``), missing biology edges fail closed —
    never silently use Cα contact topology.
    """
    allow = _ca_fallback_allowed(data, allow_ca_fallback)
    if bool(getattr(data, "hyperbolic_graph", False)):
        hei = getattr(data, "hyperbolic_edge_index", None)
        if hei is not None:
            return hei
        if not allow:
            raise ValueError(
                "hyp_biology_mp requires hyperbolic_edge_index when "
                "hyperbolic_graph=True (Cα fallback forbidden)"
            )
    if not allow:
        raise ValueError(
            "Cα edge_index fallback forbidden for hyp_biology_mp "
            "(allow_ca_fallback=false)"
        )
    return data.edge_index


__all__ = [
    "HyperbolicMessagePassingStack",
    "MobiusGraphConv",
    "resolve_hyp_mp_edges",
]
