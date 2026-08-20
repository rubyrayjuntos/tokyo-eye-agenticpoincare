"""Phase A nucleotide basin contrastive helpers (hyperbolic)."""

from __future__ import annotations

from typing import Sequence

import torch
from geoopt.manifolds.stereographic import math as pmath


def _as_k(c: float | torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(c, torch.Tensor):
        k = c.to(device=device, dtype=dtype)
    else:
        k = torch.tensor(float(c), device=device, dtype=dtype)
    if float(k.reshape(-1)[0].item()) > 0:
        k = -k.abs()
    return k


def r_star_indices(resseqs_present: set[int], n12: Sequence[int] | None = None) -> list[int]:
    """Switch I∪II ∪ N12 auth_seq list present in the graph."""
    switch = set(range(25, 41)) | set(range(57, 76))
    extra = set(int(r) for r in (n12 or []))
    return sorted((switch | extra) & resseqs_present)


def structure_embedding_logmap0(
    x_hyp: torch.Tensor,
    graph_indices: Sequence[int],
    *,
    curvature: float | torch.Tensor,
) -> torch.Tensor:
    """Mean tangent at origin over selected residue rows."""
    if not graph_indices:
        idx = torch.arange(x_hyp.shape[0], device=x_hyp.device)
    else:
        idx = torch.tensor(list(graph_indices), device=x_hyp.device, dtype=torch.long)
    k = _as_k(curvature, device=x_hyp.device, dtype=x_hyp.dtype)
    tang = pmath.logmap0(x_hyp[idx], k=k)
    return tang.mean(dim=0)


def basin_contrastive_margin_loss(
    z_off: torch.Tensor,
    z_on: torch.Tensor,
    *,
    curvature: float | torch.Tensor,
    margin: float = 0.50,
) -> torch.Tensor:
    """ReLU(m - d_B(exp0(z_off), exp0(z_on)))^2."""
    k = _as_k(curvature, device=z_off.device, dtype=z_off.dtype)
    u = pmath.project(pmath.expmap0(z_off, k=k), k=k)
    v = pmath.project(pmath.expmap0(z_on, k=k), k=k)
    d = pmath.dist(u.unsqueeze(0), v.unsqueeze(0), k=k).reshape(())
    gap = torch.relu(torch.as_tensor(margin, device=d.device, dtype=d.dtype) - d)
    return gap * gap


def ball_distance_embeddings(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    *,
    curvature: float | torch.Tensor,
) -> float:
    k = _as_k(curvature, device=z_a.device, dtype=z_a.dtype)
    u = pmath.project(pmath.expmap0(z_a, k=k), k=k)
    v = pmath.project(pmath.expmap0(z_b, k=k), k=k)
    return float(pmath.dist(u.unsqueeze(0), v.unsqueeze(0), k=k).reshape(()).item())


__all__ = [
    "ball_distance_embeddings",
    "basin_contrastive_margin_loss",
    "r_star_indices",
    "structure_embedding_logmap0",
]
