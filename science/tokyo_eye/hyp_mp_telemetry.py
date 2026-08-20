"""Cheap Hyp-MP graph telemetry — one forward, no knockout.

Edge mass = Poincaré geodesic length between endpoints.
Node strength = sum of incident edge masses → top-k% hub proxy.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from geoopt.manifolds.stereographic import math as pmath

from science.tokyo_eye.hyperbolic_mp import resolve_hyp_mp_edges


def _as_k(c: float | torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(c, torch.Tensor):
        k = c.to(device=device, dtype=dtype)
    else:
        k = torch.tensor(float(c), device=device, dtype=dtype)
    if float(k.reshape(-1)[0].item()) > 0:
        k = -k.abs()
    return k


def hyp_mp_edge_node_telemetry(
    x_hyp: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    curvature: float | torch.Tensor,
    k_frac: float = 0.10,
) -> dict[str, Any]:
    """Compute edge geodesic lengths and node strengths on the Hyp-MP graph."""
    n = int(x_hyp.shape[0])
    k = _as_k(curvature, device=x_hyp.device, dtype=x_hyp.dtype)
    if edge_index.numel() == 0:
        strengths = torch.zeros(n, device=x_hyp.device, dtype=x_hyp.dtype)
        return {
            "n_nodes": n,
            "n_edges": 0,
            "edge_index": edge_index.detach().cpu(),
            "edge_length": torch.zeros(0, dtype=x_hyp.dtype),
            "node_strength": strengths.detach().cpu(),
            "hub_indices": [],
            "hub_strengths": [],
            "k_frac": float(k_frac),
            "n_hubs": 0,
        }

    src = edge_index[0].long()
    dst = edge_index[1].long()
    # Undirected mass: average both directions if present; compute per directed edge.
    lengths = pmath.dist(x_hyp[src], x_hyp[dst], k=k).reshape(-1)
    strengths = torch.zeros(n, device=x_hyp.device, dtype=x_hyp.dtype)
    strengths.index_add_(0, src, lengths)
    strengths.index_add_(0, dst, lengths)

    top_k = max(1, int(np.ceil(float(k_frac) * n)))
    order = torch.argsort(strengths, descending=True)
    hub_idx = order[:top_k]
    return {
        "n_nodes": n,
        "n_edges": int(src.numel()),
        "edge_index": edge_index.detach().cpu(),
        "edge_length": lengths.detach().cpu(),
        "node_strength": strengths.detach().cpu(),
        "hub_indices": [int(i) for i in hub_idx.cpu().tolist()],
        "hub_strengths": [float(strengths[i].item()) for i in hub_idx],
        "k_frac": float(k_frac),
        "n_hubs": int(top_k),
        "strength_cv": float(
            (strengths.std() / (strengths.mean() + 1e-12)).item()
        ),
    }


def attach_hyp_mp_telemetry(
    output: dict[str, Any],
    data: Any,
    *,
    k_frac: float = 0.10,
) -> dict[str, Any]:
    """Add ``hyp_mp_telemetry`` onto a TokyoEye forward output dict (in-place)."""
    x_hyp = output.get("x_hyp")
    if x_hyp is None or not torch.is_tensor(x_hyp):
        output["hyp_mp_telemetry"] = {"error": "missing_x_hyp"}
        return output
    c = (output.get("audit_trail") or {}).get("curvature_value")
    if c is None:
        output["hyp_mp_telemetry"] = {"error": "missing_curvature"}
        return output
    ei = resolve_hyp_mp_edges(data)
    n = int(x_hyp.shape[0])
    # Leaf rows only if parents appended
    if hasattr(data, "x") and data.x is not None and int(data.x.shape[0]) < n:
        n = int(data.x.shape[0])
        x_hyp = x_hyp[:n]
    tel = hyp_mp_edge_node_telemetry(x_hyp, ei, curvature=c, k_frac=k_frac)
    output["hyp_mp_telemetry"] = tel
    return output


__all__ = [
    "attach_hyp_mp_telemetry",
    "hyp_mp_edge_node_telemetry",
]
