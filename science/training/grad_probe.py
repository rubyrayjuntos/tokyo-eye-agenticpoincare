"""Per-subsystem gradient norm telemetry for v6 training."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn


def _mean_param_grad_norm(params) -> float:
    norms = [
        p.grad.norm().item()
        for p in params
        if p.grad is not None and torch.isfinite(p.grad).all()
    ]
    norms = [n for n in norms if np.isfinite(n)]
    if not norms:
        return 0.0
    return float(np.mean(norms))


def collect_subsystem_grad_norms(model: nn.Module) -> dict[str, float]:
    """Collect grad norms after ``backward()`` and before ``optimizer.step()``."""
    return {
        "grad_radial": _mean_param_grad_norm(model.radial_head.parameters()),
        "grad_angular": _mean_param_grad_norm(model.angular_head.parameters()),
        "grad_backbone": _mean_param_grad_norm(model.convs.parameters()),
    }


def append_subsystem_grad_norms(model: nn.Module, accum: dict[str, list[float]]) -> None:
    """Append one protein-step grad norms into epoch accumulators."""
    for key, value in collect_subsystem_grad_norms(model).items():
        accum[key].append(value)


def finalize_subsystem_grad_norms(accum: dict[str, list[float]]) -> dict[str, float]:
    """Mean per-subsystem grad norms for one training epoch."""
    out: dict[str, float] = {}
    for key in ("grad_radial", "grad_angular", "grad_backbone"):
        vals = [v for v in accum.get(key, []) if np.isfinite(v)]
        out[key] = float(np.mean(vals)) if vals else 0.0
    return out


def grad_probe_is_live(metrics: dict[str, Any], *, subsystem: str) -> bool:
    """True when subsystem grad norm is finite and strictly positive."""
    key = f"grad_{subsystem}"
    raw = metrics.get(key)
    if raw is None:
        return False
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return False
    return np.isfinite(value) and value > 0.0
