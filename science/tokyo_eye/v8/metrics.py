"""Lightweight binary AUPRC (no sklearn dependency) for v8 validation."""

from __future__ import annotations

import torch


def binary_auprc(
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> float:
    """Macro-friendly binary AUPRC on 1-D score/label tensors."""
    s = scores.detach().float().reshape(-1)
    y = labels.detach().float().reshape(-1)
    if s.numel() == 0 or float(y.sum()) < eps:
        return float("nan")
    # Sort by score descending
    order = torch.argsort(s, descending=True)
    y_sorted = y[order]
    tp = torch.cumsum(y_sorted, dim=0)
    fp = torch.cumsum(1.0 - y_sorted, dim=0)
    precision = tp / (tp + fp).clamp_min(eps)
    recall = tp / y.sum().clamp_min(eps)
    # Trapezoid over recall
    recall_prev = torch.cat([torch.zeros(1, device=recall.device), recall[:-1]])
    auprc = torch.sum((recall - recall_prev) * precision)
    return float(auprc)
