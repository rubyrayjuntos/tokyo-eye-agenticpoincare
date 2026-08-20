"""Checkpoint width adaptation helpers for v6.6 (standalone fork)."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def adapt_checkpoint_node_emb_width(
    state_dict: dict[str, torch.Tensor],
    model_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Resize node_emb.weight input columns for barcode warm-start/resume."""
    key = "node_emb.weight"
    old_w = state_dict.get(key)
    new_w = model_state.get(key)
    if old_w is None or new_w is None or old_w.shape == new_w.shape:
        return dict(state_dict)
    if old_w.ndim != 2 or new_w.ndim != 2 or old_w.shape[0] != new_w.shape[0]:
        return dict(state_dict)

    adapted = dict(state_dict)
    resized = torch.zeros_like(new_w)
    cols = min(old_w.shape[1], new_w.shape[1])
    resized[:, :cols] = old_w[:, :cols].to(dtype=resized.dtype, device=resized.device)
    adapted[key] = resized
    if old_w.shape[1] < new_w.shape[1]:
        logger.info(
            "Expanded node_emb.weight input width %d -> %d; copied %d columns and zero-filled %d new barcode columns",
            old_w.shape[1],
            new_w.shape[1],
            cols,
            new_w.shape[1] - old_w.shape[1],
        )
    else:
        logger.info(
            "Shrank node_emb.weight input width %d -> %d; copied overlapping %d columns",
            old_w.shape[1],
            new_w.shape[1],
            cols,
        )
    return adapted
