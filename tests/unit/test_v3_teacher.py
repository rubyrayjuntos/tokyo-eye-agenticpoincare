"""Tests for v3 teacher import and v2 distill loss."""

from __future__ import annotations

import torch

from science.dtie.v3.gnn.model import GOSPConeMapper, precompute_clustering
from science.dtie.v6.loss import v2_teacher_distill_loss


def test_v3_teacher_model_instantiates() -> None:
    model = GOSPConeMapper(node_dim=4, hidden=64, num_layers=2, num_experts=4)
    assert sum(p.numel() for p in model.parameters()) > 0


def test_v2_teacher_distill_loss_shapes() -> None:
    n = 12
    output = {
        "cone_depth": torch.linspace(0.1, 0.9, n).unsqueeze(1),
        "uncertainty": {"epistemic": torch.linspace(0.2, 1.0, n).unsqueeze(1)},
    }
    teacher = {
        "cone_depth_norm": torch.linspace(0.0, 1.0, n),
        "epistemic": torch.linspace(0.1, 0.8, n),
        "expert_id": torch.zeros(n),
    }
    sasa = torch.linspace(0.0, 1.0, n)
    losses = v2_teacher_distill_loss(output, teacher, sasa)
    assert losses["v2_teacher_total"].ndim == 0
