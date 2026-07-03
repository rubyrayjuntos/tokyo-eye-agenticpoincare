"""Poincaré disc projection softness and pre-routing alignment."""

from __future__ import annotations

import torch

from science.dtie.v6.gnn.hyperbolic_moe import project_disc_2d


def test_softer_disc_projection_preserves_higher_radius() -> None:
    k = torch.tensor(-1.0)
    raw = torch.tensor([[0.92, 0.0], [0.0, 0.92], [-0.88, 0.2]])
    hard, _ = project_disc_2d(raw, k=k, softness=0.0)
    soft, r_soft = project_disc_2d(raw, k=k, softness=0.95)
    assert r_soft.mean() >= hard.norm(dim=-1).mean() - 1e-5


def test_legacy_disc_projection_infer_defaults_true() -> None:
    from science.dtie.v6.gnn.model import infer_legacy_disc_projection_from_checkpoint

    assert infer_legacy_disc_projection_from_checkpoint(training_config={}) is True
    assert (
        infer_legacy_disc_projection_from_checkpoint(
            training_config={"legacy_disc_projection": False}
        )
        is False
    )
    assert (
        infer_legacy_disc_projection_from_checkpoint(
            training_config={"disc_projection_source": "pre_routing_x_hyp"}
        )
        is False
    )
