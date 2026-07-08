"""Topology depth lineage — no SASA in loss gates or MLflow mandatory metrics."""

from __future__ import annotations

import torch

from science.dtie.v6.loss import shell_correlation_topology_loss
from science.training.topology_depth import topology_depth_lineage


def test_topology_depth_lineage_master_cold() -> None:
    assert topology_depth_lineage(master_cold=True) is True


def test_shell_correlation_topology_no_sasa_grad() -> None:
    depth = torch.tensor([0.2, 0.5, 0.9, 1.1], requires_grad=True)
    hyp = torch.tensor(
        [[0.1, 0.0], [0.3, 0.1], [0.5, 0.2], [0.7, 0.3]],
        requires_grad=True,
    )
    out = shell_correlation_topology_loss(depth, hyp)
    loss = out["shell_corr_total"]
    loss.backward()
    assert float(out["shell_corr_depth_sasa"].detach()) == 0.0
    assert float(out["shell_corr_disc_sasa"].detach()) == 0.0
    assert depth.grad is not None
