"""Per-expert biology role-card metrics from dominant routing."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data

from experiments.training.v6.train_loop import _accumulate_expert_geometry


def test_accumulate_expert_biology_role_card() -> None:
    n = 12
    num_experts = 2
    x = torch.zeros(n, 4)
    # expert 0: buried low-tau; expert 1: surface high-tau coil
    x[:6, 0] = 20.0  # rho
    x[6:, 0] = 5.0
    x[:6, 1] = 0.0  # tau
    x[6:, 1] = 1.0
    x[:6, 2] = 0.0  # helix
    x[6:, 2] = 1.0  # coil
    x[:6, 3] = 0.2  # sasa
    x[6:, 3] = 0.8
    data = Data(x=x, edge_index=torch.tensor([[0, 1], [1, 0]]))
    out = {
        "cone_depth": torch.tensor([[0.2], [0.2], [0.2], [0.2], [0.2], [0.2],
                                    [0.9], [0.9], [0.9], [0.9], [0.9], [0.9]]),
        "hyp_projections_2d": torch.tensor([[0.1, 0.0]] * 6 + [[0.5, 0.0]] * 6, dtype=torch.float32),
        "expert_weights": torch.tensor(
            [[0.9, 0.1]] * 6 + [[0.1, 0.9]] * 6, dtype=torch.float32
        ),
    }
    depth_sums = [0.0, 0.0]
    depth_counts = [0, 0]
    disc_sums = [0.0, 0.0]
    disc_counts = [0, 0]
    r_ds_pairs = [([], []), ([], [])]
    tau_sums = [0.0, 0.0]
    sasa_sums = [0.0, 0.0]
    rho_sums = [0.0, 0.0]
    coil_counts = [0, 0]
    bio_counts = [0, 0]
    _accumulate_expert_geometry(
        out,
        data,
        num_experts=num_experts,
        depth_sums=depth_sums,
        depth_counts=depth_counts,
        disc_sums=disc_sums,
        disc_counts=disc_counts,
        r_ds_pairs=r_ds_pairs,
        tau_sums=tau_sums,
        sasa_sums=sasa_sums,
        rho_sums=rho_sums,
        coil_counts=coil_counts,
        bio_counts=bio_counts,
    )
    assert depth_counts == [6, 6]
    assert tau_sums[0] == 0.0 and tau_sums[1] == 6.0
    assert sasa_sums[0] / bio_counts[0] == pytest.approx(0.2)
    assert sasa_sums[1] / bio_counts[1] == pytest.approx(0.8)
    assert coil_counts[0] == 0 and coil_counts[1] == 6
    assert rho_sums[0] / bio_counts[0] == 20.0
    assert rho_sums[1] / bio_counts[1] == 5.0
