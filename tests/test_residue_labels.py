"""Tests for ResidueStage1 residue label generation."""

from __future__ import annotations

import numpy as np
import torch

from experiments.training.v6.residue_labels import _geometric_pocket_mask


def test_geometric_pocket_prefers_buried_cluster() -> None:
    sasa = np.array([0.95, 0.90, 0.12, 0.10, 0.11, 0.88], dtype=np.float64)
    edge_index = torch.tensor(
        [[2, 3, 3, 4], [3, 2, 4, 3]],
        dtype=torch.long,
    )
    mask = _geometric_pocket_mask(sasa, edge_index)
    assert mask[3] and mask[4]
    assert not mask[0] and not mask[1] and not mask[5]
