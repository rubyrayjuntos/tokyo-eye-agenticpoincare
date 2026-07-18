"""Parent-slice T1c contract for containment oversmoothing-at-root (Task 8)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.diagnostics.t1c_containment_parent_oversmooth import (
    parent_slice_embeddings,
)
from experiments.diagnostics.t1c_mp_relative_loss_null import _rel_drop


def test_parent_slice_excludes_residues() -> None:
    """n_residue_nodes=4, n_parent=2 → parent indices 4,5 only."""
    n_residue_nodes = 4
    n_parent = 2
    hidden = 8
    full = np.arange((n_residue_nodes + n_parent) * hidden, dtype=np.float64).reshape(
        n_residue_nodes + n_parent, hidden
    )
    parent = parent_slice_embeddings(full, n_residue_nodes)
    assert parent.shape == (n_parent, hidden)
    assert np.array_equal(parent, full[4:6])
    assert not np.array_equal(parent, full[:4])

    # Torch path
    t_full = torch.arange(
        (n_residue_nodes + n_parent) * hidden, dtype=torch.float32
    ).reshape(n_residue_nodes + n_parent, hidden)
    t_parent = parent_slice_embeddings(t_full, n_residue_nodes)
    assert t_parent.shape == (n_parent, hidden)
    assert np.array_equal(t_parent, full[4:6])


def test_parent_slice_empty_when_no_parents() -> None:
    emb = np.ones((4, 3))
    sliced = parent_slice_embeddings(emb, n_residue_nodes=4)
    assert sliced.shape == (0, 3)


def test_rel_drop_reused_from_t1c() -> None:
    assert _rel_drop(10.0, 8.0) == pytest.approx(0.2)
    assert np.isnan(_rel_drop(0.0, 1.0))
