"""Tests for per-residue uncertainty extraction and sanity audit."""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from science.training.uncertainty_diagnostics import (
    audit_uncertainty_sanity,
    extract_residue_uncertainty_rows,
)


def test_extract_and_audit_uncertainty_sanity() -> None:
    n = 20
    epi = np.linspace(0.1, 2.0, n)
    ale = np.linspace(0.5, 3.0, n)
    prot = {
        "residue_ids": [f"A:{i}:" for i in range(1, n + 1)],
        "data": Data(x=torch.zeros(n, 4)),
    }
    prot["data"][:, 0] = torch.linspace(8.0, 18.0, n)
    out = {
        "uncertainty": {
            "epistemic": torch.tensor(epi, dtype=torch.float32).unsqueeze(1),
            "aleatoric": torch.tensor(ale, dtype=torch.float32).unsqueeze(1),
            "total": torch.tensor(epi + ale, dtype=torch.float32).unsqueeze(1),
        },
        "cone_depth": torch.linspace(0.2, 1.0, n).unsqueeze(1),
        "expert_weights": torch.nn.functional.one_hot(
            torch.zeros(n, dtype=torch.long), num_classes=4
        ).float(),
        "evidence": {
            "nu": torch.linspace(0.5, 2.0, n).unsqueeze(1),
            "alpha": torch.ones(n, 1) * 2.0,
            "beta": torch.ones(n, 1),
            "mu": torch.zeros(n, 1),
        },
    }
    rows = extract_residue_uncertainty_rows(out, prot)
    assert len(rows) == n
    assert rows[0]["total_matches_sum"]
    sanity = audit_uncertainty_sanity(rows)
    assert sanity["ok"]
    assert sanity["epistemic_std"] > 0.01
    assert sanity["aleatoric_std"] > 0.01
    assert abs(sanity["r_epi_ale"] - 1.0) < 0.01
