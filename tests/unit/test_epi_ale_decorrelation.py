"""Tests for epi/ale decorrelation loss and decoupled evidential head."""

from __future__ import annotations

import torch

from science.dtie.v6.gnn.evidential import (
    DecoupledEvidentialHead,
    expand_coupled_uncertainty_state_dict,
)
from science.dtie.v6.loss import epi_ale_decorrelation_loss, gosp_loss_v6


def test_epi_ale_decorrelation_penalizes_identical_signals() -> None:
    n = 64
    base = torch.linspace(0.1, 1.0, n)
    coupled = epi_ale_decorrelation_loss(base.unsqueeze(1), (base * 2 + 0.01).unsqueeze(1))
    decoupled = epi_ale_decorrelation_loss(
        base.unsqueeze(1),
        torch.randn(n, 1),
    )
    assert coupled["epi_ale_decorrelation"].item() > decoupled["epi_ale_decorrelation"].item()
    assert coupled["r_epi_ale"].item() > 0.99


def test_expand_coupled_uncertainty_state_dict_duplicates_shared_trunk() -> None:
    head = DecoupledEvidentialHead(hidden_dim=8, extra_input_dim=2)
    coupled = {}
    for key, tensor in head.state_dict().items():
        if key.startswith("epi_trunk.0."):
            coupled[f"uncertainty_head.{key.replace('epi_trunk', 'shared')}"] = tensor.clone()
        elif key.startswith("epi_trunk.2."):
            coupled[f"uncertainty_head.{key.replace('epi_trunk', 'shared')}"] = tensor.clone()
        elif key.startswith("logv_head.") or key.startswith("mu_head."):
            coupled[f"uncertainty_head.{key}"] = tensor.clone()
    expanded = expand_coupled_uncertainty_state_dict(coupled)
    assert "uncertainty_head.epi_trunk.0.weight" in expanded
    assert "uncertainty_head.ale_trunk.0.weight" in expanded
    assert torch.allclose(
        expanded["uncertainty_head.epi_trunk.0.weight"],
        expanded["uncertainty_head.ale_trunk.0.weight"],
    )


def test_gosp_loss_v6_epi_ale_decorrelation_term() -> None:
    n = 32
    device = torch.device("cpu")
    epi = torch.linspace(0.2, 1.0, n, device=device).unsqueeze(1)
    ale = epi * 1.5 + 0.02
    output = {
        "x_hyp": torch.randn(n, 2, device=device),
        "x_routed_hyp": torch.randn(n, 2, device=device),
        "radial_features": torch.linspace(0.1, 0.9, n, device=device).unsqueeze(1),
        "cone_depth": torch.linspace(0.3, 1.2, n, device=device).unsqueeze(1),
        "hyp_projections_2d": torch.randn(n, 2, device=device) * 0.25,
        "hyp_projections_3d": torch.randn(n, 3, device=device) * 0.3,
        "capacity_loss": torch.tensor(0.1, device=device),
        "routing_entropy": torch.tensor(1.0, device=device),
        "expert_load": torch.ones(4, device=device) * 0.25,
        "evidence": {
            "mu": torch.zeros(n, device=device),
            "nu": torch.ones(n, device=device),
            "alpha": torch.ones(n, device=device) * 2.0,
            "beta": torch.ones(n, device=device),
        },
        "uncertainty": {"epistemic": epi, "aleatoric": ale},
        "audit_trail": {"curvature_value": -1.0},
    }
    with_pen = gosp_loss_v6(
        output,
        torch.linspace(1.0, 20.0, n, device=device),
        torch.randn(n, 3, device=device),
        epi_ale_decorrelation_coeff=1.0,
    )
    without = gosp_loss_v6(
        output,
        torch.linspace(1.0, 20.0, n, device=device),
        torch.randn(n, 3, device=device),
        epi_ale_decorrelation_coeff=0.0,
    )
    assert with_pen["epi_ale_decorrelation"].item() > 0.0
    assert with_pen["total"].item() > without["total"].item()
