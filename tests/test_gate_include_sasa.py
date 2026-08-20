"""Explicit SASA on topology gate board (SASA enrichment pre-reg)."""

from __future__ import annotations

import torch

from science.dtie.v6.gnn.hyperbolic_moe import HyperbolicPrototypeGate
from science.dtie.v66.gnn.model import GOSPConeMapperV66, infer_v66_model_kwargs


def test_hyperbolic_gate_topo_dim_grows_with_sasa() -> None:
    g0 = HyperbolicPrototypeGate(hidden_dim=32, num_experts=4, include_sasa=False)
    g1 = HyperbolicPrototypeGate(hidden_dim=32, num_experts=4, include_sasa=True)
    assert g0.topo_dim == 8
    assert g1.topo_dim == 9
    assert g0.topo_encoder[0].in_features == 8 + g0.DISC_DIM
    assert g1.topo_encoder[0].in_features == 9 + g1.DISC_DIM


def test_gate_forward_with_sasa_running_stats() -> None:
    gate = HyperbolicPrototypeGate(
        hidden_dim=32, num_experts=4, topology_only=True, include_sasa=True
    )
    n = 17
    x_hyp = torch.randn(n, 32) * 0.01
    k = torch.tensor(-1.0)
    clustering = torch.rand(n)
    cone_depth = torch.rand(n, 1)
    degree = torch.randint(1, 8, (n,)).float()
    rho = torch.rand(n) * 20
    ss = torch.zeros(n, 3)
    ss[:, 0] = 1.0
    sasa = torch.rand(n)
    gate.train()
    scores, cap, audit = gate(
        x_hyp,
        k,
        clustering,
        cone_depth,
        degree,
        rho,
        ss,
        tau_flag=torch.zeros(n),
        sasa=sasa,
    )
    assert scores.shape == (n, 4)
    assert torch.isfinite(scores).all()
    assert float(gate.num_updates) >= 1


def test_infer_v66_kwargs_detects_sasa_board() -> None:
    model = GOSPConeMapperV66(
        node_dim=3,
        hidden=32,
        num_layers=2,
        num_experts=4,
        topology_only_gate=True,
        gate_include_sasa=True,
        hyperbolic_gate=True,
    )
    kwargs = infer_v66_model_kwargs(model.state_dict(), architecture=None, training_config=None)
    assert kwargs["gate_include_sasa"] is True
