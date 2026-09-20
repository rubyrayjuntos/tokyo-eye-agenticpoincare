"""§2.6 epsilon-greedy hard override — mutation fixtures + schedule."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from science.tokyo_eye.v8.engine import EpsilonGreedySchedule
from science.tokyo_eye.v8.moe import (
    TopologyAwareHardMoE,
    epsilon_greedy_hard_override,
)


def test_epsilon_schedule_decays_to_zero() -> None:
    sched = EpsilonGreedySchedule(0.20, 0.0, half_epochs=12)
    assert abs(sched.epsilon(0) - 0.20) < 1e-9
    assert sched.epsilon(12) < 0.20
    # By ~2*half should be at floor
    assert sched.epsilon(40) == 0.0


def test_override_is_discrete_one_hot() -> None:
    torch.manual_seed(0)
    routing = F.one_hot(torch.zeros(32, dtype=torch.long), 4).float()
    out, mask = epsilon_greedy_hard_override(routing, epsilon=1.0)
    assert torch.allclose(out.sum(-1), torch.ones(32), atol=1e-5)
    assert ((out - out.round()).abs() < 1e-5).all()
    assert float(mask.mean()) == 1.0
    # Not all still on expert 0 when eps=1
    assert out[:, 0].mean() < 0.9


def test_eps0_inert_matches_baseline_routing() -> None:
    """Known-bad fixture: eps=0 must be byte-identical to no-override path."""
    torch.manual_seed(0)
    moe_base = TopologyAwareHardMoE(dim=6, gate_hidden=4, temperature=1.0, explore_epsilon=0.0)
    moe_eps = TopologyAwareHardMoE(dim=6, gate_hidden=4, temperature=1.0, explore_epsilon=0.0)
    moe_eps.load_state_dict(moe_base.state_dict())
    moe_base.train()
    moe_eps.train()
    z = torch.randn(16, 6) * 0.04
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    torch.manual_seed(123)
    _, aux0 = moe_base(z, edge_index)
    torch.manual_seed(123)
    _, aux1 = moe_eps(z, edge_index)
    assert torch.equal(aux0["routing"], aux1["routing"])
    assert aux1["explore_epsilon"] == 0.0
    assert aux1["explore_override_frac"] == 0.0


def test_eps1_forced_uniform_not_gate_argmax() -> None:
    """Known-good fixture: eps=1.0 executes override (not a no-op)."""
    torch.manual_seed(0)
    moe = TopologyAwareHardMoE(dim=6, gate_hidden=4, temperature=0.05, explore_epsilon=1.0)
    moe.train()
    # Strongly biased logits toward E0 via gate init — still must randomize at eps=1
    n = 200
    z = torch.randn(n, 6) * 0.03
    edge_index = torch.zeros(2, 0, dtype=torch.long)
    loads = []
    for seed in range(5):
        torch.manual_seed(seed)
        _, aux = moe(z, edge_index)
        loads.append(aux["routing"].mean(0))
        assert abs(aux["explore_override_frac"] - 1.0) < 1e-5
        assert torch.allclose(aux["routing"].sum(-1), torch.ones(n), atol=1e-5)
    mean_load = torch.stack(loads).mean(0).detach()
    # Uniform within sampling tolerance — not monopoly on argmax expert
    assert float(mean_load.max()) < 0.45
    assert float(mean_load.min()) > 0.15


def test_eval_ignores_epsilon() -> None:
    torch.manual_seed(0)
    moe = TopologyAwareHardMoE(dim=4, gate_hidden=4, temperature=1.0, explore_epsilon=1.0)
    moe.eval()
    z = torch.randn(8, 4) * 0.03
    edge_index = torch.zeros(2, 0, dtype=torch.long)
    _, aux = moe(z, edge_index)
    # Eval is pure argmax — override frac 0 even if explore_epsilon=1
    assert aux["explore_override_frac"] == 0.0
    assert aux["hard"] == "argmax"
