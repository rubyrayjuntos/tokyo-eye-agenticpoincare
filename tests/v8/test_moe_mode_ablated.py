"""moe_mode=ablated — fixed expert-0; live path unchanged by default."""

from __future__ import annotations

import torch

from science.tokyo_eye.v8.moe import TopologyAwareHardMoE


def _toy_batch(dim: int = 6, n: int = 12):
    z = torch.randn(n, dim) * 0.04
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    return z, edge_index


def test_ablated_exact_load_and_zero_moe_losses() -> None:
    torch.manual_seed(0)
    moe = TopologyAwareHardMoE(dim=6, gate_hidden=4, temperature=1.0)
    moe.set_moe_mode("ablated")
    moe.train()
    z, edge_index = _toy_batch()
    out, aux = moe(z, edge_index)
    assert aux["moe_mode"] == "ablated"
    assert torch.allclose(aux["routing"].sum(-1), torch.ones(z.shape[0]))
    assert torch.equal(aux["routing"][:, 0], torch.ones(z.shape[0]))
    assert float(aux["routing"][:, 1:].sum()) == 0.0
    for key in (
        "cv_loss",
        "quota_loss",
        "majority_hinge_loss",
        "switch_lb_loss",
        "soft_quota_loss",
        "eval_proxy_lb_loss",
        "eval_proxy_quota_loss",
    ):
        assert float(aux[key]) == 0.0
    assert out.shape == z.shape
    # Gate params must not receive grads under ablated.
    loss = out.pow(2).mean()
    loss.backward()
    for p in list(moe.gate_fc1.parameters()) + list(moe.gate_fc2.parameters()):
        assert p.grad is None
        assert p.requires_grad is False


def test_ablated_experts_1_to_3_not_called() -> None:
    torch.manual_seed(1)
    moe = TopologyAwareHardMoE(dim=4, gate_hidden=4, temperature=1.0)
    moe.set_moe_mode("ablated")
    called = {i: 0 for i in range(4)}
    originals = list(moe.experts)

    class _CountingExpert(torch.nn.Module):
        def __init__(self, idx: int, inner: torch.nn.Module) -> None:
            super().__init__()
            self.idx = idx
            self.inner = inner

        def forward(self, *args, **kwargs):
            called[self.idx] += 1
            return self.inner(*args, **kwargs)

    moe.experts = torch.nn.ModuleList(
        [_CountingExpert(i, e) for i, e in enumerate(originals)]
    )
    z, edge_index = _toy_batch(dim=4)
    moe.eval()
    moe(z, edge_index)
    assert called[0] == 1
    assert called[1] == called[2] == called[3] == 0


def test_live_default_unchanged_bit_identical() -> None:
    """Default moe_mode=live must match a fresh MoE without set_moe_mode."""
    torch.manual_seed(0)
    a = TopologyAwareHardMoE(dim=6, gate_hidden=4, temperature=0.7, explore_epsilon=0.0)
    b = TopologyAwareHardMoE(dim=6, gate_hidden=4, temperature=0.7, explore_epsilon=0.0)
    b.load_state_dict(a.state_dict())
    assert a.moe_mode == "live"
    b.set_moe_mode("live")
    z, edge_index = _toy_batch()
    a.train()
    b.train()
    torch.manual_seed(99)
    out_a, aux_a = a(z, edge_index)
    torch.manual_seed(99)
    out_b, aux_b = b(z, edge_index)
    assert torch.equal(aux_a["routing"], aux_b["routing"])
    assert torch.allclose(out_a, out_b)
    assert aux_b["moe_mode"] == "live"
