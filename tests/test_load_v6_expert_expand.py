"""Resume tolerance when widening MoE expert count (e.g. 4 → 6)."""

from __future__ import annotations

import torch

from science.dtie.v6.gnn.model import GOSPConeMapperV6, load_v6_state_dict


def test_load_v6_expands_four_expert_checkpoint_to_six() -> None:
    small = GOSPConeMapperV6(
        node_dim=3,
        hidden=32,
        num_layers=2,
        num_experts=4,
        hyperbolic_gate=True,
        topology_only_gate=True,
    )
    large = GOSPConeMapperV6(
        node_dim=3,
        hidden=32,
        num_layers=2,
        num_experts=6,
        hyperbolic_gate=True,
        topology_only_gate=True,
        expert_depth_decouple=True,
        structure_gate=True,
        gate_gumbel=True,
    )
    ckpt = small.state_dict()

    missing, unexpected = load_v6_state_dict(large, ckpt)

    assert not unexpected
    assert "expert_depth_bias" in missing
    assert "gate.structure_proj.weight" in missing

    assert torch.allclose(
        large.gate.expert_bias[:4].detach(),
        small.gate.expert_bias.detach(),
    )
    assert torch.allclose(
        large.gate.prototype_bank.prototype_tangent[:4].detach(),
        small.gate.prototype_bank.prototype_tangent.detach(),
    )
    for i in range(4):
        assert torch.allclose(
            large.experts[i][0].weight.detach(),
            small.experts[i][0].weight.detach(),
        )


def test_load_v6_expands_node_embedding_width_for_barcode() -> None:
    small = GOSPConeMapperV6(
        node_dim=3,
        hidden=32,
        num_layers=2,
        num_experts=4,
        hyperbolic_gate=True,
        topology_only_gate=True,
    )
    barcode = GOSPConeMapperV6(
        node_dim=15,
        hidden=32,
        num_layers=2,
        num_experts=4,
        hyperbolic_gate=True,
        topology_only_gate=True,
    )
    ckpt = small.state_dict()

    missing, unexpected = load_v6_state_dict(barcode, ckpt)

    assert not missing
    assert not unexpected
    assert torch.allclose(
        barcode.node_emb.weight[:, :3].detach(),
        small.node_emb.weight.detach(),
    )
    assert torch.allclose(
        barcode.node_emb.weight[:, 3:].detach(),
        torch.zeros_like(barcode.node_emb.weight[:, 3:]),
    )
