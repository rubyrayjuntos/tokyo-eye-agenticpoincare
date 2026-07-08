"""Tests for slim MoE + frozen structural disc SSOT training preset."""

from __future__ import annotations

import torch
from torch_geometric.data import Data

from experiments.training.v6.train_loop import (
    prepare_training_batch,
    set_slim_moe_structural_ssot_freeze,
)
from science.dtie.v5.gnn.model import precompute_clustering
from science.dtie.v6.gnn.model import GOSPConeMapperV6
from science.training.config import (
    TrainingConfig,
    apply_slim_moe_structural_ssot_config,
    apply_slim_moe_structural_ssot_lineage,
    apply_slim_moe_structural_ssot_phases,
    default_v6_phases,
)


def test_apply_slim_moe_phases_zeros_disc_geometry_losses() -> None:
    phases = apply_slim_moe_structural_ssot_phases(default_v6_phases())
    assert all(p.slim_moe_structural_ssot_train for p in phases)
    assert all(p.freeze_radial and p.freeze_angular for p in phases)
    for phase in phases:
        c = phase.coeffs
        assert c.disc_occupancy_coeff == 0.0
        assert c.disc_eff_rank_coeff == 0.0
        assert c.disc_path_align_coeff == 0.0
        assert c.angular_coeff == 0.0
        assert c.cone_target_mode == "tau_dehydron_rim"
    p2 = next(p for p in phases if p.phase == 2)
    assert p2.coeffs.routing_load_floor_coeff >= 0.12


def test_apply_slim_moe_config_sets_structural_frozen_and_topology_gate() -> None:
    base = TrainingConfig(corpus_manifest="manifests/v6_corpus_stage_a_small_v1.json")
    cfg = apply_slim_moe_structural_ssot_config(base)
    assert cfg.structural_disc_frozen is True
    assert cfg.slim_moe_structural_ssot is True
    assert cfg.topology_only_gate is True
    assert cfg.structure_gate is True
    assert cfg.expert_depth_decouple is True
    assert cfg.v2_teacher_checkpoint is None
    assert cfg.phase_preset_name() == "slim_moe_structural_ssot"


def test_apply_slim_moe_lineage_pairs_config_and_phases() -> None:
    cfg, phases = apply_slim_moe_structural_ssot_lineage(
        TrainingConfig(corpus_manifest="manifests/v6_corpus_stage_a_small_v1.json"),
        default_v6_phases(),
    )
    assert cfg.structural_disc_frozen
    assert phases[0].slim_moe_structural_ssot_train


def test_set_slim_moe_freeze_geometry_heads_only() -> None:
    model = GOSPConeMapperV6(
        hidden=16,
        num_layers=2,
        num_experts=2,
        hyperbolic_gate=False,
        topology_only_gate=True,
    )
    set_slim_moe_structural_ssot_freeze(model)
    frozen_prefixes = (
        "radial_head.",
        "angular_head.",
        "hyp_proj_head_2d.",
        "hyp_proj_head_3d.",
    )
    for name, param in model.named_parameters():
        if name.startswith(frozen_prefixes) or name == "expert_depth_bias":
            assert not param.requires_grad, name
        elif name.startswith("gate."):
            assert param.requires_grad, name
        elif name.startswith("experts."):
            assert param.requires_grad, name


def test_prepare_training_batch_attaches_structural_disc() -> None:
    model = GOSPConeMapperV6(hidden=16, num_layers=2, num_experts=2, hyperbolic_gate=False)
    n = 10
    x = torch.randn(n, 4)
    x[:, 0] = torch.linspace(8.0, 20.0, n)
    x[:, 1] = (x[:, 0] < 13.0).float()
    edge_index = torch.tensor([[i, (i + 1) % n] for i in range(n)], dtype=torch.long).T
    prot = {
        "pdb_id": "1crn",
        "chain": "A",
        "data": Data(
            x=x,
            edge_index=edge_index,
            edge_attr=torch.randn(edge_index.size(1), 4).abs(),
        ),
        "ca_coords": torch.randn(n, 3),
        "residue_ids": [f"A:{i}:" for i in range(1, n + 1)],
    }
    data = prepare_training_batch(
        model,
        prot,
        "cpu",
        structural_disc_frozen=True,
    )
    data = precompute_clustering(data)
    out = model(data)
    assert data.structural_z_disc_frozen is True
    assert out["audit_trail"].get("structural_disc_frozen") is True
