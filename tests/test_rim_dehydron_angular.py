"""Tests for within-rim dehydron angular metrics and dehydron decoupling."""

from __future__ import annotations

import numpy as np
import torch

from science.dtie.v66.gnn.equivariant_conv_multirel import EquivariantConvMultiRel
from science.dtie.v66.role_edge_graph import (
    EDGE_ATTR_ROLE_DIM,
    GEO_DIM,
    ROLE_DEHYDRON,
    ROLE_PACKING,
    compute_role_edge_graph,
)
from science.dtie.v66.rim_dehydron_angular import (
    circular_delta_deg,
    compute_rim_dehydron_angular_stats,
    max_pairwise_circular_delta_deg,
)
from science.training.config import (
    v66_feeler_dehydron_angular_phase_config,
    v66_feeler_no_exclusivity_phase_config,
)


def test_circular_delta_wraps() -> None:
    assert circular_delta_deg(170.0, -170.0) == 20.0
    assert circular_delta_deg(10.0, 20.0) == 10.0


def test_max_pairwise_circular_delta() -> None:
    theta = np.array([0.0, 90.0, 180.0])
    assert max_pairwise_circular_delta_deg(theta) == 180.0


def test_rim_dehydron_angular_stats_spread() -> None:
    n = 8
    r = np.array([0.1, 0.2, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    theta = np.array([-80.0, -70.0, 10.0, 30.0, 50.0, 120.0, 140.0, 160.0])
    xy = np.stack([r * np.cos(np.radians(theta)), r * np.sin(np.radians(theta))], axis=1)
    rho = np.array([20.0, 20.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0])
    dehyd = rho < 13.0
    stats = compute_rim_dehydron_angular_stats(
        structure_id="TEST",
        disc_r=r,
        disc_theta_deg=theta,
        disc_xy=xy,
        rho=rho,
        dehydron=dehyd,
    )
    assert stats.n_rim_dehydron >= 2
    assert stats.within_rim_angular_std_deg > 10.0
    assert stats.max_pairwise_delta_theta_deg > 30.0
    assert stats.rim_xy_effective_rank > 1.0


def test_no_exclusivity_adds_packing_on_dehydron_pairs() -> None:
    n = 10
    coords = np.zeros((n, 3), dtype=np.float64)
    coords[:, 0] = np.arange(n) * 3.8
    rho = np.array([20.0] * 5 + [8.0] * 5, dtype=np.float64)
    tau = np.array([0.0] * 5 + [1.0] * 5, dtype=np.float64)
    rids = [f"A:{i + 1}:" for i in range(n)]
    ei0, ea0 = compute_role_edge_graph(
        coords,
        rho,
        tau_flag=tau,
        residue_ids=rids,
        dehydron_exclusivity=True,
    )
    ei1, ea1 = compute_role_edge_graph(
        coords,
        rho,
        tau_flag=tau,
        residue_ids=rids,
        dehydron_exclusivity=False,
    )
    dehyd0 = int(ea0[:, GEO_DIM + ROLE_DEHYDRON].sum())
    dehyd1 = int(ea1[:, GEO_DIM + ROLE_DEHYDRON].sum())
    pack0 = int(ea0[:, GEO_DIM + ROLE_PACKING].sum())
    pack1 = int(ea1[:, GEO_DIM + ROLE_PACKING].sum())
    assert dehyd1 == dehyd0
    assert pack1 >= pack0


def test_dehydron_angular_scale_weakens_messages() -> None:
    torch.manual_seed(0)
    conv_full = EquivariantConvMultiRel(
        32,
        num_relations=4,
        onehot_offset=GEO_DIM,
        dehydron_relation_id=ROLE_DEHYDRON,
        dehydron_angular_scale=1.0,
    )
    conv_weak = EquivariantConvMultiRel(
        32,
        num_relations=4,
        onehot_offset=GEO_DIM,
        dehydron_relation_id=ROLE_DEHYDRON,
        dehydron_angular_scale=0.3,
    )
    conv_weak.load_state_dict(conv_full.state_dict())
    n, e = 6, 8
    x = torch.randn(n, 64)
    edge_index = torch.randint(0, n, (2, e))
    ea = torch.zeros(e, EDGE_ATTR_ROLE_DIM)
    angles = torch.linspace(0, 2 * 3.14159, e)
    ea[:, 0] = torch.cos(angles)
    ea[:, 1] = torch.sin(angles)
    ea[:, 2] = 0.2
    ea[:, 3] = 1.0
    ea[:, GEO_DIM + ROLE_DEHYDRON] = 1.0
    y_full = conv_full(x, edge_index, ea)
    y_weak = conv_weak(x, edge_index, ea)
    assert not torch.allclose(y_full, y_weak)


def test_feeler_phase_configs_for_experiments() -> None:
    from science.training.config import v66_feeler_rim_decouple_phase_config

    p7 = v66_feeler_no_exclusivity_phase_config()
    p8 = v66_feeler_dehydron_angular_phase_config()
    p9 = v66_feeler_rim_decouple_phase_config()
    assert p7.phase == 7
    assert p8.phase == 8
    assert p9.phase == 9
    assert p7.coeffs.disc_occupancy_coeff == 0.0
    assert p9.coeffs.disc_occupancy_coeff == 0.0
