"""Tests for geometric angular prior (v6.6 Fix 1)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from science.dtie.v66.geometric_angular_prior import (
    DEFAULT_ALPHA,
    DiscAngularResidual,
    apply_geometric_disc_prior,
    compute_geometric_angular_prior,
    protein_pca_plane,
    project_to_plane,
)


def test_protein_pca_plane_orthonormal() -> None:
    rng = np.random.default_rng(0)
    ca = rng.normal(size=(40, 3))
    e1, e2 = protein_pca_plane(ca)
    assert abs(np.dot(e1, e1) - 1.0) < 1e-6
    assert abs(np.dot(e2, e2) - 1.0) < 1e-6
    assert abs(np.dot(e1, e2)) < 1e-6


def test_zero_dehydron_falls_back_to_peptide() -> None:
    """No residue records → no dehydrons → peptide/[1,0] prior, finite."""
    ca = np.array(
        [[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [7.6, 0.5, 0.0], [11.0, 0.0, 0.2]],
        dtype=np.float64,
    )
    out = compute_geometric_angular_prior(ca, residue_records=None, kappa=1.0)
    assert out["theta_prior"].shape == (4,)
    assert np.all(np.isfinite(out["theta_prior"]))
    assert np.all(out["blend_mass"] == 0.0)
    assert np.allclose(out["sigma"], 0.0)
    # unit priors
    norms = np.linalg.norm(out["u_prior"], axis=1)
    assert np.allclose(norms, 1.0)


def test_opposite_dehydrons_cancel_to_peptide() -> None:
    """m = ||Σwu|| → 0 when weighted dehydron vectors cancel → peptide dominates."""
    # Synthetic: force v_dh cancel via direct assembly of formula
    e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.array([0.0, 1.0, 0.0])
    u_a = project_to_plane(np.array([1.0, 0.0, 0.0]), e1, e2)
    u_b = project_to_plane(np.array([-1.0, 0.0, 0.0]), e1, e2)
    u_a = u_a / np.linalg.norm(u_a)
    u_b = u_b / np.linalg.norm(u_b)
    v = 2.0 * u_a + 2.0 * u_b
    m = float(np.linalg.norm(v))
    assert m < 1e-9
    peptide = np.array([0.0, 1.0])
    sigma = 1.0 - math.exp(-m / 1.0)
    blend = v + (1.0 - sigma) * peptide
    prior = blend / np.linalg.norm(blend)
    assert abs(prior[1] - 1.0) < 1e-6


def test_residual_bound() -> None:
    xy = torch.tensor([[0.5, 0.0], [0.0, 0.4]], dtype=torch.float32)
    prior = torch.tensor([0.0, math.pi / 2], dtype=torch.float32)
    # large raw delta → tanh saturates
    delta = torch.tensor([100.0, -100.0], dtype=torch.float32)
    xy_new, theta, r = apply_geometric_disc_prior(xy, prior, delta, alpha=DEFAULT_ALPHA)
    assert torch.allclose(r, torch.tensor([0.5, 0.4]))
    # |Δθ| ≤ α
    dth = (theta - prior).abs()
    assert torch.all(dth <= DEFAULT_ALPHA + 1e-5)
    # radius preserved
    assert torch.allclose(xy_new.norm(dim=-1), r)


def test_disc_angular_residual_zero_init() -> None:
    head = DiscAngularResidual(16)
    x = torch.randn(5, 16)
    assert torch.allclose(head(x), torch.zeros(5), atol=1e-6)


def test_phase_config_fidelity_coeff() -> None:
    from science.training.config import v66_feeler_geom_angular_prior_phase_config

    phase = v66_feeler_geom_angular_prior_phase_config(epochs=20)
    assert phase.coeffs.geometric_angular_fidelity_coeff == 0.10
    assert phase.coeffs.expert_sector_recruit_coeff == 0.0
    assert phase.phase == 12


def test_model_requires_prior_tensor() -> None:
    from science.dtie.v66.gnn.model import GOSPConeMapperV66
    from torch_geometric.data import Data

    m = GOSPConeMapperV66(
        node_dim=3,
        hidden=32,
        num_layers=1,
        num_experts=2,
        geometric_angular_prior=True,
        rim_fanout_forward=False,
    )
    n = 8
    data = Data(
        x=torch.randn(n, 3),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        edge_attr=torch.randn(3, 4),
        clustering=torch.zeros(n, 1),
        degree=torch.ones(n),
        rho=torch.ones(n) * 10.0,
        ss_onehot=torch.zeros(n, 3),
    )
    with pytest.raises(ValueError, match="geom_theta_prior"):
        m(data)
