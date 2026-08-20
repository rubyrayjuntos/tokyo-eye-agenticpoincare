"""Path 2 directionality objective + diam≤9 mask."""

from __future__ import annotations

import pytest
import torch

from science.dtie.common.directionality_objective import (
    ASYMMETRY_FLOOR,
    DIAM_LE_9_PASS_GROUP,
    directionality_asym_loss,
    pairwise_asymmetry_index,
    pdb_in_diam_le9_pass_group,
    source_norm_cosine_influence,
)
from science.dtie.v6.loss import gosp_loss_v6
from science.training.config import apply_directionality_asym_reward, apply_v66_feeler_phases


def test_diam_le9_pass_group_locked() -> None:
    assert DIAM_LE_9_PASS_GROUP == {
        "1UBQ",
        "1TEN",
        "1HHP",
        "1LYZ",
        "4OBE",
        "1TIM",
        "1MBN",
    }
    assert pdb_in_diam_le9_pass_group("4obe")
    assert not pdb_in_diam_le9_pass_group("1F88")
    assert not pdb_in_diam_le9_pass_group("1BG1")
    assert ASYMMETRY_FLOOR == pytest.approx(0.05)


def test_pairwise_asymmetry_zero_on_symmetric() -> None:
    infl = torch.tensor([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
    asym = pairwise_asymmetry_index(infl)
    assert float(asym) == pytest.approx(0.0, abs=1e-6)


def test_pairwise_asymmetry_positive_when_directed() -> None:
    infl = torch.tensor([[0.0, 2.0, 0.1], [0.5, 0.0, 3.0], [1.0, 0.2, 0.0]])
    asym = pairwise_asymmetry_index(infl)
    assert float(asym) > 0.05


def test_source_norm_proxy_breaks_symmetry_when_norms_differ() -> None:
    h = torch.tensor([[3.0, 0.0], [0.5, 0.0], [1.0, 0.0]], dtype=torch.float32)
    infl = source_norm_cosine_influence(h)
    # Same direction → high cosine; I(a→b) ∝ ||h_a|| so row 0 > row 1 toward col 2.
    assert float(infl[0, 2]) > float(infl[1, 2])
    asym = pairwise_asymmetry_index(infl)
    assert float(asym) > 0.0


def test_directionality_loss_masked_when_ineligible() -> None:
    h = torch.randn(6, 4, requires_grad=True)
    out = directionality_asym_loss(h, coeff=0.05, eligible=False)
    assert float(out["directionality_asym"]) == 0.0
    assert out["directionality_asym"].requires_grad is False or float(
        out["directionality_asym"]
    ) == 0.0


def test_directionality_loss_pushes_asym_when_eligible() -> None:
    # Heterogeneous norms → nonzero asym; loss = λ(1−asym) should backprop.
    h = torch.tensor(
        [[2.0, 0.0], [0.1, 0.0], [1.0, 0.5], [0.3, 0.8]],
        dtype=torch.float32,
        requires_grad=True,
    )
    out = directionality_asym_loss(h, coeff=1.0, eligible=True)
    loss = out["directionality_asym"]
    assert float(out["directionality_asym_index"]) > 0.0
    assert float(loss) > 0.0
    loss.backward()
    assert h.grad is not None
    assert float(h.grad.norm()) > 0.0


def test_gosp_loss_directionality_respects_eligibility() -> None:
    n, e = 8, 4
    h = torch.randn(n, 16, requires_grad=True)
    output = {
        "encoder_h": h,
        "x_hyp": torch.randn(n, 2) * 0.1,
        "x_routed_hyp": torch.randn(n, 2) * 0.1,
        "radial_features": torch.linspace(0.1, 0.9, n).unsqueeze(1),
        "cone_depth": torch.linspace(0.5, 2.0, n).unsqueeze(1),
        "hyp_projections_2d": torch.randn(n, 2) * 0.3,
        "hyp_projections_3d": torch.randn(n, 3) * 0.3,
        "capacity_loss": torch.tensor(0.0),
        "routing_entropy": torch.tensor(1.0),
        "expert_load": torch.ones(e) * 0.25,
        "evidence": {
            "mu": torch.zeros(n),
            "nu": torch.ones(n),
            "alpha": torch.ones(n) * 2.0,
            "beta": torch.ones(n),
        },
        "uncertainty": {"epistemic": torch.linspace(0.2, 1.0, n).unsqueeze(1)},
        "audit_trail": {"curvature_value": -1.0},
    }
    zero = dict(
        evidential_coeff=0.0,
        balance_coeff=0.0,
        cone_coeff=0.0,
        neighborhood_coeff=0.0,
        angular_coeff=0.0,
        domain_sep_2d_coeff=0.0,
        domain_sep_3d_coeff=0.0,
        cone_depth_anticollapse_coeff=0.0,
        shell_corr_coeff=0.0,
        proj_violation_coeff=0.0,
        epistemic_bf_align_coeff=0.0,
        epistemic_sasa_pen_coeff=0.0,
        epistemic_anticollapse_coeff=0.0,
    )
    off = gosp_loss_v6(
        output,
        target_rho=torch.rand(n),
        ca_coords=torch.randn(n, 3),
        directionality_asym_coeff=0.05,
        directionality_eligible=False,
        **zero,
    )
    assert float(off["directionality_asym"]) == 0.0

    on = gosp_loss_v6(
        output,
        target_rho=torch.rand(n),
        ca_coords=torch.randn(n, 3),
        directionality_asym_coeff=0.05,
        directionality_eligible=True,
        **zero,
    )
    assert float(on["directionality_asym"]) >= 0.0
    on["total"].backward()
    assert h.grad is not None


def test_apply_directionality_asym_reward_patches_phases() -> None:
    phases = apply_v66_feeler_phases(epochs=5)
    patched = apply_directionality_asym_reward(phases, coeff=0.05)
    assert all(
        float(p.coeffs.directionality_asym_coeff) == pytest.approx(0.05) for p in patched
    )
    untouched = apply_directionality_asym_reward(phases, coeff=0.0)
    assert all(float(p.coeffs.directionality_asym_coeff) == 0.0 for p in untouched)
