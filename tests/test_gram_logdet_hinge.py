"""Unit tests for saturating prototype Gram logdet hinge (GRAM_COND pre-reg)."""

from __future__ import annotations

import pytest
import torch

from experiments.diagnostics.prototype_repulsion_epoch import score_gram_cond_ladder
from science.dtie.v6.loss import gosp_loss_v6, prototype_gram_logdet_hinge
from science.training.config import (
    apply_prototype_gram_logdet_hinge,
    apply_v66_feeler_phases,
)
from tests.test_prototype_repulsion import _ZERO_GEOM, _minimal_output


def test_gram_hinge_pushes_when_below_tau() -> None:
    logdet = torch.tensor(-2.64, requires_grad=True)
    raw = prototype_gram_logdet_hinge(logdet, tau_logdet=-1.15)
    assert float(raw) == pytest.approx((2.64 - 1.15) ** 2, abs=1e-5)
    raw.backward()
    assert logdet.grad is not None
    assert float(logdet.grad) < 0  # increasing logdet lowers hinge


def test_gram_hinge_saturates_at_or_above_tau() -> None:
    logdet = torch.tensor(-1.0, requires_grad=True)
    raw = prototype_gram_logdet_hinge(logdet, tau_logdet=-1.15)
    assert float(raw) == 0.0
    # Saturated ReLU has zero local gradient w.r.t. logdet.
    raw.backward()
    assert logdet.grad is None or float(logdet.grad) == 0.0


def test_gosp_applies_gram_hinge_coeff() -> None:
    logdet = torch.tensor(-2.15, requires_grad=True)
    out = _minimal_output(pair_min=torch.tensor(0.40))
    out["prototype_gram_logdet"] = logdet
    losses = gosp_loss_v6(
        out,
        target_rho=torch.rand(8),
        ca_coords=torch.randn(8, 3),
        prototype_gram_logdet_coeff=0.001,
        prototype_gram_logdet_tau=-1.15,
        **_ZERO_GEOM,
    )
    expected = 0.001 * (1.0**2)
    assert float(losses["prototype_gram_logdet_hinge"]) == pytest.approx(
        expected, abs=1e-8
    )
    losses["total"].backward()
    assert logdet.grad is not None


def test_apply_gram_hinge_patches_all_phases() -> None:
    phases = apply_v66_feeler_phases(epochs=20)
    patched = apply_prototype_gram_logdet_hinge(phases, coeff=0.001, tau_logdet=-1.15)
    assert all(p.coeffs.prototype_gram_logdet_coeff == 0.001 for p in patched)
    assert all(p.coeffs.prototype_gram_logdet_tau == -1.15 for p in patched)


def test_score_gram_cond_win_and_no_move() -> None:
    win = score_gram_cond_ladder(
        {
            "frac_max_p_ge_0_60": 0.80,
            "best_tau_mean_contrast": 1.0,
            "gram_eig_min": 0.20,
            "gram_condition": 8.0,
            "gram_logdet": -1.0,
            "nearest_pair_hyp_dist": 0.40,
            "dehydron_partition_purity": {
                "passes": True,
                "n_eligible_structures": 12,
                "blurred_pdb_ids": [],
            },
        }
    )
    assert win["verdict"] == "GRAM_COND_WIN"

    no_move = score_gram_cond_ladder(
        {
            "frac_max_p_ge_0_60": 0.80,
            "best_tau_mean_contrast": 1.0,
            "gram_eig_min": 0.03,
            "gram_condition": 50.0,
            "gram_logdet": -2.64,
            "nearest_pair_hyp_dist": 0.40,
            "dehydron_partition_purity": {
                "passes": True,
                "n_eligible_structures": 12,
                "blurred_pdb_ids": [],
            },
        }
    )
    assert no_move["verdict"] == "GRAM_COND_NO_MOVE"
    assert "GRAM_NO_MOVE" in no_move["fails"]
