"""Unit tests for mean-residue routing entropy sparsity helpers."""

from __future__ import annotations

import math

import torch

from science.training.routing_sparsity import (
    advance_sparse_lam,
    detect_sparse_capacity_collision,
    mean_residue_routing_entropy,
    sparse_coeff_at_epoch,
    sparse_vs_cap_ratio,
    sparsity_governor_update,
)


def test_mean_residue_entropy_uniform_four() -> None:
    scores = torch.full((10, 4), 0.25)
    h = mean_residue_routing_entropy(scores)
    assert abs(h.item() - math.log(4)) < 1e-5


def test_mean_residue_entropy_one_hot() -> None:
    scores = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    h = mean_residue_routing_entropy(scores)
    assert h.item() < 1e-5


def test_mean_residue_entropy_requires_grad() -> None:
    logits = torch.randn(8, 4, requires_grad=True)
    scores = torch.softmax(logits, dim=-1)
    h = mean_residue_routing_entropy(scores)
    h.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_warmup_linear() -> None:
    assert sparse_coeff_at_epoch(1, peak=0.01, warmup=8) == 0.01 * (1 / 8)
    assert sparse_coeff_at_epoch(8, peak=0.01, warmup=8) == 0.01
    assert sparse_coeff_at_epoch(20, peak=0.01, warmup=8) == 0.01


def test_warmup_zero_epochs_is_peak() -> None:
    assert sparse_coeff_at_epoch(1, peak=0.0075, warmup=0) == 0.0075


def test_advance_sparse_lam_matches_linear_warmup() -> None:
    peak, warmup = 0.008, 8
    lam = 0.0
    for t in range(1, 9):
        lam, hold = advance_sparse_lam(
            peak=peak, warmup=warmup, last_lam=lam, slope_scale=1.0
        )
        assert hold == 0
        assert lam == sparse_coeff_at_epoch(t, peak=peak, warmup=warmup)


def test_advance_sparse_lam_hold_freezes_then_resumes() -> None:
    lam, hold = advance_sparse_lam(
        peak=0.008,
        warmup=8,
        last_lam=0.003,
        hold_remaining=2,
        hold_lam=0.002,
    )
    assert lam == 0.002
    assert hold == 1
    lam2, hold2 = advance_sparse_lam(
        peak=0.008,
        warmup=8,
        last_lam=lam,
        slope_scale=1.0,
        hold_remaining=hold,
        hold_lam=0.002,
    )
    assert lam2 == 0.002
    assert hold2 == 0
    lam3, _ = advance_sparse_lam(
        peak=0.008, warmup=8, last_lam=lam2, slope_scale=1.0
    )
    assert lam3 == 0.002 + (0.008 / 8)


def test_governor_half_ramp_in_warning_band() -> None:
    out = sparsity_governor_update(
        max_soft_share=0.42,
        newly_banned=[],
        prev_lam=0.003,
        slope_scale=1.0,
        half_slope_applied=False,
    )
    assert out["events"] == ["half_ramp"]
    assert out["slope_scale"] == 0.5
    assert out["half_slope_applied"] is True
    assert out["hold_remaining"] == 0


def test_governor_timeout_hold_on_ban_and_045() -> None:
    out = sparsity_governor_update(
        max_soft_share=0.45,
        newly_banned=[2],
        prev_lam=0.004,
        slope_scale=1.0,
        half_slope_applied=False,
    )
    assert "timeout_hold" in out["events"]
    assert out["hold_remaining"] == 2
    assert out["hold_lam"] == 0.004
    # timeout band is exclusive of half_ramp
    assert "half_ramp" not in out["events"]


def test_sparse_vs_cap_ratio() -> None:
    r = sparse_vs_cap_ratio(
        lam_sparse=0.0075,
        l_sparse=1.2,
        balance_coeff=0.015,
        capacity_loss=0.0,
    )
    assert r > 1e6  # capacity near zero → large ratio


def test_collision_detector_triggers_on_capacity_rise() -> None:
    series = [
        {
            "routing_entropy_mean_residue": 1.3,
            "capacity_loss": 1e-6,
            "usage_max_soft_share": 0.28,
        },
        {
            "routing_entropy_mean_residue": 1.1,
            "capacity_loss": 5e-4,
            "usage_max_soft_share": 0.30,
        },
    ]
    assert detect_sparse_capacity_collision(series) == 2


def test_collision_detector_triggers_on_max_share_warning() -> None:
    series = [
        {
            "routing_entropy_mean_residue": 1.2,
            "capacity_loss": 1e-6,
            "usage_max_soft_share": 0.28,
        },
        {
            "routing_entropy_mean_residue": 1.0,
            "capacity_loss": 1e-6,
            "usage_max_soft_share": 0.41,
        },
    ]
    assert detect_sparse_capacity_collision(series) == 2


def test_collision_detector_none_when_no_collision() -> None:
    series = [
        {
            "routing_entropy_mean_residue": 1.3,
            "capacity_loss": 1e-6,
            "usage_max_soft_share": 0.28,
        },
        {
            "routing_entropy_mean_residue": 1.1,
            "capacity_loss": 1e-6,
            "usage_max_soft_share": 0.29,
        },
    ]
    assert detect_sparse_capacity_collision(series) is None


def test_v66_model_exports_mean_residue_entropy_key() -> None:
    """Forward contract: GOSPConeMapperV66 must emit sparsity entropy field."""
    import inspect

    from science.dtie.v66.gnn import model as model_mod

    src = inspect.getsource(model_mod.GOSPConeMapperV66.forward)
    assert "routing_entropy_mean_residue" in src
    assert "mean_residue_routing_entropy" in src


def test_stage_runner_wires_sparsity_schedule_and_jsonl() -> None:
    """Stage runner must schedule λ and write routing_sparsity_per_epoch.jsonl."""
    import inspect

    from experiments.training.v66 import stage_runner as sr

    src = inspect.getsource(sr.StageRunner)
    assert "advance_sparse_lam" in src
    assert "sparsity_governor_update" in src
    assert "routing_sparsity_per_epoch.jsonl" in src
    assert "_log_routing_sparsity_epoch" in src
    assert "routing_entropy_sparsity_coeff" in src


def test_loss_coeffs_default_sparsity_off() -> None:
    from science.training.config import LossCoeffs, TrainingConfig

    assert LossCoeffs().routing_entropy_sparsity_coeff == 0.0
    assert LossCoeffs().routing_entropy_sparsity_warmup_epochs == 8
    assert TrainingConfig().routing_entropy_sparsity_coeff == 0.0
    assert TrainingConfig().routing_entropy_sparsity_warmup_epochs == 8


def test_gosp_loss_v66_wires_mean_residue_sparsity() -> None:
    """Scheduled coeff * mean_i H(p_i) enters total; H(f̄) stays monitor-only."""
    import pytest

    from science.dtie.v66.loss import gosp_loss_v6

    device = "cpu"
    n = 8
    L_sparse = torch.tensor(1.2, device=device, requires_grad=True)
    output = {
        "x_hyp": torch.randn(n, 3, device=device),
        "x_routed_hyp": torch.randn(n, 3, device=device),
        "hyp_projections_2d": 0.1 * torch.randn(n, 2, device=device),
        "hyp_projections_3d": 0.1 * torch.randn(n, 3, device=device),
        "radial_features": torch.randn(n, 1, device=device),
        "cone_depth": torch.rand(n, 1, device=device),
        "evidence": {
            "mu": torch.rand(n, device=device),
            "nu": torch.rand(n, device=device).abs() + 0.1,
            "alpha": torch.rand(n, device=device).abs() + 1.0,
            "beta": torch.rand(n, device=device).abs() + 0.1,
        },
        "uncertainty": {"epistemic": torch.rand(n, device=device)},
        "capacity_loss": torch.tensor(0.0, device=device),
        "routing_entropy": torch.tensor(1.386, device=device),
        "routing_entropy_mean_residue": L_sparse,
        "expert_load": torch.ones(4, device=device) * 0.25,
        "audit_trail": {"curvature_value": torch.tensor(-1.0, device=device)},
    }
    target = torch.rand(n, device=device)
    coords = torch.randn(n, 3, device=device)
    zeroed = dict(
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
    off = gosp_loss_v6(output, target, coords, routing_entropy_sparsity_coeff=0.0, **zeroed)
    on = gosp_loss_v6(
        output, target, coords, routing_entropy_sparsity_coeff=0.0075, **zeroed
    )
    expected = 0.0075 * 1.2
    assert float(on["routing_entropy_sparsity_loss"]) == pytest.approx(expected, rel=1e-5)
    assert float(on["routing_entropy_sparsity_coeff"]) == pytest.approx(0.0075)
    assert float(on["routing_entropy_mean_residue"]) == pytest.approx(1.2)
    assert float(on["routing_entropy"]) == pytest.approx(1.386)
    assert float(off["routing_entropy_sparsity_loss"]) == pytest.approx(0.0)
    assert float(on["total"]) - float(off["total"]) == pytest.approx(expected, rel=1e-5)
    on["total"].backward()
    assert L_sparse.grad is not None
    assert float(L_sparse.grad) == pytest.approx(0.0075, rel=1e-5)
