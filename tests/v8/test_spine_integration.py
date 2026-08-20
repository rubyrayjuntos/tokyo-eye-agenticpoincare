"""Sprint 4: TokyoEyesHyperbolicV8 integration + train_v8_step (isolated)."""

from __future__ import annotations

import pytest
import torch

from science.tokyo_eye.v8.engine import (
    CurriculumRadiusController,
    GumbelTemperatureSchedule,
    PoincareDiagnosticsEngine,
    train_v8_step,
)
from science.tokyo_eye.v8.model import TokyoEyesHyperbolicV8


def test_curriculum_radius_opens() -> None:
    ctrl = CurriculumRadiusController(tau_start=0.70, tau_end=0.995, total_epochs=10)
    assert ctrl.tau_ceiling(0) == pytest.approx(0.70)
    assert ctrl.tau_ceiling(10) == pytest.approx(0.995)
    mid = ctrl.tau_ceiling(5)
    assert 0.70 < mid < 0.995


def test_gumbel_temp_schedule_cools() -> None:
    sched = GumbelTemperatureSchedule(
        tau_start=1.0, tau_end=0.3, total_epochs=8, schedule="linear"
    )
    assert sched.temperature(0) >= sched.temperature(8)


def test_poincare_diagnostics_finite() -> None:
    eng = PoincareDiagnosticsEngine(boundary_radius=0.90, core_radius=0.30)
    z = torch.randn(20, 8) * 0.2
    z = z / (z.norm(dim=-1, keepdim=True).clamp_min(1e-5)) * 0.5
    summary = eng.summarize(z)
    assert "mean_radius" in summary
    assert "boundary_saturation" in summary
    assert "radial_entropy" in summary
    assert torch.isfinite(torch.tensor(summary["mean_radius"]))


def test_model_forward_spine() -> None:
    torch.manual_seed(0)
    n, d_s, d_v, d = 6, 16, 3, 16
    model = TokyoEyesHyperbolicV8(
        scalar_dim=d_s,
        vector_dim=d_v,
        hidden_dim=d,
        num_attn_layers=2,
        num_sdrp_classes=5,
    )
    model.train()
    s = torch.randn(n, d_s)
    v = torch.randn(n, d_v) * 0.1
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    edge_type = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    out = model(s, v, edge_index, edge_type, tau_ceiling=0.85)
    assert out["z_hyp"].shape == (n, d)
    assert float(out["z_hyp"].detach().norm(dim=-1).max()) < 0.85 + 1e-3
    assert out["sdrp_logits"].shape == (n, 5)
    assert out["evidence"].shape[0] == n
    assert out["moe_aux"]["routing"].shape == (n, 4)


def test_train_v8_step_runs_and_clips() -> None:
    torch.manual_seed(1)
    n, d_s, d_v, d = 5, 8, 3, 8
    model = TokyoEyesHyperbolicV8(
        scalar_dim=d_s,
        vector_dim=d_v,
        hidden_dim=d,
        num_attn_layers=2,
        num_sdrp_classes=3,
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = {
        "s": torch.randn(n, d_s),
        "v": torch.randn(n, d_v) * 0.1,
        "edge_index": torch.tensor([[0, 1, 2, 1], [1, 0, 1, 2]], dtype=torch.long),
        "edge_type": torch.tensor([0, 0, 2, 2], dtype=torch.long),
        "sdrp_target": torch.tensor([0, 1, 2, 1, 0], dtype=torch.long),
        "mechanism_pos": torch.tensor([0.8, 0.7, 0.6, 0.5, 0.4]),
        "mechanism_neg": torch.tensor([0.2, 0.3, 0.25, 0.35, 0.15]),
    }
    radius = CurriculumRadiusController(0.70, 0.995, 20)
    gumbel = GumbelTemperatureSchedule(1.0, 0.4, 20)
    diagnostics = PoincareDiagnosticsEngine()
    metrics = train_v8_step(
        model,
        batch,
        opt,
        epoch=2,
        radius_controller=radius,
        gumbel_schedule=gumbel,
        diagnostics=diagnostics,
        max_grad_norm=1.0,
    )
    assert "loss_total" in metrics
    assert "tau_ceiling" in metrics
    assert "gumbel_temperature" in metrics
    assert "diag_mean_radius" in metrics
    assert metrics["max_grad_norm"] == pytest.approx(1.0)
    assert metrics["grad_norm"] >= 0.0
    assert torch.isfinite(torch.tensor(metrics["loss_total"]))
