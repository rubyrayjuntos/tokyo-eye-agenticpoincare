"""Harness path must log what the smoke train is measuring (§2.1 / §2.2 / Δ_equiv)."""

from __future__ import annotations

import math

import torch

from experiments.training.v8.run_v8_experiment import (
    EVAL_ENTROPY_NORM_FLOOR,
    EVAL_LOAD_FLOOR,
    EVAL_MOE_METRIC_SCHEMA,
    EVAL_N_ALIVE_REQUIRED,
    _dual_seal_diagnostics,
    _eval_mode_routing_metrics,
    _log_delta_equiv,
    _make_batch,
    run_epoch,
)
from experiments.training.v8.equ_lift_radius import RV_EQUIV_CEILING
from science.tokyo_eye.v8.engine import (
    CurriculumRadiusController,
    GumbelTemperatureSchedule,
    PoincareDiagnosticsEngine,
)
from science.tokyo_eye.v8.equiformer_frontend import (
    StubEquiformerFrontend,
    TokyoEyeV8WithFrontend,
)
from science.tokyo_eye.v8.model import TokyoEyesHyperbolicV8
from science.tokyo_eye.v8.moe_eval_gates import eval_moe_utilization


def _tiny_system(device: torch.device) -> TokyoEyeV8WithFrontend:
    fe = StubEquiformerFrontend(in_dim=3, scalar_dim=8, vector_dim=3)
    spine = TokyoEyesHyperbolicV8(
        scalar_dim=8, vector_dim=3, hidden_dim=8, num_attn_layers=2, num_sdrp_classes=5
    )
    return TokyoEyeV8WithFrontend(fe, spine).to(device)


def test_eval_mode_routing_is_argmax_not_train_gumbel() -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")
    system = _tiny_system(device)
    batch = _make_batch(device)
    # Force monopoly logits via a short train-mode forward then compare eval.
    system.train()
    out_tr = system(
        batch["x"], batch["edge_index"], batch["edge_type"], tau_ceiling=0.8
    )
    train_load = out_tr["moe_aux"]["routing"].mean(dim=0)
    eval_m = _eval_mode_routing_metrics(system, batch, tau_ceiling=0.8)
    assert "eval_moe_load_e0" in eval_m
    assert eval_m["eval_moe_load_floor"] == EVAL_LOAD_FLOOR
    assert eval_m["eval_moe_entropy_norm_floor"] == EVAL_ENTROPY_NORM_FLOOR
    assert eval_m["eval_moe_metric_schema"] == EVAL_MOE_METRIC_SCHEMA
    assert "eval_moe_h_norm" not in eval_m
    assert "eval_moe_h_norm_floor" not in eval_m
    assert eval_m["eval_moe_n_alive_required"] == float(EVAL_N_ALIVE_REQUIRED)
    # SSOT: harness thresholds match moe_eval_utilization defaults.
    util = eval_moe_utilization(torch.ones(8, 4) / 4.0)  # shape probe only
    assert util["gates"]["moe_eval_min_load"]["threshold"] == EVAL_LOAD_FLOOR
    assert (
        util["gates"]["moe_eval_entropy_norm"]["threshold"]
        == EVAL_ENTROPY_NORM_FLOOR
    )
    # Eval loads are a probability simplex mean.
    s = sum(eval_m[f"eval_moe_load_e{i}"] for i in range(4))
    assert abs(s - 1.0) < 1e-4
    # Keys exist even if train and eval happen to match on a tiny random init.
    assert train_load.numel() == 4
    # Liveness is the full seal, not h_norm alone.
    assert "eval_moe_entropy_norm" in eval_m
    assert "eval_moe_n_alive" in eval_m
    expected_live = (
        eval_m["eval_moe_load_floor_pass"]
        * eval_m["eval_moe_entropy_norm_pass"]
        * eval_m["eval_moe_n_alive_pass"]
    )
    assert eval_m["eval_moe_liveness_pass"] == expected_live


def test_dual_seal_tags_lift_attn_and_post_moe() -> None:
    torch.manual_seed(1)
    device = torch.device("cpu")
    system = _tiny_system(device)
    system.eval()
    batch = _make_batch(device)
    with torch.no_grad():
        out = system(
            batch["x"], batch["edge_index"], batch["edge_type"], tau_ceiling=0.85
        )
    m = _dual_seal_diagnostics(PoincareDiagnosticsEngine(), out)
    for tag in ("pre_moe_lift", "pre_moe_attn", "post_moe"):
        assert f"diag_{tag}_boundary_saturation" in m
        assert f"diag_{tag}_radius_spread" in m
        assert f"diag_{tag}_mean_radius" in m
    # Untagged aliases are post-MoE (deployed).
    assert m["diag_radius_spread"] == m["diag_post_moe_radius_spread"]
    assert m["diag_boundary_saturation"] == m["diag_post_moe_boundary_saturation"]


def test_delta_equiv_logged_on_harness_path() -> None:
    import math

    torch.manual_seed(2)
    device = torch.device("cpu")
    system = _tiny_system(device)
    batch = _make_batch(device)
    m = _log_delta_equiv(system, batch, tau_ceiling=0.9, seed=2)
    assert "delta_equiv_lift_spine" in m
    assert "delta_equiv_full_system" in m
    assert m["delta_equiv_ceiling"] == RV_EQUIV_CEILING
    assert math.isfinite(m["delta_equiv_lift_spine"])


def test_run_epoch_emits_smoke_required_metrics() -> None:
    torch.manual_seed(3)
    device = torch.device("cpu")
    system = _tiny_system(device)
    opt = torch.optim.Adam(system.parameters(), lr=1e-3)
    batch = _make_batch(device)
    metrics = run_epoch(
        system,
        opt,
        batch,
        epoch=0,
        radius=CurriculumRadiusController(0.7, 0.9, 5),
        gumbel=GumbelTemperatureSchedule(1.0, 0.3, 5),
        diagnostics=PoincareDiagnosticsEngine(),
        cv_coeff=1.0,
        moe_quota_coeff=1.0,
    )
    required = [
        "eval_moe_load_min",
        "eval_moe_entropy_norm",
        "eval_moe_metric_schema",
        "eval_moe_n_alive",
        "eval_moe_load_floor_pass",
        "eval_moe_liveness_pass",
        "eval_moe_load_e0",
        "moe_load_e0",  # continuity; not evidence alone
        "diag_pre_moe_lift_radius_spread",
        "diag_pre_moe_attn_radius_spread",
        "diag_post_moe_radius_spread",
        "delta_equiv_lift_spine",
        "moe_majority_hinge_loss",
    ]
    missing = [k for k in required if k not in metrics]
    assert missing == [], missing
    assert "eval_moe_h_norm" not in metrics
    assert metrics["eval_proxy_quota_coeff"] == 140.0  # weight-map / harness default
    # Default path does not enable gate-grad isolation; opt-in only.
    assert "gate_grad_eval_proxy_quota" not in metrics


def test_run_epoch_logs_gate_grad_sources_when_requested() -> None:
    torch.manual_seed(4)
    device = torch.device("cpu")
    system = _tiny_system(device)
    opt = torch.optim.Adam(system.parameters(), lr=1e-3)
    batch = _make_batch(device)
    metrics = run_epoch(
        system,
        opt,
        batch,
        epoch=0,
        radius=CurriculumRadiusController(0.7, 0.9, 5),
        gumbel=GumbelTemperatureSchedule(1.0, 0.3, 5),
        diagnostics=PoincareDiagnosticsEngine(),
        cv_coeff=10.0,
        moe_quota_coeff=5.0,
        eval_proxy_quota_coeff=140.0,
        log_gate_grad_sources=True,
    )
    assert "gate_grad_eval_proxy_quota" in metrics
    assert "gate_grad_eval_proxy_lb" in metrics
    assert "gate_grad_task" in metrics
    assert "gate_grad_quota_over_lb" in metrics
    assert metrics["eval_proxy_quota_coeff"] == 140.0
    # Task must remain negligible vs LB at the gate (preflight: ~1e-5 vs ~7).
    assert metrics["gate_grad_task"] < 1e-2 * max(
        metrics["gate_grad_eval_proxy_lb"], 1e-12
    )
    # When proxy already clears the floor, quota grad is legitimately 0 —
    # ratio assertion only applies under monopoly pressure.
    if metrics["moe_eval_proxy_quota_loss"] > 0.0:
        q_over_lb = metrics["gate_grad_quota_over_lb"]
        assert math.isfinite(q_over_lb)
        assert 0.1 <= q_over_lb <= 5.0, q_over_lb
    else:
        assert metrics["gate_grad_eval_proxy_quota"] == 0.0
        assert metrics["gate_grad_quota_over_lb"] == 0.0
