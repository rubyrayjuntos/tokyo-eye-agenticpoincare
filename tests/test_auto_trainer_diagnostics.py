"""Tests for goal-driven auto-trainer diagnostics."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from experiments.training.v6.auto_trainer import TrainerState, load_playbook
from experiments.training.v6.auto_trainer_diagnostics import (
    DiagnosticBundle,
    LastRunContext,
    build_metrics_snapshot,
    evaluate_goals,
    plan_training_iteration,
)


@pytest.fixture
def playbook(tmp_path: Path) -> dict:
    path = tmp_path / "playbook.yaml"
    path.write_text(
        (Path(__file__).resolve().parents[1] / "manifests/auto_trainer/corpus25_playbook.yaml").read_text()
    )
    return load_playbook(path)


@pytest.fixture
def state(playbook: dict) -> TrainerState:
    return TrainerState.from_playbook(playbook)


def test_evaluate_goals_detects_failing_epi_sasa(playbook: dict) -> None:
    metrics = {
        "probe_r_epi_sasa": 0.14,
        "routing_entropy": 1.04,
        "per_structure_min_routing": 0.06,
    }
    goals = evaluate_goals(playbook, metrics)
    epi = next(g for g in goals if g.goal_id == "probe_r_epi_sasa")
    assert epi.passing is False
    assert epi.gap == pytest.approx(0.06)


def test_plan_epistemic_shell_regime(playbook: dict, state: TrainerState) -> None:
    from experiments.training.v6.auto_trainer import MetricSnapshot

    snap = MetricSnapshot(
        focus={"primary_focus": ["probe_r_epi_sasa"], "primary_focus_str": "probe_r_epi_sasa"},
        assess={"promotion_gate": {"passed": True}},
        champion_score=3.2,
        routing_entropy=1.04,
        probe_r_epi_sasa=0.14,
        promotion_gate_passed=True,
        expert_starvation_count=0,
    )
    bundle = DiagnosticBundle(
        champion_run_id=state.champion_run_id,
        champion_checkpoint=state.champion_checkpoint,
        metrics=build_metrics_snapshot(
            snap,
            {"per_structure_min_routing": 0.06, "structures_below_floor": []},
        ),
        focus=snap.focus,
        assess=snap.assess,
        routing_floor={"per_structure_min_routing": 0.06, "structures_below_floor": []},
        goals=evaluate_goals(
            playbook,
            build_metrics_snapshot(snap, {"per_structure_min_routing": 0.06}),
        ),
        last_run=LastRunContext(action_id="touchup_extended", promoted=False),
        primary_failing_goal="probe_r_epi_sasa",
    )
    plan = plan_training_iteration(playbook, state, bundle)
    assert plan is not None
    assert plan.regime_id == "epistemic_shell"
    assert plan.action_id == "touchup_from_gate"
    assert plan.env["RESUME"].endswith("gate_v4/v6_best.pt")


def test_plan_load_floor_adds_gate_coeffs(playbook: dict, state: TrainerState) -> None:
    from experiments.training.v6.auto_trainer import MetricSnapshot

    snap = MetricSnapshot(
        focus={"primary_focus_str": "none"},
        assess={"promotion_gate": {"passed": True}},
        champion_score=3.2,
        routing_entropy=1.04,
        probe_r_epi_sasa=0.25,
        promotion_gate_passed=True,
        expert_starvation_count=0,
    )
    bundle = DiagnosticBundle(
        champion_run_id=state.champion_run_id,
        champion_checkpoint=state.champion_checkpoint,
        metrics={
            **build_metrics_snapshot(snap, {"per_structure_min_routing": 0.047, "worst_pdb": "1CTF"}),
        },
        focus=snap.focus,
        assess=snap.assess,
        routing_floor={
            "per_structure_min_routing": 0.047,
            "structures_below_floor": ["1CTF", "1CRN"],
            "worst_pdb": "1CTF",
        },
        goals=evaluate_goals(
            playbook,
            build_metrics_snapshot(snap, {"per_structure_min_routing": 0.047}),
        ),
        last_run=LastRunContext(),
        primary_failing_goal="per_structure_min_routing",
    )
    plan = plan_training_iteration(playbook, state, bundle)
    assert plan is not None
    assert plan.regime_id == "load_floor"
    assert plan.action_id == "gate_promotion"
    assert plan.env.get("P4_GATE_LOAD_FLOOR_COEFF") == "14"
