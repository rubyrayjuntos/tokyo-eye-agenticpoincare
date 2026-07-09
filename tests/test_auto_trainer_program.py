"""Tests for multi-run program goals (Phase A)."""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.training.v6.auto_trainer import TrainerState, load_playbook
from experiments.training.v6.auto_trainer_diagnostics import (
    DiagnosticBundle,
    LastRunContext,
    build_metrics_snapshot,
    evaluate_goals,
    plan_training_iteration,
)
from experiments.training.v6.auto_trainer_program import (
    apply_program_evaluation,
    evaluate_program_goals,
    rebuild_progress_from_journal,
    requires_program_approval,
    update_program_progress_after_iteration,
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


def _bundle(playbook: dict, state: TrainerState, *, metrics: dict[str, float]) -> DiagnosticBundle:
    from experiments.training.v6.auto_trainer import MetricSnapshot

    snap = MetricSnapshot(
        focus={"primary_focus": ["probe_r_epi_sasa"], "primary_focus_str": "probe_r_epi_sasa"},
        assess={"promotion_gate": {"passed": True}},
        champion_score=3.2,
        routing_entropy=metrics.get("routing_entropy", 1.04),
        probe_r_epi_sasa=metrics.get("probe_r_epi_sasa", 0.14),
        promotion_gate_passed=True,
        expert_starvation_count=0,
    )
    merged = build_metrics_snapshot(snap, metrics)
    return DiagnosticBundle(
        champion_run_id=state.champion_run_id,
        champion_checkpoint=state.champion_checkpoint,
        metrics=merged,
        focus=snap.focus,
        assess=snap.assess,
        routing_floor={"per_structure_min_routing": metrics.get("per_structure_min_routing", 0.06)},
        goals=evaluate_goals(playbook, merged),
        last_run=LastRunContext(action_id="touchup_extended", promoted=False),
        primary_failing_goal="probe_r_epi_sasa",
    )


def test_rebuild_progress_counts_unpromoted_streak() -> None:
    events = [
        {"event": "train_complete", "run_id": "r1", "promoted": False},
        {"event": "plan", "run_id": "r1", "regime_id": "epistemic_shell", "metrics": {"probe_r_epi_sasa": 0.14}},
        {"event": "train_complete", "run_id": "r2", "promoted": False},
        {"event": "plan", "run_id": "r2", "regime_id": "epistemic_shell", "metrics": {"probe_r_epi_sasa": 0.14}},
        {"event": "train_complete", "run_id": "r3", "promoted": True},
        {"event": "plan", "run_id": "r3", "regime_id": "epistemic_shell", "metrics": {"probe_r_epi_sasa": 0.15}},
    ]
    progress = rebuild_progress_from_journal(events, critical_goal_ids=["probe_r_epi_sasa"])
    assert progress["consecutive_unpromoted"] == 0
    assert progress["consecutive_critical_promotions"] == 1
    assert progress["regime_attempts"]["epistemic_shell"] == 3
    assert progress["best_metrics"]["probe_r_epi_sasa"] == pytest.approx(0.15)


def test_metric_stall_escalates_to_full_push(playbook: dict, state: TrainerState, tmp_path: Path) -> None:
    state.program_progress = {
        "best_metrics": {"probe_r_epi_sasa": 0.14},
        "iterations_since_improvement": {"probe_r_epi_sasa": 4},
        "consecutive_unpromoted": 1,
        "consecutive_critical_promotions": 0,
        "regime_attempts": {"epistemic_shell": 2},
        "total_train_iterations": 4,
    }
    bundle = _bundle(playbook, state, metrics={"probe_r_epi_sasa": 0.14, "per_structure_min_routing": 0.06})
    journal = tmp_path / "journal.jsonl"
    journal.write_text("")

    evaluation = evaluate_program_goals(playbook, state, bundle, journal_path=journal)
    stall = next(c for c in evaluation.checks if c.check_id == "metric_progress.probe_r_epi_sasa")
    assert stall.status == "warn"
    assert evaluation.escalation_action == "full_push"
    assert evaluation.status == "warn"


def test_consecutive_unpromoted_blocks_program(playbook: dict, state: TrainerState, tmp_path: Path) -> None:
    state.program_progress = {
        "consecutive_unpromoted": 3,
        "iterations_since_improvement": {},
        "best_metrics": {},
        "regime_attempts": {},
        "total_train_iterations": 3,
    }
    bundle = _bundle(playbook, state, metrics={"probe_r_epi_sasa": 0.14})
    evaluation = evaluate_program_goals(
        playbook, state, bundle, journal_path=tmp_path / "empty.jsonl"
    )
    assert evaluation.status == "blocked"
    assert "consecutive unpromoted" in (evaluation.block_reason or "").lower()


def test_forced_escalation_plan(playbook: dict, state: TrainerState) -> None:
    bundle = _bundle(playbook, state, metrics={"probe_r_epi_sasa": 0.14})
    plan = plan_training_iteration(playbook, state, bundle, forced_action_id="full_push")
    assert plan is not None
    assert plan.action_id == "full_push"
    assert plan.regime_id in {"epistemic_shell", "program_escalation"}
    assert plan.env["TOUCHUP_EXTENDED_RUN"] == plan.run_id
    assert plan.env["GATE_PROMOTION_RUN"] == f"{plan.run_id}_gate"
    assert plan.env["GATE_TOUCHUP_RUN"] == f"{plan.run_id}_final"


def test_full_push_requires_approval(playbook: dict) -> None:
    assert requires_program_approval(playbook, "full_push") is True
    assert requires_program_approval(playbook, "touchup_from_gate") is False


def test_update_program_progress_tracks_metric_improvement(playbook: dict, state: TrainerState) -> None:
    bundle = _bundle(playbook, state, metrics={"probe_r_epi_sasa": 0.16})
    progress = update_program_progress_after_iteration(
        {},
        bundle=bundle,
        plan_regime="epistemic_shell",
        promoted=True,
        critical_goal_ids=["probe_r_epi_sasa"],
    )
    assert progress["best_metrics"]["probe_r_epi_sasa"] == pytest.approx(0.16)
    assert progress["iterations_since_improvement"]["probe_r_epi_sasa"] == 0
    assert progress["consecutive_critical_promotions"] == 1


def test_apply_program_evaluation_updates_state(
    playbook: dict, state: TrainerState, tmp_path: Path
) -> None:
    bundle = _bundle(playbook, state, metrics={"probe_r_epi_sasa": 0.14})
    evaluation = evaluate_program_goals(
        playbook, state, bundle, journal_path=tmp_path / "journal.jsonl"
    )
    apply_program_evaluation(state, evaluation)
    assert state.program_status == evaluation.status
    assert state.program_progress is not None
    state_path = tmp_path / "state.json"
    state.save(state_path)
    restored = TrainerState.load(state_path, playbook)
    assert restored.program_status == state.program_status
