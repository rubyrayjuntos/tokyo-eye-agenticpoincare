"""Tests for autonomous v6 training playbook and action selection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from experiments.training.v6.auto_trainer import (
    MetricSnapshot,
    TrainerState,
    build_next_suggested_payload,
    build_run_id,
    conditions_match,
    load_playbook,
    queue_next_suggested,
    select_action,
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


def test_load_playbook_rejects_unknown_priority(tmp_path: Path) -> None:
    raw = {
        "version": "1.0",
        "champion": {"run_id": "x", "checkpoint": "y.pt"},
        "priority": ["missing_action"],
        "actions": {},
    }
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.dump(raw))
    with pytest.raises(ValueError, match="unknown action"):
        load_playbook(path)


def test_select_touchup_from_gate_when_extended_stalled(playbook: dict, state: TrainerState) -> None:
    state.last_action_id = "touchup_extended"
    snap = MetricSnapshot(
        focus={
            "primary_focus": ["probe_r_epi_sasa"],
            "primary_focus_str": "probe_r_epi_sasa",
            "needs_work": [{"metric": "probe_r_epi_sasa", "value": 0.14, "matters": True}],
        },
        assess={"promotion_gate": {"passed": True}, "moe": {"routing_entropy_mean": 1.04}},
        champion_score=3.2,
        routing_entropy=1.04,
        probe_r_epi_sasa=0.14,
        promotion_gate_passed=True,
        expert_starvation_count=0,
    )
    selected = select_action(playbook, snap, state)
    assert selected is not None
    action_id, _ = selected
    assert action_id == "touchup_from_gate"


def test_select_touchup_extended_without_gate_best(playbook: dict, state: TrainerState) -> None:
    state.gate_run_id = "missing_gate_run"
    snap = MetricSnapshot(
        focus={
            "primary_focus": ["probe_r_epi_sasa"],
            "primary_focus_str": "probe_r_epi_sasa",
            "needs_work": [{"metric": "probe_r_epi_sasa", "value": 0.14, "matters": True}],
        },
        assess={"promotion_gate": {"passed": True}, "moe": {"routing_entropy_mean": 1.04}},
        champion_score=3.2,
        routing_entropy=1.04,
        probe_r_epi_sasa=0.14,
        promotion_gate_passed=True,
        expert_starvation_count=0,
    )
    selected = select_action(playbook, snap, state)
    assert selected is not None
    action_id, _ = selected
    assert action_id == "touchup_extended"


def test_gate_promotion_when_routing_diffuse(playbook: dict, state: TrainerState) -> None:
    snap = MetricSnapshot(
        focus={"primary_focus": [], "primary_focus_str": "none"},
        assess={"promotion_gate": {"passed": True}, "moe": {"routing_entropy_mean": 1.35}},
        champion_score=3.0,
        routing_entropy=1.35,
        probe_r_epi_sasa=0.5,
        promotion_gate_passed=True,
        expert_starvation_count=1,
    )
    selected = select_action(playbook, snap, state)
    assert selected is not None
    assert selected[0] == "gate_promotion"


def test_conditions_require_gate_best(playbook: dict, state: TrainerState) -> None:
    state.gate_run_id = "nonexistent_gate_run"
    snap = MetricSnapshot(
        focus={
            "primary_focus": ["probe_r_epi_sasa"],
            "primary_focus_str": "probe_r_epi_sasa",
        },
        assess=None,
        champion_score=3.0,
        routing_entropy=1.04,
        probe_r_epi_sasa=0.14,
        promotion_gate_passed=True,
        expert_starvation_count=0,
    )
    action = playbook["actions"]["touchup_from_gate"]
    assert conditions_match(action["when"], snap, state) is False


def test_build_run_id_prefix() -> None:
    run_id = build_run_id("auto_touchup_ext")
    assert run_id.startswith("auto_touchup_ext_")


def test_metric_snapshot_reads_focus(tmp_path: Path) -> None:
    run_dir = tmp_path / "my_run"
    run_dir.mkdir()
    focus = {
        "primary_focus_str": "probe_r_epi_sasa",
        "healthy": [{"metric": "routing_entropy", "value": 1.03}],
        "needs_work": [{"metric": "probe_r_epi_sasa", "value": 0.12}],
    }
    (run_dir / "focus_summary.json").write_text(json.dumps(focus))
    snap = MetricSnapshot.from_run_dir(run_dir)
    assert snap.routing_entropy == pytest.approx(1.03)
    assert snap.probe_r_epi_sasa == pytest.approx(0.12)


def test_build_next_suggested_payload(playbook: dict, state: TrainerState) -> None:
    snap = MetricSnapshot(
        focus={"primary_focus_str": "probe_r_epi_sasa"},
        assess=None,
        champion_score=3.2,
        routing_entropy=1.04,
        probe_r_epi_sasa=0.14,
        promotion_gate_passed=True,
        expert_starvation_count=0,
    )
    state.iteration = 3
    action = playbook["actions"]["touchup_from_gate"]
    payload = build_next_suggested_payload("touchup_from_gate", action, snap, state=state)
    assert payload["action_id"] == "touchup_from_gate"
    assert payload["queued_for_iteration"] == 4
    assert payload["make_target"] == "train-v6-p4-corpus25-touchup"


def test_queue_flags_stall_after_repeat_unpromoted(
    playbook: dict, state: TrainerState, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    journal = tmp_path / "journal.jsonl"
    journal.write_text("")
    monkeypatch.setattr("experiments.training.v6.auto_trainer._JOURNAL_PATH", journal)
    state.program_progress = {
        "best_metrics": {},
        "iterations_since_improvement": {"probe_r_epi_sasa": 0},
        "consecutive_unpromoted": 0,
        "consecutive_critical_promotions": 0,
        "regime_attempts": {},
        "total_train_iterations": 0,
    }
    state.iteration = 2
    state.last_action_id = "touchup_extended"
    state.unpromoted_same_action = 1
    snap = MetricSnapshot(
        focus={
            "primary_focus": ["probe_r_epi_sasa"],
            "primary_focus_str": "probe_r_epi_sasa",
        },
        assess={"promotion_gate": {"passed": True}},
        champion_score=3.2,
        routing_entropy=1.04,
        probe_r_epi_sasa=0.14,
        promotion_gate_passed=True,
        expert_starvation_count=0,
    )
    queue_next_suggested(playbook, state, promoted=False, snap=snap)
    assert state.next_suggested is not None
    assert state.next_suggested["action_id"] == "touchup_from_gate"
    assert state.stalled is False

    state.last_action_id = "touchup_from_gate"
    state.unpromoted_same_action = 1
    queue_next_suggested(playbook, state, promoted=False, snap=snap)
    assert state.stalled is True
    assert state.stall_reason
