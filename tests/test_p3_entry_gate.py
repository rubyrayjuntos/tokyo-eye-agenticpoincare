"""Property tests for P3_ENTRY_GATE (routing stability before gate-locked P3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from science.training.checkpoint_score import ROUTING_ENTROPY_SAVE_MAX
from science.training.p3_entry_gate import (
    DEFAULT_CONSECUTIVE_EPOCHS,
    GATE_NAME,
    p3_entry_gate_passed,
    p3_entry_gate_verdict,
)


def test_p3_entry_gate_name() -> None:
    assert GATE_NAME == "P3_ENTRY_GATE"


def test_p3_entry_gate_accepts_sustained_stable_routing() -> None:
    """Synthetic stable P2: entropy holds under ceiling → P3 may start."""
    stable = [1.15] * DEFAULT_CONSECUTIVE_EPOCHS
    verdict = p3_entry_gate_verdict(stable)
    assert verdict.passed is True
    assert verdict.max_consecutive_below_ceiling >= DEFAULT_CONSECUTIVE_EPOCHS


def test_p3_entry_gate_rejects_single_epoch_graze() -> None:
    """One trough at the ceiling (ep-97 pattern) must not qualify."""
    graze = [1.32, 1.28, 1.30, 1.2095, 1.34, 1.31]
    verdict = p3_entry_gate_verdict(graze)
    assert verdict.passed is False
    assert verdict.max_consecutive_below_ceiling == 1


def test_p3_entry_gate_rejects_limit_cycle_band() -> None:
    """Oscillation straddling 1.21 — no N-consecutive below."""
    cycle = []
    for i in range(30):
        cycle.append(1.28 if i % 2 == 0 else 1.24)
    verdict = p3_entry_gate_verdict(cycle)
    assert verdict.passed is False


def test_p3_entry_gate_n_rejects_almost_stable() -> None:
    """Four consecutive below is insufficient when N=5."""
    almost = [1.30, 1.18, 1.17, 1.16, 1.15, 1.32]
    assert p3_entry_gate_passed(almost, consecutive_epochs=5) is False
    assert p3_entry_gate_passed(almost, consecutive_epochs=4) is True


def test_p3_entry_gate_n6_uses_scaled_ceiling() -> None:
    """N=6 stable routing at H≈1.55 must not be judged against N=4's 1.21 bar."""
    from science.training.routing_gate_bounds import routing_entropy_save_ceiling

    ceiling_n6 = routing_entropy_save_ceiling(num_experts=6)
    stable_n6 = [1.55] * DEFAULT_CONSECUTIVE_EPOCHS
    assert p3_entry_gate_passed(stable_n6, ceiling=ceiling_n6) is True
    assert p3_entry_gate_passed(stable_n6, ceiling=ROUTING_ENTROPY_SAVE_MAX) is False


@pytest.mark.integration
def test_p3_entry_gate_rejects_master_cold_p2_history() -> None:
    """This run's P2 history: only epoch 97 below 1.21 → P3 correctly blocked."""
    metrics_path = Path("checkpoints/v6/runs/stage_a_small_master_cold_v1/metrics.json")
    if not metrics_path.is_file():
        pytest.skip("master cold metrics not present")
    metrics = json.loads(metrics_path.read_text())
    p2_h = [
        float(row["losses"]["routing_entropy"])
        for row in metrics
        if row.get("phase") == 2
    ]
    assert len(p2_h) >= 100
    below = sum(1 for h in p2_h if h < 1.21)
    assert below == 1
    verdict = p3_entry_gate_verdict(p2_h)
    assert verdict.passed is False
    assert verdict.max_consecutive_below_ceiling == 1
