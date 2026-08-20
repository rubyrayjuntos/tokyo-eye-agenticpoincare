"""Unit tests for B1 teleconnections Pass verdict logic."""

from __future__ import annotations

from experiments.diagnostics.b1_teleconnections_grade import OFF_NOISE_FLOOR, _verdict


def _arm(as_n12: float, as_scr: float, *, finite: bool = True) -> dict:
    return {
        "allele_sens_n12": as_n12,
        "allele_sens_scramble": as_scr,
        "finite": finite,
    }


def test_verdict_pass_relative() -> None:
    v = _verdict(_arm(0.01, 0.005), _arm(0.04, 0.01))
    assert v["pass"] is True


def test_verdict_fail_on_not_gt_off() -> None:
    v = _verdict(_arm(0.05, 0.01), _arm(0.02, 0.01))
    assert v["pass"] is False
    assert v["bars"]["on_conduit_gt_off_conduit"]["pass"] is False


def test_verdict_off_near_zero_waiver() -> None:
    v = _verdict(_arm(OFF_NOISE_FLOOR * 0.5, 0.01), _arm(0.02, 0.01))
    assert v["bars"]["off_conduit_gt_off_scramble_or_near_zero_waiver"]["near_zero_waiver"]
    assert v["bars"]["off_conduit_gt_off_scramble_or_near_zero_waiver"]["pass"] is True
    assert v["pass"] is True
