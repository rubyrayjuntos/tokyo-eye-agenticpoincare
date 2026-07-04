"""P_STOP_ENFORCEMENT — inference-mode stop halts on trip, not on clean runs."""

from __future__ import annotations

from pathlib import Path

import pytest

from science.training.config import TrainingConfig
from science.training.stage_a_stop import (
    STAGE_A_STOP_CANARY_PDB,
    check_stage_a_inference_stop,
    format_stop_message,
    should_enforce_stage_a_stop,
)


def _healthy_infer() -> dict[str, float]:
    return {
        "effective_experts": 3.8,
        "effective_experts_min": 3.2,
        "min_routing_fraction": 0.18,
        f"eval_min_routing_fraction.{STAGE_A_STOP_CANARY_PDB}": 0.12,
        "eval_min_routing_fraction.1CRN": 0.21,
    }


def test_p_stop_enforcement_healthy_inference_does_not_trip() -> None:
    verdict = check_stage_a_inference_stop(_healthy_infer())
    assert not verdict.tripped
    assert verdict.reasons == ()


def test_p_stop_enforcement_inference_min_r_below_floor_trips() -> None:
    routing = _healthy_infer()
    routing["min_routing_fraction"] = 0.049
    routing["eval_min_routing_fraction.1PGB"] = 0.049
    verdict = check_stage_a_inference_stop(routing)
    assert verdict.tripped
    assert any("min_routing_fraction" in r for r in verdict.reasons)
    assert any("canary 1PGB" in r for r in verdict.reasons)


def test_p_stop_enforcement_effective_experts_min_trips() -> None:
    routing = _healthy_infer()
    routing["effective_experts_min"] = 2.4
    verdict = check_stage_a_inference_stop(routing)
    assert verdict.tripped
    assert any("effective_experts_min" in r for r in verdict.reasons)


def test_p_stop_enforcement_train_dropout_zeros_do_not_trip_without_infer() -> None:
    """Stop reads inference_routing only — train min_r=0 is not an input here."""
    verdict = check_stage_a_inference_stop(_healthy_infer())
    assert not verdict.tripped


def test_p_stop_enforcement_message_includes_reasons() -> None:
    routing = _healthy_infer()
    routing["min_routing_fraction"] = 0.04
    routing["eval_min_routing_fraction.1PGB"] = 0.04
    verdict = check_stage_a_inference_stop(routing)
    msg = format_stop_message(verdict)
    assert "Stage A stop enforced" in msg
    assert "0.04" in msg


def test_should_enforce_stage_a_stop_locked_manifest_phase2() -> None:
    cfg = TrainingConfig(
        corpus_manifest=Path("manifests/v6_corpus_stage_a.json"),
        enforce_stage_a_stop=True,
    )
    assert should_enforce_stage_a_stop(cfg, 2) is True


def test_should_enforce_stage_a_stop_not_phase1() -> None:
    cfg = TrainingConfig(
        corpus_manifest=Path("manifests/v6_corpus_stage_a.json"),
        enforce_stage_a_stop=True,
    )
    assert should_enforce_stage_a_stop(cfg, 1) is False


def test_should_enforce_stage_a_stop_disabled_flag() -> None:
    cfg = TrainingConfig(
        corpus_manifest=Path("manifests/v6_corpus_stage_a.json"),
        enforce_stage_a_stop=False,
    )
    assert should_enforce_stage_a_stop(cfg, 2) is False


def test_should_enforce_stage_a_stop_non_locked_manifest() -> None:
    cfg = TrainingConfig(
        corpus_manifest=Path("manifests/v6_corpus_disc_target.json"),
        enforce_stage_a_stop=True,
    )
    assert should_enforce_stage_a_stop(cfg, 2) is False
