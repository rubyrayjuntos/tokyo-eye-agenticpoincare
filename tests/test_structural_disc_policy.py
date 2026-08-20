"""Unit tests for structural disc SSOT policy resolution."""

from __future__ import annotations

from pathlib import Path

import torch

from science.dtie.common.structural_disc_policy import (
    read_checkpoint_structural_disc_frozen,
    resolve_structural_disc_frozen,
)


def test_job_params_override_wins(tmp_path: Path) -> None:
    ckpt = tmp_path / "m.pt"
    torch.save({"training_config": {"structural_disc_frozen": False}}, ckpt)
    frozen, reason = resolve_structural_disc_frozen(
        ckpt,
        pipeline_config_value=False,
        job_params={"structural_disc_frozen": True},
    )
    assert frozen is True
    assert reason == "job_params"


def test_checkpoint_master_cold_disables_ssot(tmp_path: Path) -> None:
    ckpt = tmp_path / "m.pt"
    torch.save(
        {
            "training_config": {
                "master_cold_lineage": True,
                "structural_disc_frozen": False,
            }
        },
        ckpt,
    )
    frozen, reason = resolve_structural_disc_frozen(ckpt)
    assert frozen is False
    assert reason == "checkpoint_training_config"


def test_checkpoint_slim_enables_ssot(tmp_path: Path) -> None:
    ckpt = tmp_path / "m.pt"
    torch.save(
        {"training_config": {"slim_moe_structural_ssot": True}},
        ckpt,
    )
    assert read_checkpoint_structural_disc_frozen(ckpt) is True
    frozen, reason = resolve_structural_disc_frozen(ckpt)
    assert frozen is True
    assert reason == "checkpoint_training_config"


def test_unknown_checkpoint_falls_back_to_legacy_default(tmp_path: Path) -> None:
    ckpt = tmp_path / "m.pt"
    torch.save({"model_state_dict": {}}, ckpt)
    frozen, reason = resolve_structural_disc_frozen(
        ckpt, pipeline_config_value=True
    )
    assert frozen is True
    assert reason == "pipeline_config_or_legacy_default"
