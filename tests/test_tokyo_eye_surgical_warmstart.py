"""Tests for Tokyo Eye B′ surgical warm-start (deny Euc trunk)."""

from __future__ import annotations

from pathlib import Path

import torch

from science.tokyo_eye.surgical_warmstart import (
    DEFAULT_DENY_PREFIXES,
    filter_donor_state,
    surgical_warmstart_tokyo_eye,
)


def test_filter_donor_state_denies_convs_and_norms() -> None:
    state = {
        "node_emb.weight": torch.zeros(2),
        "convs.0.lin.weight": torch.zeros(2),
        "norms.0.weight": torch.zeros(2),
        "hyp_mp.layers.0.weight": torch.zeros(2),
        "radial_head.0.weight": torch.zeros(2),
        "gate.logit_scale": torch.zeros(1),
    }
    kept, denied = filter_donor_state(state)
    assert "node_emb.weight" in kept
    assert "radial_head.0.weight" in kept
    assert "gate.logit_scale" in kept
    assert "convs.0.lin.weight" not in kept
    assert "norms.0.weight" not in kept
    assert "hyp_mp.layers.0.weight" not in kept
    assert set(denied) >= {
        "convs.0.lin.weight",
        "norms.0.weight",
        "hyp_mp.layers.0.weight",
    }
    assert DEFAULT_DENY_PREFIXES == ("convs.", "norms.")


def test_surgical_warmstart_from_fix1_champion(tmp_path: Path) -> None:
    from experiments.training.v66.healthy_fix1 import FIX1_SPARSITY_CHAMPION_CKPT

    if not FIX1_SPARSITY_CHAMPION_CKPT.is_file():
        import pytest

        pytest.skip("Fix-1 sparsity champion not on disk")

    out = tmp_path / "v7_bprime.pt"
    report = surgical_warmstart_tokyo_eye(FIX1_SPARSITY_CHAMPION_CKPT, out_path=out)
    assert out.is_file()
    assert report.hyp_mp_primary is True
    assert report.se3_aux is False
    assert report.denied_keys > 0
    assert report.transferred_keys > 0

    blob = torch.load(out, map_location="cpu", weights_only=False)
    assert blob["global_epoch"] == 0
    assert blob["phase"] == 1
    assert blob["training_config"]["gnn_lineage"] == "v7"
    assert blob["architecture"]["hyp_mp_primary"] is True
    sd = blob["model_state_dict"]
    assert any(k.startswith("hyp_mp.") for k in sd)
    # Euc trunk may exist as freshly init modules, but donor transfer denied them
    # from overwriting — presence of hyp_mp + flags is the contract.
    assert blob["meta"]["denied_keys"] == report.denied_keys
