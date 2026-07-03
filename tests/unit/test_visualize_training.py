"""Tests for v6 training visualization helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_load_metrics_and_discover_checkpoints(tmp_path: Path) -> None:
    from experiments.training.v6.visualize_training import discover_filmstrip_checkpoints, load_metrics

    metrics = [
        {
            "global_epoch": 1,
            "score": 1.2,
            "checkpoint_eligible": True,
            "health": {"probe_r_depth_sasa": 0.35, "cone_range_mean": 0.5},
            "losses": {"total": 9.0, "routing_entropy": 1.38},
        },
        {
            "global_epoch": 2,
            "score": 0.8,
            "checkpoint_eligible": False,
            "health": {"probe_r_depth_sasa": -0.4, "cone_range_mean": 1.2},
            "losses": {"total": 8.0, "routing_entropy": 1.39},
        },
    ]
    (tmp_path / "metrics.json").write_text(json.dumps(metrics))
    loaded = load_metrics(tmp_path)
    assert len(loaded) == 2
    assert loaded[0]["checkpoint_eligible"] is True

    epoch_dir = tmp_path / "epochs"
    epoch_dir.mkdir()
    (epoch_dir / "epoch_001.pt").write_bytes(b"stub")
    (tmp_path / "v6_best.pt").write_bytes(b"stub")
    ckpts = discover_filmstrip_checkpoints(tmp_path)
    assert any("ep1" in label for _, _, label in ckpts)
    assert any(label == "v6_best" for _, _, label in ckpts)


def test_plot_training_curves_writes_png(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from experiments.training.v6.visualize_training import plot_training_curves

    metrics = [
        {
            "global_epoch": i,
            "score": float(i),
            "checkpoint_eligible": i == 1,
            "health": {
                "probe_r_depth_sasa": 0.3 - i * 0.1,
                "probe_r_epi_sasa": 0.5,
                "cone_range_mean": 0.4 + i * 0.1,
            },
            "losses": {"total": 10 - i, "routing_entropy": 1.38},
        }
        for i in range(1, 4)
    ]
    out = tmp_path / "curves.png"
    plot_training_curves(metrics, out, "test_run")
    assert out.is_file()
    assert out.stat().st_size > 500
