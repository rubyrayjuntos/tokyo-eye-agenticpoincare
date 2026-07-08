"""Tests for τ-rim shell signal gate and split-screen viewer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from science.dtie.common.interfaces import GNNNodeOutput
from science.dtie.v6.visualization.interactive_viewer import (
    disc_payload_from_nodes,
    residue_metrics_payload,
    write_split_screen_viewer_html,
)
from science.dtie.v6.visualization.shell_signal_gate import evaluate_shell_signal_ssot


def _node(
    i: int,
    *,
    epistemic: float,
    aleatoric: float,
    depth: float,
    disc_r: float,
    tau: float,
) -> GNNNodeOutput:
    angle = i / 40.0 * 2 * np.pi
    return GNNNodeOutput(
        residue_index=i + 1,
        chain_label="A",
        input_features=np.array([12.0, tau, 1.0, 0.3], dtype=np.float64),
        projections=np.zeros(8),
        cone_depth=depth,
        cone_width=0.5,
        epistemic_uncertainty=epistemic,
        aleatoric_uncertainty=aleatoric,
        hyp_projections=np.array([disc_r * np.cos(angle), disc_r * np.sin(angle)]),
        expert_weights=np.array([0.7, 0.1, 0.1, 0.1]),
    )


def test_shell_gate_passes_with_spread_and_tau_rim() -> None:
    nodes: list[GNNNodeOutput] = []
    for i in range(40):
        disc_r = 0.15 + 0.55 * (i / 39)
        depth = 0.15 + 0.65 * (i / 39)
        tau = 1.0 if depth >= 0.45 else 0.0
        nodes.append(
            _node(
                i,
                epistemic=0.1 + 0.05 * (i % 7),
                aleatoric=0.08 + 0.04 * (i % 5),
                depth=depth,
                disc_r=disc_r,
                tau=tau,
            )
        )
    verdict = evaluate_shell_signal_ssot(
        nodes, structure_id="test", structural_disc_frozen=True
    )
    assert verdict.passed is True
    assert verdict.metrics["epistemic_std"] > 0.01
    assert verdict.metrics["r_disc_r_depth"] > 0.25


def test_shell_gate_fails_flat_epistemic() -> None:
    nodes = [
        _node(
            i,
            epistemic=1.5,
            aleatoric=0.2 + 0.01 * i,
            depth=0.2 + 0.6 * (i / 19),
            disc_r=0.2 + 0.5 * (i / 19),
            tau=float(i % 2),
        )
        for i in range(20)
    ]
    verdict = evaluate_shell_signal_ssot(nodes, structure_id="flat", structural_disc_frozen=True)
    assert verdict.passed is False
    assert any("flat epistemic" in f for f in verdict.failures)


def test_write_split_screen_viewer_html(tmp_path: Path) -> None:
    nodes = [
        _node(
            i,
            epistemic=0.5 + i * 0.02,
            aleatoric=0.1 + i * 0.01,
            depth=float(i),
            disc_r=0.1 * i,
            tau=0.0,
        )
        for i in range(1, 8)
    ]
    pdb = (
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  0.50 88.00           C\n"
        "END\n"
    )
    out = tmp_path / "4obe_split_viewer.html"
    write_split_screen_viewer_html(
        structure_id="4obe",
        pdb_text=pdb,
        points=disc_payload_from_nodes(nodes),
        residue_metrics=residue_metrics_payload(nodes),
        model_version="GOSPConeMapper-v6",
        output_path=out,
        disc_layout_label="structural SSOT — test",
    )
    text = out.read_text(encoding="utf-8")
    assert "panel-3d" in text
    assert "panel-disc" in text
    assert "NGL.Stage" in text
    assert "hoverKey" in text
    assert "selectedKey" in text
    assert "viewScale" in text
    assert "structural SSOT" in text
