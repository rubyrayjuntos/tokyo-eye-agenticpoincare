"""Assembly gate ENFORCED — Architecture SSOT fail-closed checks.

Standing rule: prose without a failing test always loses to a convenient default.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from science.tokyo_eye.v8.assembly_gate import (
    AssemblyGateError,
    assert_claim_bearing_biology,
    assert_governed_assembly,
    assert_governed_frontend,
    assert_pool_frontend_dependencies,
    check_pool_frontend_dependencies,
    detect_edge_type_label_leakage,
)
from science.tokyo_eye.v8.biology_grad_sources import (
    biology_grad_by_source,
    dehydron_grad_by_source,
)
from science.tokyo_eye.v8.heads import MechanismScoreHead


def test_pool_frontend_dependencies_present_or_explicit() -> None:
    ok, missing = check_pool_frontend_dependencies()
    # In the sealed CI/dev env these should be present; if not, the assert
    # message must name missing deps (not look like an architecture veto).
    if not ok:
        with pytest.raises(AssemblyGateError, match="missing frontend dependencies"):
            assert_pool_frontend_dependencies()
        assert missing
    else:
        assert_pool_frontend_dependencies()


def test_governed_frontend_rejects_stub_without_allow() -> None:
    with pytest.raises(AssemblyGateError, match="equiformer_pool"):
        assert_governed_frontend(
            "stub",
            load_info={"backbone_mode": "live_se3_lite"},
            allow_off_path_frontend=False,
        )


def test_forbid_se3_lite_alone_is_not_pool() -> None:
    """Known-bad fixture: forbid_se3_lite without pool construction must fail.

    Adjacent tags must not stand in for a direct Equiformer-pool check
    (same class of gap as pure_hyp ball-detection heuristics).
    """
    with pytest.raises(AssemblyGateError, match="equiformer_pool"):
        assert_governed_frontend(
            "unknown",
            load_info={"forbid_se3_lite": True},
            allow_off_path_frontend=False,
        )
    with pytest.raises(AssemblyGateError, match="equiformer_pool"):
        assert_governed_frontend(
            "stub",
            load_info={"forbid_se3_lite": True, "backbone_mode": "something_else"},
            allow_off_path_frontend=False,
        )


def test_governed_frontend_allows_stub_with_explicit_flag() -> None:
    assert_governed_frontend(
        "stub",
        load_info={"backbone_mode": "live_se3_lite"},
        allow_off_path_frontend=True,
    )


def test_governed_frontend_accepts_pool() -> None:
    assert_governed_frontend(
        "equiformer_pool",
        load_info={"backbone_mode": "equiformer_v3_pool", "forbid_se3_lite": True},
        allow_off_path_frontend=False,
    )


def test_assembly_gate_off_path_skips_pure_hyp_not_as_pass() -> None:
    """Skip must not look like a clean pure_hyp seal."""
    result = assert_governed_assembly(
        frontend_kind="stub",
        load_info={"backbone_mode": "live_se3_lite"},
        allow_off_path_frontend=True,
        require_pure_hyp=True,
        check_deps=False,
    )
    assert result.passed
    assert result.allow_off_path_frontend is True
    assert result.pure_hyp_checked is False
    assert result.pure_hyp_ok is False  # never checked ≠ verified clean


def test_harness_rejects_stub_without_allow_subprocess() -> None:
    root = Path(__file__).resolve().parents[2]
    cmd = [
        sys.executable,
        str(root / "experiments/training/v8/run_v8_experiment.py"),
        "--smoke",
        "--no-mlflow",
        "--frontend",
        "stub",
        "--epochs",
        "1",
        "--device",
        "cpu",
        "--out-dir",
        str(root / "checkpoints/v8/runs"),
        "--run-name",
        "assembly_gate_stub_reject",
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        cwd=str(root),
        env={
            **dict(**{k: v for k, v in __import__("os").environ.items()}),
            "PYTHONPATH": str(root),
        },
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr
    blob = (proc.stderr + proc.stdout).lower()
    assert "assembly_gate" in blob or "equiformer_pool" in blob


def test_claim_bearing_refuses_leaking_batch_structurally() -> None:
    """Known-bad: still-leaking batch fails even if caller asserts non-leak."""
    import numpy as np
    import torch

    from science.tokyo_eye.v8.loader import dehydron_labels_from_edges

    n, e = 6, 4
    # Two R2 edges → nodes 0,1,2 labeled dehydron via incidence
    edge_index = np.array([[0, 1, 2, 3], [1, 2, 0, 4]], dtype=np.int64)
    edge_type = np.array([2, 2, 0, 0], dtype=np.int64)  # 2 = R2_DEHYDRON
    labels = dehydron_labels_from_edges(n, edge_index, edge_type)
    batch = {
        "edge_index": torch.tensor(edge_index),
        "edge_type": torch.tensor(edge_type),
        "dehydron_labels": torch.tensor(labels),
    }
    # Name-based args claim "non-leaking" — must not waive structural match.
    with pytest.raises(AssemblyGateError, match="defect B|reconstruct"):
        assert_claim_bearing_biology(
            batch=batch,
            biology_grad_sources_logged=True,
            edge_type_is_model_input=True,
            dehydron_or_sdrp_target_from_edge_type=False,
        )


def test_claim_bearing_flag_alone_without_batch_refuses() -> None:
    """Caller-asserted non-leak without a batch is not sufficient."""
    with pytest.raises(AssemblyGateError, match="structural batch required"):
        assert_claim_bearing_biology(
            biology_grad_sources_logged=True,
            edge_type_is_model_input=True,
            dehydron_or_sdrp_target_from_edge_type=False,
        )


def test_claim_bearing_refuses_missing_biology_grad_log() -> None:
    batch = {
        "edge_index": torch.zeros(2, 0, dtype=torch.long),
        "edge_type": torch.zeros(0, dtype=torch.long),
        "dehydron_labels": torch.zeros(4),
    }
    # Empty edges → labels zeros match reconstruction, but no R2 → not leaking
    # if labels are zeros and recon is zeros — that IS a match. Use mismatched.
    batch["dehydron_labels"] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    with pytest.raises(AssemblyGateError, match="biology_grad"):
        assert_claim_bearing_biology(
            batch=batch,
            biology_grad_sources_logged=False,
        )


def test_claim_bearing_passes_when_labels_differ_from_edge_type() -> None:
    """Non-leaking: dehydron labels are NOT R2-incidence of the input edges."""
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_type = torch.tensor([2, 2], dtype=torch.long)  # R2 present
    # Labels do not match incidence (which would mark 0,1,2)
    batch = {
        "edge_index": edge_index,
        "edge_type": edge_type,
        "dehydron_labels": torch.tensor([0.0, 0.0, 0.0]),
    }
    assert_claim_bearing_biology(batch=batch, biology_grad_sources_logged=True)


def test_claim_bearing_assembly_refuses_with_off_path() -> None:
    with pytest.raises(AssemblyGateError, match="allow-off-path"):
        assert_governed_assembly(
            frontend_kind="stub",
            allow_off_path_frontend=True,
            claim_bearing_biology=True,
            check_deps=False,
            require_pure_hyp=False,
            biology_grad_sources_logged=True,
            claim_batch={
                "edge_index": torch.zeros(2, 0, dtype=torch.long),
                "edge_type": torch.zeros(0, dtype=torch.long),
                "dehydron_labels": torch.tensor([1.0, 0.0]),
            },
        )


def test_detect_edge_type_label_leakage_on_loader_batch() -> None:
    from science.tokyo_eye.v8.loader import dehydron_labels_from_edges

    ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    et = torch.tensor([2, 2], dtype=torch.long)
    lab = dehydron_labels_from_edges(3, ei.numpy(), et.numpy())
    report = detect_edge_type_label_leakage(
        {
            "edge_index": ei,
            "edge_type": et,
            "dehydron_labels": torch.tensor(lab),
        }
    )
    assert report["leaking"] is True
    assert report["dehydron_matches_r2_incidence"] is True


def test_dehydron_biology_grad_is_euc_skip_dominated() -> None:
    """MechanismScoreHead(h_euc) → dehydron BCE must attribute to euc skip, not z_hyp."""
    torch.manual_seed(0)
    n, d = 8, 4
    z = torch.randn(n, d, requires_grad=True)
    h = torch.randn(n, d, requires_grad=True)
    labels = torch.zeros(n)
    labels[:3] = 1.0
    head = MechanismScoreHead(d)
    metrics = dehydron_grad_by_source(
        head, z_hyp=z, h_euc=h, dehydron_labels=labels
    )
    assert metrics["biology_grad_euc_skip"] > 0.0
    assert metrics["biology_grad_hyp"] == 0.0
    assert metrics["biology_grad_euc_share"] == pytest.approx(1.0)


def test_biology_grad_by_source_splits_both_paths() -> None:
    torch.manual_seed(1)
    z = torch.randn(5, 3, requires_grad=True)
    h = torch.randn(5, 3, requires_grad=True)

    def loss_fn(z_in: torch.Tensor, h_in: torch.Tensor) -> torch.Tensor:
        return z_in.pow(2).sum() + 2.0 * h_in.pow(2).sum()

    m = biology_grad_by_source(z_hyp=z, h_euc=h, loss_fn=loss_fn)
    assert m["biology_grad_hyp"] > 0.0
    assert m["biology_grad_euc_skip"] > 0.0
    assert abs(m["biology_grad_hyp_share"] + m["biology_grad_euc_share"] - 1.0) < 1e-5
