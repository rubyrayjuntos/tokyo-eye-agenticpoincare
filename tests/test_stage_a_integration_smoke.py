"""P_STAGE_A_SMOKE — one-epoch assembled-stack validation before full Stage A curriculum."""

from __future__ import annotations

from pathlib import Path

import pytest

from science.training.mlflow_governance import corpus_manifest_hash
from science.training.stage_a_smoke import (
    LOCKED_CORPUS_MANIFEST,
    collect_run_artifact_names,
    smoke_assertions_doc,
    validate_stage_a_smoke_run,
)


def test_stage_a_smoke_assertions_doc() -> None:
    doc = smoke_assertions_doc()
    assert doc["gate_id"] == "P_STAGE_A_SMOKE"
    assert "v6_corpus_stage_a.json" in doc["corpus_manifest"]
    assert doc["make_target"] == "train-v6-stage-a-smoke"


def test_validate_stage_a_smoke_run_accepts_per_fold_loss() -> None:
    if not LOCKED_CORPUS_MANIFEST.is_file():
        pytest.skip("locked manifest not committed")
    params = {
        "branch": "residue-only",
        "parent_run_id": "cold_start",
        "corpus_manifest_hash": corpus_manifest_hash(LOCKED_CORPUS_MANIFEST),
        "corpus_size": "5",
        "curvature_mode": "warm_start",
        "curvature_final": "0.7",
        "scale": "micro",
        "feature_set": "dehydron-only",
        "curriculum_schedule": "[]",
        "git_commit": "abc",
        "spec_version": "TRAINING_GOVERNANCE_AND_MLFLOW_SCHEMA:2026-07-02",
        "space_name": "gospconemapper_v6_hyp128",
    }
    metric_keys = {
        "log_c",
        "effective_experts",
        "effective_experts_min",
        "min_routing_fraction",
        "sigma2_sigma1",
        "disc_thick",
        "r_d_s",
        "r_e_s",
        "stage_gate_passed",
        "per_fold_loss.3_40_50_300",
        "per_fold_loss.3_80_20_20",
    }
    errors = validate_stage_a_smoke_run(
        params,
        metric_keys,
        {},
        {
            "poincare_disc_overlay.png",
            "angular_distribution_stats.json",
            "probe_curvature_sources.json",
        },
        min_per_fold_loss_keys=2,
    )
    assert errors == [], errors


def test_validate_stage_a_smoke_run_rejects_per_family_loss() -> None:
    if not LOCKED_CORPUS_MANIFEST.is_file():
        pytest.skip("locked manifest not committed")
    params = {
        "corpus_manifest_hash": "dead",
        "curvature_final": "0.7",
    }
    errors = validate_stage_a_smoke_run(
        params,
        {"per_family_loss.gtpase", "effective_experts"},
        {},
        set(),
    )
    assert any("per_family_loss" in e for e in errors)


@pytest.mark.integration
def test_stage_a_smoke_mlflow_run_from_env() -> None:
    """After `make train-v6-stage-a-smoke`, set STAGE_A_SMOKE_RUN_ID to verify the run."""
    import os

    run_id = os.environ.get("STAGE_A_SMOKE_RUN_ID", "").strip()
    if not run_id:
        pytest.skip("Set STAGE_A_SMOKE_RUN_ID to a completed smoke MLflow run id")

    mlflow = pytest.importorskip("mlflow")
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    run = mlflow.get_run(run_id)
    params = dict(run.data.params)
    metric_keys = set(run.data.metrics.keys())

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    artifact_names = collect_run_artifact_names(run_id, tracking_uri=tracking_uri)

    errors = validate_stage_a_smoke_run(params, metric_keys, dict(run.data.metrics), artifact_names)
    assert errors == [], "P_STAGE_A_SMOKE failed:\n" + "\n".join(errors)
