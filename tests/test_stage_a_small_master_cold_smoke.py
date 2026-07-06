"""P_MASTER_COLD_SMOKE — one-epoch MASTER cold-start before full curriculum."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from science.training.mlflow_governance import corpus_fold_ids, corpus_manifest_hash
from science.training.stage_a_small_master_cold_smoke import (
    SMALL_CORPUS_MANIFEST,
    smoke_assertions_doc,
    validate_master_cold_smoke_run,
)
from science.training.stage_a_smoke import collect_run_artifact_names


def test_master_cold_smoke_assertions_doc() -> None:
    doc = smoke_assertions_doc()
    assert doc["gate_id"] == "P_MASTER_COLD_SMOKE"
    assert doc["structures"] == 12
    assert doc["make_target"] == "train-v6-stage-a-small-master-cold-smoke"
    assert doc["prerequisite"] == "make gate-p-feature-01"


def test_validate_master_cold_smoke_run_accepts_lineage() -> None:
    if not SMALL_CORPUS_MANIFEST.is_file():
        pytest.skip("small corpus manifest not committed")
    params = {
        "branch": "residue-only",
        "parent_run_id": "null",
        "warm_start": "none",
        "lineage_root": "true",
        "corpus_manifest_hash": corpus_manifest_hash(SMALL_CORPUS_MANIFEST),
        "corpus_size": "12",
        "curvature_mode": "free",
        "curvature_final": "0.7",
        "scale": "micro",
        "feature_set": "master_four_vector",
        "p_feature_01_passed": "true",
        "rho_def": "dehydron_wrapping_6.5A",
        "tau_def": "rho_lt_13.0",
        "ss_def": "biotite_psea_dssp_class",
        "sasa_def": "freesasa_heavy_atom_A2",
        "feature_module_sha256": "abc",
        "curriculum_schedule": "[]",
        "git_commit": "abc",
        "spec_version": "TRAINING_GOVERNANCE_AND_MLFLOW_SCHEMA:2026-07-02",
        "space_name": "gospconemapper_v6_hyp128",
    }
    from science.training.mlflow_governance import fold_id_to_mlflow_key

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
    }
    for fid in corpus_fold_ids(SMALL_CORPUS_MANIFEST):
        metric_keys.add(f"per_fold_loss.{fold_id_to_mlflow_key(fid)}")
    errors = validate_master_cold_smoke_run(
        params,
        metric_keys,
        {
            "poincare_disc_overlay.png",
            "angular_distribution_stats.json",
            "probe_curvature_sources.json",
        },
        full_corpus=True,
    )
    assert errors == [], errors


@pytest.mark.integration
def test_master_cold_smoke_mlflow_run_from_env() -> None:
    """After `make train-v6-stage-a-small-master-cold-smoke`, set MASTER_COLD_SMOKE_RUN_ID."""
    run_id = os.environ.get("MASTER_COLD_SMOKE_RUN_ID", "").strip()
    if not run_id:
        pytest.skip("Set MASTER_COLD_SMOKE_RUN_ID to a completed smoke MLflow run id")

    mlflow = pytest.importorskip("mlflow")
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    run = mlflow.get_run(run_id)
    params = dict(run.data.params)
    metric_keys = set(run.data.metrics.keys())
    artifact_names = collect_run_artifact_names(run_id, tracking_uri=tracking_uri)

    errors = validate_master_cold_smoke_run(params, metric_keys, artifact_names)
    assert errors == [], "P_MASTER_COLD_SMOKE failed:\n" + "\n".join(errors)
