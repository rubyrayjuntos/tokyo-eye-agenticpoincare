"""P_MLFLOW_01 — mandatory MLflow governance schema."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from science.training.config import TrainingConfig
from science.training.mlflow_governance import (
    MANDATORY_ARTIFACTS,
    MANDATORY_METRICS,
    MANDATORY_PARAMS,
    build_governance_params,
    corpus_fold_ids,
    export_disc_governance_artifacts,
    finalize_governance_run,
    fold_id_to_mlflow_key,
    governance_epoch_metrics,
    validate_finished_run,
)
from science.training.tracking import TrainingTracker


class _StubModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.log_c = nn.Parameter(torch.tensor(0.5))

    @property
    def curvature(self) -> torch.Tensor:
        import torch.nn.functional as F

        return F.softplus(self.log_c) + 1e-4


def test_build_governance_params_manifest_hash() -> None:
    cfg = TrainingConfig(
        corpus_manifest="manifests/v6_corpus_disc_target.json",
        resume=Path("checkpoints/v6/runs/lever_a_clean_slate_v1/v6_best_disc.pt"),
    )
    params = build_governance_params(cfg, proteins=[{"pdb_id": "11QE"}] * 3)
    assert params["branch"] == "residue-only"
    assert params["curvature_mode"] == "warm_start"
    assert params["space_name"] == "gospconemapper_v6_hyp128"
    assert len(params["corpus_manifest_hash"]) == 64
    assert params["corpus_size"] == "3"
    assert params["spec_version"].startswith("TRAINING_GOVERNANCE")


def test_governance_epoch_metrics_maps_shell_probes() -> None:
    model = _StubModel()
    health = {
        "disc_sigma2_sigma1_mean": 0.665,
        "disc_line_thickness_pre_mean": 0.219,
        "probe_r_depth_sasa": 0.73,
        "probe_r_epi_sasa": 0.78,
    }
    losses = {
        "effective_experts": 3.9,
        "effective_experts_min": 3.5,
        "min_routing_fraction": 0.18,
        "per_fold_loss.3_40_50_300": 1.2,
        "per_fold_loss.3_80_20_20": 1.1,
    }
    metrics = governance_epoch_metrics(health, losses, model)
    assert metrics["sigma2_sigma1"] == pytest.approx(0.665)
    assert metrics["disc_thick"] == pytest.approx(0.219)
    assert metrics["r_d_s"] == pytest.approx(0.73)
    assert metrics["r_e_s"] == pytest.approx(0.78)
    assert "log_c" in metrics
    assert metrics["effective_experts"] == pytest.approx(3.9)
    assert "stage_gate_passed" in metrics


def test_governance_epoch_metrics_prefers_inference_routing() -> None:
    model = _StubModel()
    health = {
        "disc_sigma2_sigma1_mean": 0.665,
        "disc_line_thickness_pre_mean": 0.219,
        "probe_r_depth_sasa": 0.73,
        "probe_r_epi_sasa": 0.78,
    }
    train_losses = {
        "effective_experts": 3.7,
        "effective_experts_min": 0.0,
        "min_routing_fraction": 0.0,
        "per_fold_loss.3_40_50_300": 1.2,
        "per_fold_loss.3_80_20_20": 1.1,
    }
    infer_routing = {
        "effective_experts": 3.9,
        "effective_experts_min": 3.5,
        "min_routing_fraction": 0.18,
        "eval_min_routing_fraction.1PGB": 0.06,
    }
    metrics = governance_epoch_metrics(
        health, train_losses, model, inference_routing=infer_routing
    )
    assert metrics["min_routing_fraction"] == pytest.approx(0.18)
    assert metrics["train_min_routing_fraction"] == pytest.approx(0.0)
    assert metrics["stage_gate_passed"] == pytest.approx(1.0)
    assert metrics["eval_min_routing_fraction.1PGB"] == pytest.approx(0.06)


def test_p_mlflow_01_schema_on_finished_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mlflow = pytest.importorskip("mlflow")

    from science.training.tracking import TrainingTracker

    tracking_uri = (tmp_path / "mlruns").as_uri()
    cfg = TrainingConfig(
        output_dir=tmp_path / "run",
        corpus_manifest=Path("manifests/v6_corpus_disc_target.json"),
        pdb_dir=Path("/tmp/dtie_pdb_cache"),
        mlflow_tracking_uri=tracking_uri,
        mlflow_experiment="test-governance",
        device="cpu",
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    model = _StubModel()
    proteins = [{"pdb_id": "11QE", "gene": "KRAS"}]

    tracker = TrainingTracker(cfg, run_name="p_mlflow_01")

    def _fake_export(
        model: nn.Module,
        config: TrainingConfig,
        out_dir: Path,
        *,
        device: str = "cpu",
    ) -> dict[str, Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "poincare_disc_overlay.png": out_dir / "poincare_disc_overlay.png",
            "angular_distribution_stats.json": out_dir / "angular_distribution_stats.json",
            "probe_curvature_sources.json": out_dir / "probe_curvature_sources.json",
        }
        paths["poincare_disc_overlay.png"].write_bytes(b"png")
        paths["angular_distribution_stats.json"].write_text(
            json.dumps({"angular_distribution_stats": []})
        )
        paths["probe_curvature_sources.json"].write_text(json.dumps({"model_checkpoint_c": 0.7}))
        return paths

    monkeypatch.setattr(
        "science.training.mlflow_governance.export_disc_governance_artifacts",
        _fake_export,
    )

    with tracker.start_run(proteins=proteins, phases=[]):
        health = {
            "disc_sigma2_sigma1_mean": 0.665,
            "disc_line_thickness_pre_mean": 0.219,
            "probe_r_depth_sasa": 0.73,
            "probe_r_epi_sasa": 0.78,
        }
        losses = {
            "effective_experts": 3.9,
            "effective_experts_min": 3.5,
            "min_routing_fraction": 0.18,
            "per_fold_loss.3_40_50_300": 1.0,
            "per_fold_loss.3_80_20_20": 1.0,
        }
        tracker.log_metrics(governance_epoch_metrics(health, losses, model), step=1)
        finalize_governance_run(tracker, model, cfg, proteins, device="cpu")

    assert tracker.run_id is not None
    run = mlflow.get_run(tracker.run_id)
    params = dict(run.data.params)
    metric_keys = set(run.data.metrics.keys())
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    artifact_paths = {
        Path(a.path).name
        for a in client.list_artifacts(tracker.run_id)
    }
    for art in client.list_artifacts(tracker.run_id, path="governance"):
        artifact_paths.add(Path(art.path).name)
    for art in client.list_artifacts(tracker.run_id, path="governance/flat"):
        artifact_paths.add(Path(art.path).name)

    errors = validate_finished_run(
        params,
        metric_keys,
        artifact_paths,
        manifest_path=cfg.corpus_manifest,
    )
    assert not errors, errors
    assert MANDATORY_PARAMS <= set(params)
    assert MANDATORY_METRICS <= metric_keys
    assert MANDATORY_ARTIFACTS <= artifact_paths
    assert params["curvature_final"]
    for fid in corpus_fold_ids(cfg.corpus_manifest):
        assert f"per_fold_loss.{fold_id_to_mlflow_key(fid)}" in metric_keys


def test_export_disc_governance_artifacts_requires_structure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Artifact export is integration-tested when PDB cache is present."""
    pytest.importorskip("matplotlib")
    cfg = TrainingConfig(
        output_dir=tmp_path,
        pdb_dir=Path("/tmp/dtie_pdb_cache"),
        disc_scatter_structure="11QE:A",
        device="cpu",
    )
    ckpt = Path("checkpoints/v6/runs/lever_a_clean_slate_v1/v6_best_disc.pt")
    if not ckpt.is_file():
        pytest.skip("lever_a checkpoint not present")
    from experiments.training.v6.assess_checkpoint import load_v6_model

    model = load_v6_model(ckpt, "cpu")
    paths = export_disc_governance_artifacts(model, cfg, tmp_path / "gov", device="cpu")
    assert paths["poincare_disc_overlay.png"].is_file()
    stats = json.loads(paths["angular_distribution_stats.json"].read_text())
    assert "angular_distribution_stats" in stats
