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
    assert params["warm_start"] == "resume"
    assert params["space_name"] == "gospconemapper_v6_hyp128"
    assert len(params["corpus_manifest_hash"]) == 64
    assert params["corpus_size"] == "3"
    assert params["spec_version"].startswith("TRAINING_GOVERNANCE")


def test_master_cold_lineage_governance_params() -> None:
    cfg = TrainingConfig(
        corpus_manifest="manifests/v6_corpus_stage_a_small_v1.json",
        master_cold_lineage=True,
    )
    params = build_governance_params(cfg, proteins=[{"pdb_id": "4OBE"}] * 12)
    assert params["warm_start"] == "none"
    assert params["parent_run_id"] == "null"
    assert params["curvature_mode"] == "free"
    assert params["lineage_root"] == "true"
    assert params["topology_only_gate"] == "true"
    assert params["v2_teacher"] == "disabled"
    assert params["feature_set"] == "master_four_vector"


def test_master_cold_tracker_skips_duplicate_topology_only_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Config logs topology_only_gate; governance must not re-log the same key."""
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    from science.training.config import apply_master_cold_dehydron_config
    from science.training.tracking import TrainingTracker

    tracking_uri = (tmp_path / "mlruns").as_uri()
    cfg = TrainingConfig(
        output_dir=tmp_path / "run",
        corpus_manifest=Path("manifests/v6_corpus_stage_a_small_v1.json"),
        pdb_dir=Path("/tmp/dtie_pdb_cache"),
        mlflow_tracking_uri=tracking_uri,
        mlflow_experiment="test-master-cold-dedup",
        device="cpu",
        master_cold_lineage=True,
    )
    cfg = apply_master_cold_dehydron_config(cfg)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    tracker = TrainingTracker(cfg, run_name="master_cold_dedup")
    with tracker.start_run(proteins=[{"pdb_id": "4OBE"}] * 12, phases=[]):
        pass
    assert tracker.run_id is not None
    run = mlflow.get_run(tracker.run_id)
    assert run.data.params["topology_only_gate"] == "true"
    assert run.data.params["v2_teacher"] == "disabled"


def test_build_governance_params_includes_num_experts() -> None:
    cfg = TrainingConfig(
        corpus_manifest="manifests/v6_corpus_stage_a_small_v1.json",
        num_experts=6,
        master_cold_lineage=True,
    )
    params = build_governance_params(cfg, proteins=[{"pdb_id": "4OBE"}] * 12)
    assert params["num_experts"] == "6"


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


def test_telemetry_track_metrics_maps_health_and_flags_dead_ale() -> None:
    from science.training.mlflow_governance import telemetry_track_metrics

    alive = telemetry_track_metrics(
        {
            "epistemic_std_mean": 4.4,
            "aleatoric_std_mean": 0.011,
            "probe_r_epi_ale": 0.977,
            "uncertainty_probe_alive_epi": 1.0,
            "uncertainty_probe_alive_ale": 1.0,
            "uncertainty_informative_ale": 0.0,
            "edge_telemetry_alive_fraction": 1.0,
            "same_expert_rate_mean": 0.784,
            "same_expert_null_rate_mean": 0.322,
        }
    )
    assert alive["track/epistemic_std"] == pytest.approx(4.4)
    assert alive["track/uncertainty_alive_epi"] == pytest.approx(1.0)
    assert alive["track/uncertainty_informative_ale"] == pytest.approx(0.0)
    assert "track/edge_resistance_corr" not in alive


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
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")

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
        **_ignored: object,
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


@pytest.mark.integration
def test_export_disc_governance_artifacts_requires_structure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Artifact export is integration-tested when PDB cache is present."""
    from tests.conftest import integration_db_skip_reason

    skip_reason = integration_db_skip_reason()
    if skip_reason:
        pytest.skip(skip_reason)
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


def test_export_passes_dehydron_barcode_flags_to_biology_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Governance must widen graphs with barcode sidecars when training used them."""
    captured: dict[str, object] = {}

    class _Bio:
        structure_id = "4OBE"
        chain = "A"
        disc_layout_source = "structural_ssot_frozen"
        disc_xy = [[0.1, 0.0], [0.2, 0.1]]
        disc_r = [0.1, 0.22]
        disc_theta_deg = [0.0, 26.5]
        rho = [0.1, 0.9]
        dehydron = [True, False]
        res_ids = ["A:1:", "A:2:"]
        cone_depth = [0.5, 0.6]
        n_residues = 2

    def _fake_load(pdb_id, chain, model, pdb_dir, device, **kwargs):
        captured.update(kwargs)
        captured["pdb_id"] = pdb_id
        return _Bio()

    monkeypatch.setattr(
        "experiments.diagnostics.crescent_biology_projection._load_biology_arrays",
        _fake_load,
    )
    monkeypatch.setattr(
        "experiments.diagnostics.crescent_biology_projection._plot_composite",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "experiments.diagnostics.crescent_biology_projection._compute_angular_stats",
        lambda bio: type("A", (), {"n_dehydron": 1, "n_non": 1})(),
    )
    monkeypatch.setattr(
        "experiments.diagnostics.crescent_biology_projection._angular_stats_to_dict",
        lambda ang: {"ok": True},
    )

    barcode_dir = tmp_path / "barcodes"
    barcode_dir.mkdir()
    cfg = TrainingConfig(
        output_dir=tmp_path / "run",
        pdb_dir=tmp_path / "pdb",
        disc_scatter_structure="4OBE:A",
        use_dehydron_barcode=True,
        use_binned_dehydron=False,
        dehydron_barcode_dir=barcode_dir,
        device="cpu",
    )
    model = _StubModel()
    paths = export_disc_governance_artifacts(model, cfg, tmp_path / "gov", device="cpu")
    assert captured["use_dehydron_barcode"] is True
    assert captured["use_binned_dehydron"] is False
    assert Path(captured["dehydron_barcode_dir"]) == barcode_dir
    assert "poincare_disc_overlay.png" in paths
    assert "angular_distribution_stats.json" in paths
    assert "probe_curvature_sources.json" in paths


def test_export_skips_overlay_on_node_dim_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*_a, **_k):
        raise RuntimeError("mat1 and mat2 shapes cannot be multiplied (491x4 and 15x128)")

    monkeypatch.setattr(
        "experiments.diagnostics.crescent_biology_projection._load_biology_arrays",
        _boom,
    )
    cfg = TrainingConfig(
        output_dir=tmp_path / "run",
        pdb_dir=tmp_path / "pdb",
        disc_scatter_structure="4OBE:A",
        use_dehydron_barcode=True,
        dehydron_barcode_dir=tmp_path / "barcodes",
        device="cpu",
    )
    paths = export_disc_governance_artifacts(_StubModel(), cfg, tmp_path / "gov", device="cpu")
    assert set(paths) == {"probe_curvature_sources.json"}
    assert paths["probe_curvature_sources.json"].is_file()
