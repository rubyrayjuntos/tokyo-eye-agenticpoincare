"""Tokyo Eye MLflow-native governance (taxonomy, evaluate gate, aliases)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from science.tokyo_eye.governance.evaluate import (
    PromoteBlocked,
    ThresholdSpec,
    assert_promotable,
    evaluate_metrics,
)
from science.tokyo_eye.governance.taxonomy import (
    TaxonomyError,
    experiment_path,
    mandatory_run_tags,
    validate_run_name,
)


def test_experiment_path_nested() -> None:
    assert (
        experiment_path("chemical", "affinity-head")
        == "tokyoeye/equiformer-v3-moe/chemical/affinity-head"
    )


def test_experiment_path_rejects_unknown_domain() -> None:
    with pytest.raises(TaxonomyError, match="domain"):
        experiment_path("quantum", "affinity-head")


def test_experiment_path_rejects_unknown_subsystem() -> None:
    with pytest.raises(TaxonomyError, match="subsystem"):
        experiment_path("chemical", "not-a-subsystem")


def test_validate_run_name_capability_goal() -> None:
    assert validate_run_name("affinity_core_pearson_ge_0.40") == (
        "affinity_core_pearson_ge_0.40"
    )


def test_validate_run_name_rejects_spaces() -> None:
    with pytest.raises(TaxonomyError):
        validate_run_name("bad name")


def test_mandatory_run_tags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GIT_SHA", "abc123")
    tags = mandatory_run_tags(
        domain="chemical",
        subsystem="affinity-head",
        capability_goal="affinity_core_pearson_ge_0.40",
        package_revision="source_snapshot",
    )
    assert tags["model"] == "TokyoEye"
    assert tags["lineage"] == "equiformer-v3-moe"
    assert tags["git_sha"] == "abc123"
    assert tags["package_revision"] == "source_snapshot"


def test_mandatory_run_tags_do_not_default_package_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GIT_SHA", "abc123")
    tags = mandatory_run_tags(
        domain="geometric",
        subsystem="full-stack",
        capability_goal="disc_health_smoke",
    )
    assert tags["git_sha"] == "abc123"
    assert "package_revision" not in tags


def test_evaluate_metrics_blocks_below_threshold() -> None:
    result = evaluate_metrics(
        {"core_pearson_r": 0.35},
        [ThresholdSpec(metric="core_pearson_r", minimum=0.40)],
    )
    assert result.passed is False
    with pytest.raises(PromoteBlocked):
        result.raise_if_failed()


def test_evaluate_metrics_passes() -> None:
    result = evaluate_metrics(
        {"core_pearson_r": 0.41},
        [ThresholdSpec(metric="core_pearson_r", minimum=0.40)],
    )
    assert result.passed is True


def test_registry_alias_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")

    tracking = (tmp_path / "mlruns").as_uri()
    ckpt = tmp_path / "weights.pt"
    ckpt.write_bytes(b"fake-ckpt")

    from science.tokyo_eye.governance.registry import (
        ALIAS_EXPERIMENTAL,
        get_model_by_alias,
        register_checkpoint_file,
        resolve_alias_uri,
        set_model_alias,
    )

    mlflow.set_tracking_uri(tracking)
    mlflow.set_registry_uri(tracking)
    exp = experiment_path("chemical", "affinity-head")
    mlflow.set_experiment(exp)
    with mlflow.start_run(run_name="affinity_core_pearson_ge_0.40") as run:
        mlflow.log_metric("core_pearson_r", 0.41)
        run_id = run.info.run_id

    reg = register_checkpoint_file(
        checkpoint_path=ckpt,
        run_id=run_id,
        tags=mandatory_run_tags(
            domain="chemical",
            subsystem="affinity-head",
            capability_goal="affinity_core_pearson_ge_0.40",
        ),
        tracking_uri=tracking,
    )
    aliased = set_model_alias(
        version=reg["version"],
        alias=ALIAS_EXPERIMENTAL,
        tracking_uri=tracking,
    )
    assert aliased["alias"] == ALIAS_EXPERIMENTAL
    got = get_model_by_alias(alias=ALIAS_EXPERIMENTAL, tracking_uri=tracking)
    assert got is not None
    assert got["version"] == reg["version"]
    assert resolve_alias_uri(ALIAS_EXPERIMENTAL) == "models:/TokyoEye@experimental"


def test_promote_requires_evaluate_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    tracking = (tmp_path / "mlruns").as_uri()
    ckpt = tmp_path / "weights.pt"
    ckpt.write_bytes(b"fake")
    thr = tmp_path / "thr.json"
    thr.write_text(json.dumps({"core_pearson_r": 0.40}))

    mlflow.set_tracking_uri(tracking)
    mlflow.set_experiment(experiment_path("chemical", "affinity-head"))
    with mlflow.start_run(run_name="affinity_fail") as run:
        mlflow.log_metric("core_pearson_r", 0.10)
        run_id = run.info.run_id

    with pytest.raises(PromoteBlocked):
        assert_promotable(
            run_id=run_id,
            thresholds=[ThresholdSpec(metric="core_pearson_r", minimum=0.40)],
            tracking_uri=tracking,
        )

    from science.tokyo_eye.governance.entrypoints import main

    rc = main(
        [
            "--tracking-uri",
            tracking,
            "promote",
            "--run-id",
            run_id,
            "--thresholds",
            str(thr),
            "--domain",
            "chemical",
            "--subsystem",
            "affinity-head",
            "--capability-goal",
            "affinity_core_pearson_ge_0.40",
            "--alias",
            "experimental",
            "--checkpoint",
            str(ckpt),
        ]
    )
    assert rc == 2


def test_pyfunc_real_forward_equiformer_moe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """load_context loads Equiformer+MoE; predict returns forward summaries."""
    pytest.importorskip("torch")
    import torch

    from science.tokyo_eye.governance.pyfunc_model import (
        TokyoEyePyFuncModel,
        load_tokyoeye_system,
    )
    from science.tokyo_eye.v8.equiformer_frontend import (
        StubEquiformerFrontend,
        TokyoEyeV8WithFrontend,
        load_weight_map,
    )
    from science.tokyo_eye.v8.model import TokyoEyesHyperbolicV8

    cfg = load_weight_map("science/tokyo_eye/v8/configs/equiformer_v3_weight_map.json")
    frontend = StubEquiformerFrontend(
        in_dim=3,
        scalar_dim=int(cfg["scalar_dim"]),
        vector_dim=int(cfg["vector_dim"]),
        num_backbone_blocks=int(cfg.get("num_backbone_blocks", 7)),
        live_backbone=True,
    )
    spine = TokyoEyesHyperbolicV8(
        scalar_dim=int(cfg["scalar_dim"]),
        vector_dim=int(cfg["vector_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        num_attn_layers=int(cfg.get("num_attn_layers", 2)),
        num_relations=int(cfg.get("num_relations", 6)),
        num_sdrp_classes=int(cfg.get("num_sdrp_classes", 5)),
        gate_hidden=int(cfg.get("gate_hidden", 16)),
        c=float(cfg.get("curvature_c", 1.0)),
        eps=float(cfg.get("eps", 1e-5)),
        moe_temperature=1.0,
    )
    system = TokyoEyeV8WithFrontend(frontend, spine)
    ckpt = tmp_path / "minimal_tokyoeye.pt"
    torch.save({"model": system.state_dict(), "cfg": cfg, "epoch": 0}, ckpt)

    model = TokyoEyePyFuncModel(config={"device": "cpu", "n_nodes": 6})
    assert model._checkpoint_path is None  # noqa: SLF001

    class _Ctx:
        artifacts = {"checkpoint": str(ckpt)}

    model.load_context(_Ctx())
    assert model._loaded is True  # noqa: SLF001
    assert model._system is not None  # noqa: SLF001
    out = model.predict(None, pd.DataFrame({"n_nodes": [6]}))
    assert out.iloc[0]["status"] == "ok"
    assert float(out.iloc[0]["n_nodes"]) == 6.0
    assert float(out.iloc[0]["has_moe_aux"]) == 1.0
    assert float(out.iloc[0]["moe_n_experts"]) >= 2.0

    # reload helper path
    reloaded = load_tokyoeye_system(ckpt, device="cpu")
    assert reloaded is not None


def test_train_pipeline_skip_train_sets_experimental_not_champion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mlflow = pytest.importorskip("mlflow")
    pytest.importorskip("torch")
    import torch

    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    tracking = (tmp_path / "mlruns").as_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking)
    monkeypatch.setenv("MLFLOW_REGISTRY_URI", tracking)

    from science.tokyo_eye.v8.equiformer_frontend import (
        StubEquiformerFrontend,
        TokyoEyeV8WithFrontend,
        load_weight_map,
    )
    from science.tokyo_eye.v8.model import TokyoEyesHyperbolicV8
    from science.tokyo_eye.governance.registry import get_model_by_alias
    from science.tokyo_eye.governance.train_pipeline import (
        run_train_evaluate_register_experimental,
    )

    cfg = load_weight_map("science/tokyo_eye/v8/configs/equiformer_v3_weight_map.json")
    frontend = StubEquiformerFrontend(
        in_dim=3,
        scalar_dim=int(cfg["scalar_dim"]),
        vector_dim=int(cfg["vector_dim"]),
        num_backbone_blocks=2,
        live_backbone=True,
    )
    spine = TokyoEyesHyperbolicV8(
        scalar_dim=int(cfg["scalar_dim"]),
        vector_dim=int(cfg["vector_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        num_attn_layers=1,
        num_relations=int(cfg.get("num_relations", 6)),
        num_sdrp_classes=int(cfg.get("num_sdrp_classes", 5)),
        gate_hidden=int(cfg.get("gate_hidden", 16)),
        c=1.0,
        eps=1e-5,
        moe_temperature=1.0,
    )
    system = TokyoEyeV8WithFrontend(frontend, spine)
    ckpt = tmp_path / "pipe.pt"
    torch.save({"model": system.state_dict(), "cfg": cfg}, ckpt)

    mlflow.set_tracking_uri(tracking)
    mlflow.set_registry_uri(tracking)
    out = run_train_evaluate_register_experimental(
        domain="geometric",
        subsystem="full-stack",
        capability_goal="pipeline_smoke_do_not_champion",
        tracking_uri=tracking,
        thresholds={"smoke_metric": 0.5},
        skip_train=True,
        best_checkpoint=ckpt,
        set_experimental=True,
        metrics={"smoke_metric": 0.9},
    )
    assert out["champion_unchanged"] is True
    assert out["alias"]["alias"] == "experimental"
    assert get_model_by_alias(alias="champion", tracking_uri=tracking) is None
    exp = get_model_by_alias(alias="experimental", tracking_uri=tracking)
    assert exp is not None
    assert exp["version"] == out["register"]["version"]


def test_governance_modules_do_not_import_healthy_v8() -> None:
    """MLflow path must stay separate from filesystem seal modules."""
    import science.tokyo_eye.governance.entrypoints as ep
    import science.tokyo_eye.governance.import_weights as iw
    import science.tokyo_eye.governance.resolve as rz

    for mod in (ep, iw, rz):
        src = Path(mod.__file__).read_text()
        assert "experiments.training.v8.healthy" not in src
        assert "HEALTHY_V8_" not in src


def test_import_weights_leaves_source_and_resolve_uses_mlflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    tracking = (tmp_path / "mlruns").as_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking)
    monkeypatch.setenv("MLFLOW_REGISTRY_URI", tracking)
    monkeypatch.setenv("TOKYOEYE_MLFLOW_CACHE", str(tmp_path / "alias_cache"))
    src = tmp_path / "seal_source.pt"
    src.write_bytes(b"seal-bytes-do-not-delete")
    src_mtime = src.stat().st_mtime

    from science.tokyo_eye.governance.import_weights import import_checkpoint_into_mlflow
    from science.tokyo_eye.governance.resolve import ResolveError, resolve_alias_checkpoint

    mlflow.set_tracking_uri(tracking)
    mlflow.set_registry_uri(tracking)

    out = import_checkpoint_into_mlflow(
        checkpoint_path=src,
        domain="chemical",
        subsystem="affinity-head",
        capability_goal="affinity_core_pearson_ge_0.40",
        tracking_uri=tracking,
        alias="experimental",
        metrics={"core_pearson_r": 0.41},
        log_pyfunc=True,
        overwrite_alias=True,
    )
    assert out["source_deleted"] is False
    assert src.read_bytes() == b"seal-bytes-do-not-delete"
    assert src.stat().st_mtime == src_mtime
    assert out["alias"]["alias"] == "experimental"

    resolved = resolve_alias_checkpoint(alias="experimental", tracking_uri=tracking)
    assert Path(resolved["path"]).is_file()
    # Cache path must not be the original seal path
    assert Path(resolved["path"]).resolve() != src.resolve()

    with pytest.raises(ResolveError, match="No model version"):
        resolve_alias_checkpoint(alias="champion", tracking_uri=tracking)

def test_train_shell_does_not_touch_healthy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    tracking = (tmp_path / "mlruns").as_uri()
    from science.tokyo_eye.governance.entrypoints import main

    rc = main(
        [
            "--tracking-uri",
            tracking,
            "train",
            "--domain",
            "geometric",
            "--subsystem",
            "full-stack",
            "--capability-goal",
            "disc_health_smoke",
        ]
    )
    assert rc == 0


def test_import_weights_alias_none_parses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("mlflow")
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    tracking = (tmp_path / "mlruns").as_uri()
    ckpt = tmp_path / "weights.pt"
    ckpt.write_bytes(b"fake")

    from science.tokyo_eye.governance.entrypoints import main

    rc = main(
        [
            "--tracking-uri",
            tracking,
            "import-weights",
            "--checkpoint",
            str(ckpt),
            "--domain",
            "geometric",
            "--subsystem",
            "full-stack",
            "--capability-goal",
            "disc_health_smoke",
            "--alias",
            "none",
            "--no-pyfunc",
        ]
    )
    assert rc == 0


def test_threshold_pack_seed_and_resolve_from_mlflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    tracking = (tmp_path / "mlruns").as_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking)

    from science.tokyo_eye.governance.thresholds import (
        resolve_thresholds,
        seed_threshold_pack_template,
    )

    seeded = seed_threshold_pack_template(
        domain="geometric",
        subsystem="full-stack",
        pack={"smoke_metric": 0.5, "core_pearson_r": 0.40},
        tracking_uri=tracking,
    )
    assert seeded["run_id"]
    specs, prov = resolve_thresholds(
        domain="geometric",
        subsystem="full-stack",
        tracking_uri=tracking,
    )
    assert prov["source"] == "mlflow_template_run"
    assert {s.metric: s.minimum for s in specs}["smoke_metric"] == 0.5

    specs2, prov2 = resolve_thresholds(
        thresholds_run_id=seeded["run_id"],
        tracking_uri=tracking,
    )
    assert prov2["source"] == "mlflow_run_artifact"
    assert len(specs2) == 2


def test_legacy_local_thresholds_warn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("mlflow")
    thr = tmp_path / "legacy.json"
    thr.write_text(json.dumps({"smoke_metric": 0.5}))

    from science.tokyo_eye.governance.thresholds import resolve_thresholds

    with pytest.warns(DeprecationWarning, match="legacy"):
        specs, prov = resolve_thresholds(thresholds_path=thr)
    assert prov["source"] == "legacy_local_json"
    assert specs[0].metric == "smoke_metric"


def test_pipeline_logs_threshold_pack_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mlflow = pytest.importorskip("mlflow")
    pytest.importorskip("torch")
    import torch

    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    tracking = (tmp_path / "mlruns").as_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking)
    monkeypatch.setenv("MLFLOW_REGISTRY_URI", tracking)

    from science.tokyo_eye.v8.equiformer_frontend import (
        StubEquiformerFrontend,
        TokyoEyeV8WithFrontend,
        load_weight_map,
    )
    from science.tokyo_eye.v8.model import TokyoEyesHyperbolicV8
    from science.tokyo_eye.governance.train_pipeline import (
        run_train_evaluate_register_experimental,
    )
    from science.tokyo_eye.governance.thresholds import load_thresholds_from_mlflow_artifact

    cfg = load_weight_map("science/tokyo_eye/v8/configs/equiformer_v3_weight_map.json")
    frontend = StubEquiformerFrontend(
        in_dim=3,
        scalar_dim=int(cfg["scalar_dim"]),
        vector_dim=int(cfg["vector_dim"]),
        num_backbone_blocks=2,
        live_backbone=True,
    )
    spine = TokyoEyesHyperbolicV8(
        scalar_dim=int(cfg["scalar_dim"]),
        vector_dim=int(cfg["vector_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        num_attn_layers=1,
        num_relations=int(cfg.get("num_relations", 6)),
        num_sdrp_classes=int(cfg.get("num_sdrp_classes", 5)),
        gate_hidden=int(cfg.get("gate_hidden", 16)),
        c=1.0,
        eps=1e-5,
        moe_temperature=1.0,
    )
    system = TokyoEyeV8WithFrontend(frontend, spine)
    ckpt = tmp_path / "pipe_thr.pt"
    torch.save({"model": system.state_dict(), "cfg": cfg}, ckpt)

    mlflow.set_tracking_uri(tracking)
    mlflow.set_registry_uri(tracking)
    out = run_train_evaluate_register_experimental(
        domain="geometric",
        subsystem="full-stack",
        capability_goal="threshold_pack_on_train_run",
        tracking_uri=tracking,
        thresholds={"smoke_metric": 0.5},
        skip_train=True,
        best_checkpoint=ckpt,
        set_experimental=False,
        metrics={"smoke_metric": 0.9},
    )
    loaded = load_thresholds_from_mlflow_artifact(
        run_id=out["run_id"], tracking_uri=tracking
    )
    assert loaded[0].metric == "smoke_metric"
    assert loaded[0].minimum == 0.5


def test_ensure_taxonomy_archives_legacy_fs_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HTTP tracking: filesystem artifact roots are archived; proxied name reused."""
    mlflow = pytest.importorskip("mlflow")
    from unittest.mock import MagicMock

    from science.tokyo_eye.governance import train_pipeline as tp

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

    legacy = MagicMock()
    legacy.experiment_id = "8"
    legacy.artifact_location = "/app/mlflow-artifacts/8"
    legacy.name = "tokyoeye/equiformer-v3-moe/geometric/full-stack"

    client = MagicMock()
    # First lookup returns legacy; after rename, get_experiment_by_name returns None
    # then create; second call path after rename sets exp=None in code.
    client.get_experiment_by_name.return_value = legacy

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

    # Patch MlflowClient constructor used inside ensure
    created = {}

    class FakeClient:
        def get_experiment_by_name(self, name):
            if created.get("renamed"):
                return None
            return legacy if name == legacy.name else None

        def rename_experiment(self, eid, new_name):
            assert eid == "8"
            assert new_name.endswith(".legacy_fs_8")
            created["renamed"] = True

        def create_experiment(self, name, artifact_location=None):
            created["name"] = name
            created["artifact_location"] = artifact_location
            return "11"

    monkeypatch.setattr(mlflow.tracking, "MlflowClient", FakeClient)
    monkeypatch.setattr(mlflow, "set_experiment", lambda name: None)
    monkeypatch.setattr(tp, "ensure_tracking", lambda uri=None: None)

    out = tp.ensure_taxonomy_experiment(
        "geometric", "full-stack", tracking_uri="http://127.0.0.1:5000"
    )
    assert out == legacy.name
    assert created["renamed"] is True
    assert created["artifact_location"] == tp.ARTIFACT_LOCATION
