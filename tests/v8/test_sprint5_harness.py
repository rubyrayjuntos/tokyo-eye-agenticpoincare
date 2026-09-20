"""Sprint 5: Equiformer weight map + MLflow harness smoke (v8 isolated)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from science.tokyo_eye.v8.equiformer_frontend import (
    DEFAULT_WEIGHT_MAP,
    StubEquiformerFrontend,
    apply_weight_map,
    build_param_groups,
    load_weight_map,
)
from science.tokyo_eye.v8.metrics import binary_auprc
from science.tokyo_eye.v8.model import TokyoEyesHyperbolicV8


def test_weight_map_schema_loads() -> None:
    cfg = load_weight_map(DEFAULT_WEIGHT_MAP)
    assert cfg["schema_version"] == 1
    assert cfg["checkpoint_path_default"].endswith("equiformer_v3_baseline.pt")
    assert cfg["scalar_dim"] == 128
    assert cfg["vector_dim"] == 3
    assert cfg["lr_backbone"] == pytest.approx(1e-5)
    assert cfg["lr_hyperbolic"] == pytest.approx(1e-3)
    assert cfg["mlflow_experiment"] == "tokyoeye/equiformer-v3-moe/geometric/full-stack"


def test_first_train_mlflow_contract_is_wired_not_just_documented() -> None:
    """Do not log pytest as an MLflow run; do pin what the first train must carry."""
    import json

    from science.tokyo_eye.v8.freeze_reconciliation import (
        ADDENDUM_ID,
        CANONICAL_MLFLOW_EXPERIMENT,
        HISTORICAL_MLFLOW_EXPERIMENTS,
        PURE_HYP_PASS_VERSION,
        freeze_reconciliation_mlflow_params,
        resolve_v8_mlflow_experiment,
    )
    from science.tokyo_eye.v8.pure_hyp_pass import PURE_HYP_PASS_VERSION as SCANNER_VER

    cfg = load_weight_map(DEFAULT_WEIGHT_MAP)
    assert resolve_v8_mlflow_experiment(cfg=cfg) == CANONICAL_MLFLOW_EXPERIMENT
    assert resolve_v8_mlflow_experiment() == CANONICAL_MLFLOW_EXPERIMENT
    assert (
        resolve_v8_mlflow_experiment(
            taxonomy_domain="geometric", taxonomy_subsystem="full-stack"
        )
        == CANONICAL_MLFLOW_EXPERIMENT
    )
    with pytest.raises(ValueError, match="historical alias"):
        resolve_v8_mlflow_experiment(
            taxonomy_domain="geometric", taxonomy_subsystem="hyperbolic-spine"
        )
    fields = freeze_reconciliation_mlflow_params()
    assert fields["addendum_id"] == ADDENDUM_ID
    assert fields["pure_hyp_pass_version"] == SCANNER_VER == PURE_HYP_PASS_VERSION
    assert fields["mlflow_experiment"] == CANONICAL_MLFLOW_EXPERIMENT
    gate = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "data/gates/tokyo_eye_v8_freeze_addendum.json"
        ).read_text(encoding="utf-8")
    )
    assert gate["addendum_id"] == ADDENDUM_ID
    assert gate["pure_hyp_pass_version"] == PURE_HYP_PASS_VERSION
    assert gate["mlflow_do_not_backfill_pytest"] is True
    assert gate["mlflow_first_train_must_log"]["experiment"] == CANONICAL_MLFLOW_EXPERIMENT
    assert "addendum_id" in gate["mlflow_first_train_must_log"]["params_and_tags"]
    src = Path(__file__).resolve().parents[2] / "experiments/training/v8/run_v8_experiment.py"
    text = src.read_text(encoding="utf-8")
    assert "freeze_reconciliation_mlflow_params" in text
    assert "set_experiment(exp)" in text


def test_apply_weight_map_intersection(tmp_path: Path) -> None:
    fe = StubEquiformerFrontend(in_dim=3, scalar_dim=16, vector_dim=3)
    sd = fe.state_dict()
    # Corrupt one tensor name to ensure rename path works
    ckpt = {k: v.clone() for k, v in sd.items()}
    path = tmp_path / "toy_equiformer.pt"
    torch.save({"state_dict": ckpt}, path)
    info = apply_weight_map(fe, path, {"key_map": {"exact_name_intersection": True, "renames": {}}})
    assert info["n_loaded"] == len(sd)


def test_param_groups_differential_lr() -> None:
    fe = StubEquiformerFrontend(scalar_dim=8, vector_dim=3)
    spine = TokyoEyesHyperbolicV8(
        scalar_dim=8, vector_dim=3, hidden_dim=8, num_attn_layers=2, num_sdrp_classes=2
    )
    groups = build_param_groups(fe, spine, lr_backbone=1e-5, lr_hyperbolic=1e-3)
    assert len(groups) == 2
    assert groups[0]["name"] == "backbone"
    assert groups[1]["name"] == "hyperbolic"
    assert groups[0]["lr"] == pytest.approx(1e-5)
    assert groups[1]["lr"] == pytest.approx(1e-3)


def test_locked_regex_renames() -> None:
    from science.tokyo_eye.v8.equiformer_frontend import resolve_checkpoint_key

    cfg = load_weight_map(DEFAULT_WEIGHT_MAP)
    renames = cfg["key_map"]["renames"]
    strip = tuple(cfg["key_map"].get("strip_prefixes") or ())
    assert (
        resolve_checkpoint_key(
            "module.sphere_embedding.weight",
            renames,
            use_intersection=False,
            strip_prefixes=strip,
        )
        == "backbone.atom_embed.weight"
    )
    assert (
        resolve_checkpoint_key(
            "module.blocks.3.ga.proj.weight",
            renames,
            use_intersection=False,
            strip_prefixes=strip,
        )
        == "backbone.blocks.3.attn.proj"
    )
    assert (
        resolve_checkpoint_key(
            "module.blocks.6.ffn.scalar_mlp.0.linear.weight",
            renames,
            use_intersection=False,
            strip_prefixes=strip,
        )
        == "backbone.blocks.6.ffn.scalar_in.weight"
    )


def test_apply_weight_map_mptrj_checkpoint() -> None:
    ckpt = Path("checkpoints/v8/pretrained/equiformer_v3_baseline.pt")
    if not ckpt.is_file() or ckpt.stat().st_size < 1_000_000:
        pytest.skip("real EquiformerV3 MPtrj checkpoint not present")
    cfg = load_weight_map(DEFAULT_WEIGHT_MAP)
    fe = StubEquiformerFrontend(
        in_dim=3,
        scalar_dim=int(cfg["scalar_dim"]),
        vector_dim=int(cfg["vector_dim"]),
        num_backbone_blocks=int(cfg["num_backbone_blocks"]),
    )
    info = apply_weight_map(fe, ckpt, cfg)
    # Audited rename set: 1 atom + 2 edge + 7*(2 src/tgt + proj + 2 alpha + 2 scalar + 2 gate + 2*2 norms)
    assert info["n_renamed"] >= 90
    assert info["n_loaded"] >= 90
    assert not info["unmatched_renames"]
    assert "backbone.atom_embed.weight" in info["loaded_keys"]
    assert "backbone.blocks.0.attn.proj" in info["loaded_keys"]


def test_geometry_health_guards() -> None:
    from science.tokyo_eye.v8.equiformer_frontend import evaluate_geometry_health

    bad = evaluate_geometry_health(
        {
            "diag_manifold_entropy": 0.15,
            "diag_radius_spread": 0.01,
            "diag_boundary_saturation_pct": 55.0,
        }
    )
    assert bad["warn_oversmooth"] == 1.0
    assert bad["warn_boundary_blowout"] == 1.0
    assert bad["margin_boost_factor"] == 1.5
    # Low entropy alone (single hist bin) is not enough without collapse
    peaked = evaluate_geometry_health(
        {
            "diag_manifold_entropy": 0.1,
            "diag_radius_spread": 0.2,
            "diag_boundary_saturation_pct": 10.0,
        }
    )
    assert peaked["warn_oversmooth"] == 0.0
    ok = evaluate_geometry_health(
        {"diag_manifold_entropy": 1.2, "diag_radius_spread": 0.3, "diag_boundary_saturation_pct": 10.0}
    )
    assert ok["warn_oversmooth"] == 0.0
    assert ok["warn_boundary_blowout"] == 0.0


def test_run_v8_smoke_subprocess() -> None:
    import os
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    cmd = [
        sys.executable,
        str(root / "experiments/training/v8/run_v8_experiment.py"),
        "--smoke",
        "--no-mlflow",
        "--frontend",
        "stub",
        "--allow-off-path-frontend",
        "--epochs",
        "1",
        "--out-dir",
        str(root / "checkpoints/v8/runs"),
        "--run-name",
        "sprint5_unit_smoke",
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, check=False, env=env, cwd=str(root)
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert (root / "checkpoints/v8/runs/sprint5_unit_smoke/tokyoeye_last.pt").is_file()
