"""One-epoch MASTER-feature cold-start smoke before full stage_a_small_master_cold curriculum."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from science.training.topology_depth import TOPOLOGY_MANDATORY_METRICS
from science.training.mlflow_governance import (
    MANDATORY_METRICS,
    corpus_fold_ids,
    corpus_manifest_hash,
    fold_id_to_mlflow_key,
    validate_finished_run,
)
from science.training.stage_a_smoke import collect_run_artifact_names

SMALL_CORPUS_MANIFEST = Path("manifests/v6_corpus_stage_a_small_v1.json")
SMALL_CORPUS_STRUCTURE_COUNT = 12


def validate_master_cold_smoke_run(
    params: dict[str, str],
    metric_keys: set[str],
    artifact_names: set[str],
    *,
    manifest_path: Path = SMALL_CORPUS_MANIFEST,
    full_corpus: bool = True,
) -> list[str]:
    """Validate 1-epoch cold-start smoke: lineage + MASTER features + P_MLFLOW_01 core."""
    errors: list[str] = []

    if not manifest_path.is_file():
        errors.append(f"manifest missing: {manifest_path}")
        return errors

    expected_hash = corpus_manifest_hash(manifest_path)
    logged_hash = params.get("corpus_manifest_hash", "")
    if logged_hash != expected_hash:
        errors.append(
            f"corpus_manifest_hash mismatch: run logged {logged_hash[:16]}… "
            f"expected {expected_hash[:16]}…"
        )

    expected_size = (
        str(SMALL_CORPUS_STRUCTURE_COUNT)
        if full_corpus
        else params.get("corpus_size", "")
    )
    if full_corpus:
        corpus_size = params.get("corpus_size", "")
        if corpus_size != expected_size:
            errors.append(
                f"full-corpus smoke expected corpus_size={expected_size}, got {corpus_size!r}"
            )

    for key, expected in (
        ("warm_start", "none"),
        ("parent_run_id", "null"),
        ("lineage_root", "true"),
        ("feature_set", "master_topology_three_vector"),
        ("curvature_mode", "free"),
        ("p_feature_01_passed", "true"),
    ):
        actual = params.get(key, "")
        if actual != expected:
            errors.append(f"{key} expected {expected!r}, got {actual!r}")

    for prov in ("rho_def", "tau_def", "ss_def", "sasa_def", "feature_module_sha256"):
        if not params.get(prov, "").strip():
            errors.append(f"missing gate-sourced provenance param: {prov}")

    fold_keys = sorted(k for k in metric_keys if k.startswith("per_fold_loss."))
    expected_folds = corpus_fold_ids(manifest_path) if full_corpus else set()
    if full_corpus:
        for fid in expected_folds:
            key = f"per_fold_loss.{fold_id_to_mlflow_key(fid)}"
            if key not in metric_keys:
                errors.append(f"missing metric {key}")
        if len(fold_keys) != len(expected_folds):
            errors.append(
                f"expected {len(expected_folds)} per_fold_loss keys, got {len(fold_keys)}: {fold_keys}"
            )
    elif len(fold_keys) < 1:
        errors.append(f"expected at least 1 per_fold_loss.* metric, got {fold_keys}")

    routing = ("effective_experts", "effective_experts_min", "min_routing_fraction")
    missing_routing = [k for k in routing if k not in metric_keys]
    if missing_routing:
        errors.append(f"routing gate metrics missing from run: {missing_routing}")

    mlflow_errors = validate_finished_run(
        params,
        metric_keys,
        artifact_names,
        manifest_path=manifest_path,
        master_cold_lineage=True,
    )
    if not full_corpus:
        mlflow_errors = [
            e
            for e in mlflow_errors
            if not e.startswith("missing metric per_fold_loss.")
        ]
    errors.extend(mlflow_errors)

    missing_core = TOPOLOGY_MANDATORY_METRICS - metric_keys
    if missing_core:
        errors.append(f"missing mandatory metrics: {sorted(missing_core)}")

    if "stage_gate_passed" not in metric_keys:
        errors.append("stage_gate_passed not logged — routing gate not wired")

    return errors


def smoke_assertions_doc() -> dict[str, Any]:
    return {
        "gate_id": "P_MASTER_COLD_SMOKE",
        "corpus_manifest": str(SMALL_CORPUS_MANIFEST),
        "structures": SMALL_CORPUS_STRUCTURE_COUNT,
        "epochs": 1,
        "assertions": [
            "warm_start=none, parent_run_id=null, lineage_root=true",
            "feature_set=master_topology_three_vector, curvature_mode=free",
            "p_feature_01_passed=true with stamp-derived rho/ss/sasa/tau defs + feature_module_sha256",
            "corpus_manifest_hash matches v6_corpus_stage_a_small_v1.json",
            "corpus_size=12 — all enabled structures loaded",
            "all 12 per_fold_loss.{fold_id} keys (one per CATH fold in manifest)",
            "routing metrics + stage_gate_passed wired",
            "P_MLFLOW_01 params, core metrics, governance artifacts",
            "does NOT assert P_FEATURE_01 parity — run make gate-p-feature-01 separately",
        ],
        "make_target": "train-v6-stage-a-small-master-cold-smoke",
        "prerequisite": "make gate-p-feature-01",
    }
