"""MLflow governance schema per docs/TRAINING_GOVERNANCE_AND_MLFLOW_SCHEMA.md."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from science.dtie.common.curvature_loader import CANONICAL_V6_CURVATURE, V6_HYP_SPACE_NAME
from science.training.config import PhaseConfig, TrainingConfig

logger = logging.getLogger(__name__)

SPEC_VERSION = "TRAINING_GOVERNANCE_AND_MLFLOW_SCHEMA:2026-07-02"

GENE_TO_FAMILY: dict[str, str] = {
    "KRAS": "gtpase",
    "NRAS": "gtpase",
    "HRAS": "gtpase",
    "EGFR": "kinase",
    "BRAF": "kinase",
    "SRC": "kinase",
    "ABL1": "kinase",
    "CDK12": "kinase",
    "SHP2": "phosphatase",
    "STAT3": "tf",
    "CTNNB1": "tf",
}

MANDATORY_PARAMS = frozenset(
    {
        "branch",
        "parent_run_id",
        "corpus_manifest_hash",
        "corpus_size",
        "curvature_mode",
        "curvature_final",
        "scale",
        "feature_set",
        "curriculum_schedule",
        "git_commit",
        "spec_version",
        "space_name",
    }
)

MANDATORY_METRICS = frozenset(
    {
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
)

# Stage A→B routing gate (§5) — effective count scale, not routing fractions.
STAGE_A_EFFECTIVE_EXPERTS_MIN = 3.0
STAGE_A_EFFECTIVE_EXPERTS_MAX = 4.5
STAGE_A_EFFECTIVE_EXPERTS_FLOOR = 2.5
STAGE_A_MIN_ROUTING_FRACTION = 0.05

MANDATORY_ARTIFACTS = frozenset(
    {
        "poincare_disc_overlay.png",
        "angular_distribution_stats.json",
        "probe_curvature_sources.json",
    }
)


def gene_to_protein_family(gene: str) -> str:
    return GENE_TO_FAMILY.get(str(gene or "").upper(), "other")


def corpus_manifest_hash(manifest_path: Path) -> str:
    data = Path(manifest_path).read_bytes()
    return hashlib.sha256(data).hexdigest()


def corpus_families(manifest_path: Path) -> set[str]:
    from experiments.training.v6.corpus import load_corpus_manifest

    data = load_corpus_manifest(manifest_path)
    families: set[str] = set()
    for entry in data.get("proteins", []):
        if not entry.get("enabled", True):
            continue
        families.add(gene_to_protein_family(entry.get("gene", "")))
    return families


def corpus_enabled_count(manifest_path: Path) -> int:
    from experiments.training.v6.corpus import load_corpus_manifest

    data = load_corpus_manifest(manifest_path)
    return sum(1 for e in data.get("proteins", []) if e.get("enabled", True))


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return "unknown"


def _curvature_mode(config: TrainingConfig) -> str:
    if config.resume is not None:
        return "warm_start"
    return "free"


def _learned_curvature(model: nn.Module) -> float:
    if hasattr(model, "curvature"):
        return float(model.curvature.detach().cpu().item())
    if hasattr(model, "log_c"):
        return float(F.softplus(model.log_c).item() + 1e-4)
    return float("nan")


def _log_c_raw(model: nn.Module) -> float:
    if hasattr(model, "log_c"):
        return float(model.log_c.detach().cpu().item())
    return float("nan")


def build_curriculum_schedule_json(phases: list[PhaseConfig]) -> str:
    payload = [
        {
            "phase": p.phase,
            "name": p.name,
            "epochs": p.epochs,
            "lr": p.lr,
            "freeze_radial": p.freeze_radial,
            "freeze_angular": p.freeze_angular,
            "freeze_gate": p.freeze_gate,
        }
        for p in phases
    ]
    return json.dumps(payload, sort_keys=True)


def build_governance_params(
    config: TrainingConfig,
    *,
    proteins: list[dict[str, Any]] | None = None,
    parent_run_id: str | None = None,
    phases: list[PhaseConfig] | None = None,
    branch: str | None = None,
) -> dict[str, str]:
    manifest = Path(config.corpus_manifest)
    corpus_size = len(proteins) if proteins is not None else corpus_enabled_count(manifest)
    params: dict[str, str] = {
        "branch": branch or "residue-only",
        "parent_run_id": parent_run_id or "cold_start",
        "corpus_manifest_hash": corpus_manifest_hash(manifest),
        "corpus_size": str(corpus_size),
        "curvature_mode": _curvature_mode(config),
        "scale": "micro",
        "feature_set": "dehydron-only",
        "curriculum_schedule": build_curriculum_schedule_json(phases or []),
        "git_commit": _git_commit(),
        "spec_version": SPEC_VERSION,
        "space_name": V6_HYP_SPACE_NAME,
    }
    return params


def governance_epoch_metrics(
    health: dict[str, float],
    losses: dict[str, float],
    model: nn.Module,
) -> dict[str, float]:
    """Map loop outputs to mandatory MLflow metric names (§3.2)."""
    from science.training.routing_metrics import collapse_metrics_from_epoch_losses

    disc_thick = health.get("disc_line_thickness_pre_mean")
    if disc_thick is None or not math.isfinite(disc_thick):
        disc_thick = health.get("disc_line_thickness_rms_mean", float("nan"))

    routing = collapse_metrics_from_epoch_losses(losses)

    metrics: dict[str, float] = {
        "log_c": _log_c_raw(model),
        **routing,
        "sigma2_sigma1": float(health.get("disc_sigma2_sigma1_mean", float("nan"))),
        "disc_thick": float(disc_thick),
        "r_d_s": float(health.get("probe_r_depth_sasa", float("nan"))),
        "r_e_s": float(health.get("probe_r_epi_sasa", float("nan"))),
        "stage_gate_passed": float(stage_a_gate_passed(health, losses)),
    }

    for key, value in losses.items():
        if key.startswith("per_family_loss."):
            metrics[key] = float(value)

    return {k: v for k, v in metrics.items() if v is not None and math.isfinite(v)}


def stage_a_gate_passed(health: dict[str, float], losses: dict[str, float]) -> int:
    """Stage A→B pre-registered gate (§5). Returns 1 if pass, 0 if fail."""
    from science.training.routing_metrics import collapse_metrics_from_epoch_losses

    routing = collapse_metrics_from_epoch_losses(losses)
    eff = routing.get("effective_experts", float("nan"))
    eff_min = routing.get("effective_experts_min", float("nan"))
    min_frac = routing.get("min_routing_fraction", float("nan"))
    sigma = float(health.get("disc_sigma2_sigma1_mean", float("nan")))
    r_ds = float(health.get("probe_r_depth_sasa", float("nan")))

    if not math.isfinite(eff) or not (
        STAGE_A_EFFECTIVE_EXPERTS_MIN <= eff <= STAGE_A_EFFECTIVE_EXPERTS_MAX
    ):
        return 0
    if not math.isfinite(eff_min) or eff_min <= STAGE_A_EFFECTIVE_EXPERTS_FLOOR:
        return 0
    if not math.isfinite(min_frac) or min_frac < STAGE_A_MIN_ROUTING_FRACTION:
        return 0
    if not math.isfinite(sigma) or abs(sigma - 0.665) / 0.665 >= 0.10:
        return 0
    if not math.isfinite(r_ds) or abs(r_ds - 0.730) >= 0.05:
        return 0

    family_losses = [float(v) for k, v in losses.items() if k.startswith("per_family_loss.")]
    if len(family_losses) >= 2:
        ratio = max(family_losses) / max(min(family_losses), 1e-8)
        if ratio >= 3.0:
            return 0
    return 1


def probe_curvature_sources_training(model: nn.Module) -> dict[str, Any]:
    """Lightweight SSOT probe for training runs (no DB required)."""
    c = _learned_curvature(model)
    return {
        "model_checkpoint_c": c,
        "canonical_v6_pin": CANONICAL_V6_CURVATURE,
        "space_name": V6_HYP_SPACE_NAME,
        "model_vs_canonical_match": math.isclose(c, CANONICAL_V6_CURVATURE, rel_tol=1e-4, abs_tol=1e-5),
        "note": "Training-time probe; full probe_curvature_sources requires DB at ingest.",
    }


def export_disc_governance_artifacts(
    model: nn.Module,
    config: TrainingConfig,
    out_dir: Path,
    *,
    device: str = "cpu",
) -> dict[str, Path]:
    """Write mandatory disc overlay + angular stats artifacts (§3.3)."""
    from experiments.diagnostics.crescent_biology_projection import (
        _angular_stats_to_dict,
        _compute_angular_stats,
        _load_biology_arrays,
        _plot_composite,
        PHARMACOPHORE_SITES,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    spec = config.disc_scatter_structure.strip() or "11QE:A"
    if ":" in spec:
        pdb_id, chain = spec.split(":", 1)
    else:
        pdb_id, chain = spec, "A"
    pdb_id = pdb_id.upper()

    curvature = _learned_curvature(model)
    bio = _load_biology_arrays(pdb_id, chain, model, config.pdb_dir, device)
    ang = _compute_angular_stats(bio)
    sites = PHARMACOPHORE_SITES.get(pdb_id, [])

    overlay_path = out_dir / "poincare_disc_overlay.png"
    _plot_composite(bio, sites, overlay_path, curvature, ang)

    stats_path = out_dir / "angular_distribution_stats.json"
    stats_payload = {
        "structure_id": pdb_id,
        "chain": chain,
        "checkpoint_curvature": curvature,
        "angular_distribution_stats": [_angular_stats_to_dict(ang)],
    }
    stats_path.write_text(json.dumps(stats_payload, indent=2), encoding="utf-8")

    probe_path = out_dir / "probe_curvature_sources.json"
    probe_path.write_text(
        json.dumps(probe_curvature_sources_training(model), indent=2),
        encoding="utf-8",
    )

    return {
        "poincare_disc_overlay.png": overlay_path,
        "angular_distribution_stats.json": stats_path,
        "probe_curvature_sources.json": probe_path,
    }


def finalize_governance_run(
    tracker: Any,
    model: nn.Module,
    config: TrainingConfig,
    proteins: list[dict[str, Any]],
    *,
    device: str = "cpu",
) -> None:
    """Log end-of-run params and mandatory artifacts."""
    c_final = _learned_curvature(model)
    tracker.log_params({"curvature_final": f"{c_final:.16g}"})

    artifact_dir = config.output_dir / "mlflow_governance"
    paths = export_disc_governance_artifacts(model, config, artifact_dir, device=device)
    for name, path in paths.items():
        tracker.log_artifact(path, artifact_path="governance")

    # Flat names for P_MLFLOW_01 artifact set checks
    flat_dir = artifact_dir / "flat"
    flat_dir.mkdir(parents=True, exist_ok=True)
    for name, path in paths.items():
        dest = flat_dir / name
        shutil.copy2(path, dest)
        tracker.log_artifact(dest)


def validate_finished_run(
    params: dict[str, str],
    metric_keys: set[str],
    artifact_names: set[str],
    *,
    manifest_path: Path,
) -> list[str]:
    """Return list of schema violations (empty = pass)."""
    errors: list[str] = []
    missing_params = MANDATORY_PARAMS - set(params)
    if missing_params:
        errors.append(f"missing params: {sorted(missing_params)}")
    if params.get("curvature_final", "").strip() == "":
        errors.append("curvature_final empty")
    missing_metrics = MANDATORY_METRICS - metric_keys
    if missing_metrics:
        errors.append(f"missing metrics: {sorted(missing_metrics)}")
    missing_artifacts = MANDATORY_ARTIFACTS - artifact_names
    if missing_artifacts:
        errors.append(f"missing artifacts: {sorted(missing_artifacts)}")
    for fam in corpus_families(manifest_path):
        key = f"per_family_loss.{fam}"
        if key not in metric_keys:
            errors.append(f"missing metric {key}")
    return errors
