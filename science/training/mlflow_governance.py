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
from science.dtie.common.residue_features import GnnInputMode, gnn_feature_set_id_for_barcode
from science.training.config import PhaseConfig, TrainingConfig

logger = logging.getLogger(__name__)

SPEC_VERSION = "TRAINING_GOVERNANCE_AND_MLFLOW_SCHEMA:2026-07-07"

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

from science.training.topology_depth import TOPOLOGY_MANDATORY_METRICS, topology_depth_lineage
from science.training.routing_gate_bounds import model_num_experts


def mandatory_metrics_for_lineage(*, master_cold_lineage: bool = False) -> frozenset[str]:
    if topology_depth_lineage(master_cold=master_cold_lineage):
        return TOPOLOGY_MANDATORY_METRICS
    return MANDATORY_METRICS


# Stage A→B routing gate (§5) — effective count scale, not routing fractions.
# Reference band at N=4; use stage_a_effective_experts_bounds(num_experts) at runtime.
STAGE_A_EFFECTIVE_EXPERTS_MIN = 3.0
STAGE_A_EFFECTIVE_EXPERTS_MAX = 4.5
STAGE_A_EFFECTIVE_EXPERTS_FLOOR = 2.5
STAGE_A_MIN_ROUTING_FRACTION = 0.05
_REF_NUM_EXPERTS = 4

MANDATORY_ARTIFACTS = frozenset(
    {
        "poincare_disc_overlay.png",
        "angular_distribution_stats.json",
        "probe_curvature_sources.json",
    }
)


def fold_id_to_mlflow_key(fold_id: str) -> str:
    """Sanitize CATH topology code for MLflow metric keys (§ stage-a-corpus-selection)."""
    return str(fold_id).replace(".", "_")


def mlflow_key_to_fold_id(key: str) -> str:
    """Reverse map from MLflow key segment to CATH fold_id."""
    parts = key.split("_")
    if len(parts) >= 4 and all(p.isdigit() for p in parts[:4]):
        return ".".join(parts[:4])
    return key.replace("_", ".")


def resolve_protein_fold_id(entry: dict[str, Any]) -> str | None:
    """CATH topology code for a manifest entry (manifest field or frozen PDBe cache)."""
    if entry.get("fold_id"):
        return str(entry["fold_id"])
    from experiments.training.v6.cath_coverage_probe import resolve_fold_from_cache

    cath = resolve_fold_from_cache(str(entry["pdb_id"]), str(entry.get("chain", "A")))
    return cath.get("fold_id")


def corpus_fold_ids(manifest_path: Path) -> set[str]:
    from experiments.training.v6.corpus import load_corpus_manifest

    data = load_corpus_manifest(manifest_path)
    fold_ids: set[str] = set()
    for entry in data.get("proteins", []):
        if not entry.get("enabled", True):
            continue
        fid = resolve_protein_fold_id(entry)
        if fid:
            fold_ids.add(fid)
    return fold_ids


def gene_to_protein_family(gene: str) -> str:
    return GENE_TO_FAMILY.get(str(gene or "").upper(), "other")


def corpus_manifest_hash(manifest_path: Path) -> str:
    data = Path(manifest_path).read_bytes()
    return hashlib.sha256(data).hexdigest()


def corpus_families(manifest_path: Path) -> set[str]:
    """Deprecated — use corpus_fold_ids. Retained for transitional imports."""
    return {fold_id_to_mlflow_key(f) for f in corpus_fold_ids(manifest_path)}


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
    if config.master_cold_lineage or config.slim_moe_structural_ssot:
        parent_value = "null"
        warm_start_value = "none"
        curvature_mode = "free"
    else:
        parent_value = parent_run_id or "cold_start"
        warm_start_value = "resume" if config.resume is not None else "none"
        curvature_mode = _curvature_mode(config)
    params: dict[str, str] = {
        "branch": branch or "residue-only",
        "parent_run_id": parent_value,
        "warm_start": warm_start_value,
        "corpus_manifest_hash": corpus_manifest_hash(manifest),
        "corpus_size": str(corpus_size),
        "curvature_mode": curvature_mode,
        "scale": "micro",
        "feature_set": gnn_feature_set_id_for_barcode(
            config.use_dehydron_barcode,
            config.use_binned_dehydron,
            mode=GnnInputMode.TOPOLOGY_THREE_VECTOR if config.use_dehydron_barcode else None,
        ),
        "curriculum_schedule": build_curriculum_schedule_json(phases or []),
        "git_commit": _git_commit(),
        "spec_version": SPEC_VERSION,
        "space_name": V6_HYP_SPACE_NAME,
        "num_experts": str(config.num_experts),
        "gnn_lineage": config.gnn_lineage,
        "model_version": config.model_version,
    }
    from science.training.routing_gate_bounds import (
        CAPACITY_OUTCOME_A_VS_C,
        routing_entropy_save_ceiling,
    )

    params["routing_entropy_save_ceiling"] = str(
        routing_entropy_save_ceiling(num_experts=config.num_experts)
    )
    if config.num_experts != 4:
        params["capacity_outcome_discriminator"] = CAPACITY_OUTCOME_A_VS_C
    if config.master_cold_lineage or config.slim_moe_structural_ssot:
        params["lineage_root"] = "true"
        params["topology_only_gate"] = "true"
        params["v2_teacher"] = "disabled"
    if config.structural_disc_frozen:
        params["disc_layout_source"] = "structural_ssot_frozen"
    try:
        from science.training.p_feature_01_gate import (
            DEFAULT_STAMP_PATH,
            governance_params_from_stamp,
            load_gate_stamp,
            validate_gate_stamp,
        )

        stamp = load_gate_stamp(DEFAULT_STAMP_PATH)
        validate_gate_stamp(stamp, manifest)
        params.update(governance_params_from_stamp(stamp))
    except (FileNotFoundError, RuntimeError, OSError):
        params["p_feature_01_passed"] = "false"
    return params


def governance_epoch_metrics(
    health: dict[str, float],
    losses: dict[str, float],
    model: nn.Module,
    *,
    inference_routing: dict[str, float] | None = None,
    master_cold_lineage: bool = False,
) -> dict[str, float]:
    """Map loop outputs to mandatory MLflow metric names (§3.2)."""
    from science.training.routing_metrics import collapse_metrics_from_epoch_losses

    disc_thick = health.get("disc_line_thickness_pre_mean")
    if disc_thick is None or not math.isfinite(disc_thick):
        disc_thick = health.get("disc_line_thickness_rms_mean", float("nan"))

    train_routing = collapse_metrics_from_epoch_losses(losses)
    gate_routing = inference_routing if inference_routing is not None else train_routing

    metrics: dict[str, float] = {
        "log_c": _log_c_raw(model),
        **gate_routing,
        "sigma2_sigma1": float(health.get("disc_sigma2_sigma1_mean", float("nan"))),
        "disc_thick": float(disc_thick),
        "r_d_tau": float(health.get("probe_r_depth_tau", float("nan"))),
        "r_d_rho": float(health.get("probe_r_depth_rho", float("nan"))),
        "stage_gate_passed": float(
            stage_gate_passed(
                health,
                losses,
                inference_routing=inference_routing,
                num_experts=model_num_experts(model),
                master_cold_lineage=master_cold_lineage,
            )
        ),
    }
    if topology_depth_lineage(master_cold=master_cold_lineage):
        pass  # SASA shell probes omitted on topology lineage
    else:
        metrics["r_d_s"] = float(health.get("probe_r_depth_sasa", float("nan")))
        metrics["r_e_s"] = float(health.get("probe_r_epi_sasa", float("nan")))
    if inference_routing is not None:
        metrics["train_effective_experts"] = train_routing["effective_experts"]
        metrics["train_effective_experts_min"] = train_routing["effective_experts_min"]
        metrics["train_min_routing_fraction"] = train_routing["min_routing_fraction"]

    for key, value in losses.items():
        if key.startswith("per_fold_loss."):
            metrics[key] = float(value)

    for key, value in (inference_routing or {}).items():
        if key.startswith("eval_min_routing_fraction."):
            metrics[key] = float(value)

    return {k: v for k, v in metrics.items() if v is not None and math.isfinite(v)}


# Track-only telemetry (``telemetry/track`` tag — not P-entry gates). See §3.2 telemetry table.
TELEMETRY_TRACK_HEALTH_TO_MLFLOW: dict[str, str] = {
    "epistemic_std_mean": "track/epistemic_std",
    "aleatoric_std_mean": "track/aleatoric_std",
    "probe_r_epi_ale": "track/epi_ale_corr",
    "uncertainty_probe_alive_epi": "track/uncertainty_alive_epi",
    "uncertainty_probe_alive_ale": "track/uncertainty_alive_ale",
    "uncertainty_informative_ale": "track/uncertainty_informative_ale",
    "node_aleatoric_tau_lift": "track/node_tau_boundary_ale_lift",
    "node_aleatoric_tau_lift_relative": "track/node_tau_boundary_ale_lift_relative",
    "edge_embed_resistance_corr_mean": "track/edge_resistance_corr",
    "edge_epistemic_var_std_mean": "track/edge_epistemic_std",
    "edge_aleatoric_var_std_mean": "track/edge_aleatoric_std",
    "same_expert_rate_mean": "track/same_expert_rate",
    "same_expert_null_rate_mean": "track/same_expert_null_rate",
    "same_expert_excess_mean": "track/same_expert_excess",
    "flow_excess_high_minus_low_mean": "track/flow_excess_high_minus_low",
    "edge_telemetry_alive_fraction": "track/edge_telemetry_alive",
    "healthy_flow_alignment_fraction": "track/healthy_flow_alignment",
    "edge_tau_boundary_aleatoric_lift_mean": "track/edge_tau_boundary_ale_lift",
    "uncertainty_epistemic_non_degenerate": "track/epistemic_non_degenerate",
    "uncertainty_tau_ale_elevated": "track/tau_ale_elevated",
    "uncertainty_decomposition_valid": "track/decomposition_valid",
    "evidence_nu_cv_mean": "track/nu_cv",
}


def telemetry_track_metrics(health: dict[str, float]) -> dict[str, float]:
    """Map geometry health telemetry to MLflow ``track/*`` keys (not gate-critical)."""
    out: dict[str, float] = {}
    for health_key, mlflow_key in TELEMETRY_TRACK_HEALTH_TO_MLFLOW.items():
        val = health.get(health_key)
        if val is None:
            continue
        try:
            fval = float(val)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fval):
            out[mlflow_key] = fval
    return out


def stage_a_gate_passed(
    health: dict[str, float],
    losses: dict[str, float],
    *,
    inference_routing: dict[str, float] | None = None,
    num_experts: int = _REF_NUM_EXPERTS,
) -> int:
    """Stage A→B pre-registered gate (§5). Returns 1 if pass, 0 if fail."""
    from science.training.routing_gate_bounds import stage_a_effective_experts_bounds
    from science.training.routing_metrics import collapse_metrics_from_epoch_losses

    eff_min_bound, eff_max_bound, eff_floor_bound = stage_a_effective_experts_bounds(num_experts)

    routing = (
        inference_routing
        if inference_routing is not None
        else collapse_metrics_from_epoch_losses(losses)
    )
    eff = routing.get("effective_experts", float("nan"))
    eff_min = routing.get("effective_experts_min", float("nan"))
    min_frac = routing.get("min_routing_fraction", float("nan"))
    sigma = float(health.get("disc_sigma2_sigma1_mean", float("nan")))
    r_ds = float(health.get("probe_r_depth_sasa", float("nan")))

    if not math.isfinite(eff) or not (eff_min_bound <= eff <= eff_max_bound):
        return 0
    if not math.isfinite(eff_min) or eff_min <= eff_floor_bound:
        return 0
    if not math.isfinite(min_frac) or min_frac < STAGE_A_MIN_ROUTING_FRACTION:
        return 0
    if not math.isfinite(sigma) or abs(sigma - 0.665) / 0.665 >= 0.10:
        return 0
    if not math.isfinite(r_ds) or abs(r_ds - 0.730) >= 0.05:
        return 0

    fold_losses = [float(v) for k, v in losses.items() if k.startswith("per_fold_loss.")]
    if len(fold_losses) >= 2:
        ratio = max(fold_losses) / max(min(fold_losses), 1e-8)
        if ratio >= 3.0:
            return 0
    return 1


def master_cold_stage_gate_passed(
    health: dict[str, float],
    losses: dict[str, float],
    *,
    inference_routing: dict[str, float] | None = None,
    num_experts: int = _REF_NUM_EXPERTS,
) -> int:
    """MASTER cold lineage: routing + disc occupancy + P_DEHYDRON_CONE_01 (τ-rim only)."""
    from science.training.dehydron_cone_gate import dehydron_cone_gate_passed
    from science.training.routing_gate_bounds import stage_a_effective_experts_bounds
    from science.training.routing_metrics import collapse_metrics_from_epoch_losses

    eff_min_bound, eff_max_bound, eff_floor_bound = stage_a_effective_experts_bounds(num_experts)

    routing = (
        inference_routing
        if inference_routing is not None
        else collapse_metrics_from_epoch_losses(losses)
    )
    eff = routing.get("effective_experts", float("nan"))
    eff_min = routing.get("effective_experts_min", float("nan"))
    min_frac = routing.get("min_routing_fraction", float("nan"))
    sigma = float(health.get("disc_sigma2_sigma1_mean", float("nan")))

    if not math.isfinite(eff) or not (eff_min_bound <= eff <= eff_max_bound):
        return 0
    if not math.isfinite(eff_min) or eff_min <= eff_floor_bound:
        return 0
    if not math.isfinite(min_frac) or min_frac < STAGE_A_MIN_ROUTING_FRACTION:
        return 0
    if not math.isfinite(sigma) or abs(sigma - 0.665) / 0.665 >= 0.10:
        return 0
    if not dehydron_cone_gate_passed(health):
        return 0

    fold_losses = [float(v) for k, v in losses.items() if k.startswith("per_fold_loss.")]
    if len(fold_losses) >= 2:
        ratio = max(fold_losses) / max(min(fold_losses), 1e-8)
        if ratio >= 3.0:
            return 0
    return 1


def stage_gate_passed(
    health: dict[str, float],
    losses: dict[str, float],
    *,
    inference_routing: dict[str, float] | None = None,
    num_experts: int = _REF_NUM_EXPERTS,
    master_cold_lineage: bool = False,
) -> int:
    """Stage A→B gate — SASA shell (default) or dehydron-rim (topology lineage)."""
    if master_cold_lineage or topology_depth_lineage():
        return master_cold_stage_gate_passed(
            health,
            losses,
            inference_routing=inference_routing,
            num_experts=num_experts,
        )
    return stage_a_gate_passed(
        health,
        losses,
        inference_routing=inference_routing,
        num_experts=num_experts,
    )


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


def _disc_scatter_candidates(
    config: TrainingConfig,
    proteins: list[dict[str, Any]] | None = None,
) -> list[tuple[str, str]]:
    """Structures to try for end-of-run disc governance overlay (config first, then corpus)."""
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []

    def _add(pdb_id: str, chain: str) -> None:
        key = (pdb_id.upper(), chain or "A")
        if key not in seen:
            seen.add(key)
            out.append(key)

    spec = (config.disc_scatter_structure or "11QE:A").strip()
    if ":" in spec:
        pdb_id, chain = spec.split(":", 1)
    else:
        pdb_id, chain = spec, "A"
    _add(pdb_id, chain)

    for prot in proteins or []:
        pdb = str(prot.get("pdb_id") or prot.get("structure_id") or "").strip()
        if pdb:
            _add(pdb, str(prot.get("chain") or "A"))
    return out


def export_disc_governance_artifacts(
    model: nn.Module,
    config: TrainingConfig,
    out_dir: Path,
    *,
    device: str = "cpu",
    proteins: list[dict[str, Any]] | None = None,
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
    curvature = _learned_curvature(model)
    bio = None
    pdb_id = ""
    chain = "A"
    last_err: Exception | None = None
    for pdb_id, chain in _disc_scatter_candidates(config, proteins):
        try:
            bio = _load_biology_arrays(pdb_id, chain, model, config.pdb_dir, device)
            break
        except RuntimeError as exc:
            last_err = exc
            continue
    if bio is None:
        raise last_err or RuntimeError("No structure available for disc governance export")

    ang = _compute_angular_stats(bio)
    sites = PHARMACOPHORE_SITES.get(pdb_id, [])

    overlay_path = out_dir / "poincare_disc_overlay.png"
    _plot_composite(bio, sites, overlay_path, curvature, ang)

    stats_path = out_dir / "angular_distribution_stats.json"
    stats_payload = {
        "structure_id": pdb_id,
        "chain": chain,
        "checkpoint_curvature": curvature,
        "disc_layout_source": "structural_ssot_frozen",
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
    paths = export_disc_governance_artifacts(
        model, config, artifact_dir, device=device, proteins=proteins
    )
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
    master_cold_lineage: bool = False,
) -> list[str]:
    """Return list of schema violations (empty = pass)."""
    errors: list[str] = []
    missing_params = MANDATORY_PARAMS - set(params)
    if missing_params:
        errors.append(f"missing params: {sorted(missing_params)}")
    if params.get("curvature_final", "").strip() == "":
        errors.append("curvature_final empty")
    missing_metrics = mandatory_metrics_for_lineage(master_cold_lineage=master_cold_lineage) - metric_keys
    if missing_metrics:
        errors.append(f"missing metrics: {sorted(missing_metrics)}")
    missing_artifacts = MANDATORY_ARTIFACTS - artifact_names
    if missing_artifacts:
        errors.append(f"missing artifacts: {sorted(missing_artifacts)}")
    for fid in corpus_fold_ids(manifest_path):
        key = f"per_fold_loss.{fold_id_to_mlflow_key(fid)}"
        if key not in metric_keys:
            errors.append(f"missing metric {key}")
    return errors
