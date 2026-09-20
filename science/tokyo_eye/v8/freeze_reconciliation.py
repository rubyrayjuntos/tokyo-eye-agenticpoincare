"""Contract IDs for the first post-addendum *training* MLflow run.

Pytest / property tests are not MLflow runs. Do not backfill them.
"""

from __future__ import annotations

from typing import Any

ADDENDUM_ID = "tokyo_eye_v8_freeze_reconciliation_v1"
CANONICAL_MLFLOW_EXPERIMENT = "tokyoeye/equiformer-v3-moe/geometric/full-stack"
# Pre-§2.4 filename-grep / retired veto. Never log this as a live gate.
PURE_HYP_PASS_VERSION = "section_2_4_live_forward_tracer_v1"
HISTORICAL_MLFLOW_EXPERIMENTS = frozenset(
    {
        "tokyo-eyes-v8",
        "tokyoeye/equiformer-v3-moe/geometric/hyperbolic-spine",
    }
)


def freeze_reconciliation_mlflow_params() -> dict[str, str]:
    """Params/tags the first real train must log. Not a substitute for a run."""
    return {
        "addendum_id": ADDENDUM_ID,
        "pure_hyp_pass_version": PURE_HYP_PASS_VERSION,
        "mlflow_experiment": CANONICAL_MLFLOW_EXPERIMENT,
    }


def resolve_v8_mlflow_experiment(
    *,
    cli_experiment: str = "",
    cfg: dict[str, Any] | None = None,
    taxonomy_domain: str = "",
    taxonomy_subsystem: str = "",
) -> str:
    """Weight-map / signed canonical name, not historical aliases."""
    if taxonomy_domain and taxonomy_subsystem:
        from science.tokyo_eye.governance.taxonomy import experiment_path

        exp = experiment_path(taxonomy_domain, taxonomy_subsystem)
    else:
        mapped = ""
        if cfg:
            mapped = str(cfg.get("mlflow_experiment") or "")
        exp = (cli_experiment or mapped or CANONICAL_MLFLOW_EXPERIMENT).strip()
    if exp in HISTORICAL_MLFLOW_EXPERIMENTS:
        raise ValueError(
            f"MLflow experiment {exp!r} is a historical alias. "
            f"Use {CANONICAL_MLFLOW_EXPERIMENT!r} (addendum {ADDENDUM_ID})."
        )
    return exp
