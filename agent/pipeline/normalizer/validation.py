"""Normalize validation mining pipeline output into fact_validation_mining_run.

Write path:
    ValidationMiningResponse  →  fact_validation_mining_run  (one row per run)

Schema columns (from 001_base_schema.sql §fact_validation_mining_run):
    run_id, structure_id, chain_label, focus_residues,
    status, rho_threshold, correlation_radius,
    top_n_correlations, min_dehydrons_per_site,
    warnings, started_at, finished_at, duration_sec
"""
import json
from typing import Optional

import sqlalchemy as sa

from gosp.models.data_models import ValidationMiningResponse


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

_INSERT = sa.text("""
    INSERT INTO fact_validation_mining_run
        (run_id, structure_id, chain_label, focus_residues,
         status, rho_threshold, correlation_radius,
         top_n_correlations, min_dehydrons_per_site,
         warnings, started_at, finished_at, duration_sec)
    VALUES
        (:run_id, :structure_id, :chain_label, :focus_residues,
         :status, :rho_threshold, :correlation_radius,
         :top_n_correlations, :min_dehydrons_per_site,
         :warnings, :started_at, :finished_at, :duration_sec)
    ON CONFLICT (run_id) DO NOTHING
""")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def normalize_validation_run(
    conn,
    structure_id: str,
    response: ValidationMiningResponse,
    run_id: Optional[str] = None,
) -> str:
    """Write a ValidationMiningResponse to fact_validation_mining_run.

    Parameters
    ----------
    conn:
        Async SQLAlchemy connection (or any object with an ``execute``
        coroutine).
    structure_id:
        UUID v4 identifying the parent structure in dim_structure.
    response:
        :class:`~gosp.models.data_models.ValidationMiningResponse` produced
        by the validation mining pipeline.
    run_id:
        Override the run identifier.  When ``None``, the ``run_id`` embedded
        in *response* is used.

    Returns
    -------
    str
        The ``run_id`` of the persisted row.
    """
    effective_run_id = run_id or response.run_id

    inp = response.input or {}
    chain_label = inp.get("chain")
    focus_residues = inp.get("focus_residues")
    rho_threshold = inp.get("rho_threshold")
    correlation_radius = inp.get("correlation_radius")
    top_n = inp.get("top_n_correlations")
    min_deh = inp.get("min_dehydrons_per_site")

    # Collect warnings from the conclusion (and any future top-level list).
    warnings_list = list(getattr(response.conclusion, "warnings", []))
    warnings_json = json.dumps(warnings_list) if warnings_list else None

    # Timing: the response carries timings.total; started_at can be derived
    # from response.timestamp minus duration, but we only store what we have.
    duration_sec = response.timings.total

    await conn.execute(_INSERT, {
        "run_id":                 effective_run_id,
        "structure_id":           structure_id,
        "chain_label":            chain_label,
        "focus_residues":         focus_residues,
        "status":                 response.status,
        "rho_threshold":          rho_threshold,
        "correlation_radius":     correlation_radius,
        "top_n_correlations":     top_n,
        "min_dehydrons_per_site": min_deh,
        "warnings":               warnings_json,
        "started_at":             response.timestamp,
        "finished_at":            None,
        "duration_sec":           duration_sec,
    })

    return effective_run_id
