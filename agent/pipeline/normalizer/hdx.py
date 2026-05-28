"""Normalize HDX-MS correlation results into fact_hdx_correlation and fact_hdx_residue.

Write path:
    HDXCorrelationResult  →  fact_hdx_correlation   (one run row)
                          →  fact_hdx_residue        (one row per residue in residue_data)

Schema columns:
    fact_hdx_correlation: hdx_run_id, structure_id, r_squared, spearman_rho,
                          p_value, passed, num_residues, duration_sec,
                          source_type, computed_at
    fact_hdx_residue:     hdx_residue_id, hdx_run_id, residue_index,
                          fractional_uptake, wrapping_count, num_peptides
"""
import uuid
from typing import List

import sqlalchemy as sa


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

_INSERT_RUN = sa.text("""
    INSERT INTO fact_hdx_correlation
        (hdx_run_id, structure_id,
         r_squared, spearman_rho, p_value,
         passed, num_residues, duration_sec,
         source_type)
    VALUES
        (:hdx_run_id, :structure_id,
         :r_squared, :spearman_rho, :p_value,
         :passed, :num_residues, :duration_sec,
         :source_type)
    ON CONFLICT (hdx_run_id) DO NOTHING
""")

_INSERT_RESIDUE = sa.text("""
    INSERT INTO fact_hdx_residue
        (hdx_residue_id, hdx_run_id,
         residue_index, fractional_uptake,
         wrapping_count, num_peptides)
    VALUES
        (:hdx_residue_id, :hdx_run_id,
         :residue_index, :fractional_uptake,
         :wrapping_count, :num_peptides)
    ON CONFLICT (hdx_residue_id) DO NOTHING
""")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def normalize_hdx(
    conn,
    structure_id: str,
    hdx_results: list,
    source_type: str = "empirical",
) -> int:
    """Write HDX-MS correlation results to fact_hdx_correlation and fact_hdx_residue.

    Parameters
    ----------
    conn:
        Async SQLAlchemy connection (or any object with an ``execute``
        coroutine).
    structure_id:
        UUID v4 identifying the parent structure in dim_structure.
    hdx_results:
        List of ``HDXCorrelationResult`` objects produced by the HDX-MS
        validation service.  An empty list is a no-op and returns 0.
    source_type:
        Provenance tag — ``"empirical"`` for real HDX-MS measurements.

    Returns
    -------
    int
        Number of fact_hdx_correlation rows inserted.
    """
    if not hdx_results:
        return 0

    count = 0
    for hdx_result in hdx_results:
        run_id = str(uuid.uuid4())

        await conn.execute(_INSERT_RUN, {
            "hdx_run_id":    run_id,
            "structure_id":  structure_id,
            "r_squared":     hdx_result.r_squared,
            "spearman_rho":  hdx_result.spearman_rho,
            "p_value":       hdx_result.p_value,
            "passed":        hdx_result.passed,
            "num_residues":  hdx_result.num_residues,
            "duration_sec":  hdx_result.computation_time_sec,
            "source_type":   source_type,
        })

        # Write one residue row for each per-residue data point.
        for res in getattr(hdx_result, "residue_data", []):
            await conn.execute(_INSERT_RESIDUE, {
                "hdx_residue_id":    str(uuid.uuid4()),
                "hdx_run_id":        run_id,
                "residue_index":     res.residue_id,
                "fractional_uptake": res.fractional_uptake,
                "wrapping_count":    res.wrapping_count,
                "num_peptides":      res.num_peptides,
            })

        count += 1

    return count
