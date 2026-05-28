"""Normalize energy calculation results into fact_energy_calculation.

Write path:
    EnergyCalculationResult  →  fact_energy_calculation

Schema columns (from 001_base_schema.sql + 002_schema_extensions.sql):
    energy_id, structure_id,
    requested_force_field, actual_method,
    delta_g, potential_energy, solvation_term, sasa,
    converged, confidence, duration_sec,
    source_type
"""
import uuid
from typing import Optional

import sqlalchemy as sa

from gosp.models.data_models import EnergyCalculationResult


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

_INSERT = sa.text("""
    INSERT INTO fact_energy_calculation
        (energy_id, structure_id,
         requested_force_field, actual_method,
         delta_g, potential_energy, solvation_term, sasa,
         converged, confidence, duration_sec,
         source_type)
    VALUES
        (:energy_id, :structure_id,
         :requested_force_field, :actual_method,
         :delta_g, :potential_energy, :solvation_term, :sasa,
         :converged, :confidence, :duration_sec,
         :source_type)
    ON CONFLICT (energy_id) DO NOTHING
""")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def normalize_energy(
    conn,
    structure_id: str,
    energy_result: EnergyCalculationResult,
    source_type: str = "deterministic",
) -> int:
    """Write an energy calculation result to fact_energy_calculation.

    Parameters
    ----------
    conn:
        Async SQLAlchemy connection (or any object with an ``execute``
        coroutine).
    structure_id:
        UUID v4 identifying the parent structure in dim_structure.
    energy_result:
        :class:`~gosp.models.data_models.EnergyCalculationResult` produced
        by the physics kernel.
    source_type:
        Provenance tag — ``"deterministic"`` for the physics kernel.

    Returns
    -------
    int
        Always 1: a single fact_energy_calculation row is inserted per call.
    """
    provenance = energy_result.provenance

    # Derive schema columns from the model + optional provenance chain.
    # force_field is stored on the result itself; actual_method and confidence
    # come from the provenance record when present, with sensible fallbacks.
    requested_force_field = energy_result.force_field
    actual_method = (
        provenance.actual_method
        if provenance is not None
        else energy_result.force_field
    )
    confidence = (
        provenance.confidence
        if provenance is not None
        else "high" if energy_result.converged else "low"
    )

    await conn.execute(_INSERT, {
        "energy_id":             str(uuid.uuid4()),
        "structure_id":          structure_id,
        "requested_force_field": requested_force_field,
        "actual_method":         actual_method,
        "delta_g":               energy_result.delta_g,
        "potential_energy":      energy_result.potential_energy,
        "solvation_term":        energy_result.solvation_term,
        "sasa":                  energy_result.sasa,
        "converged":             energy_result.converged,
        "confidence":            confidence,
        "duration_sec":          energy_result.computation_time_sec,
        "source_type":           source_type,
    })
    return 1
