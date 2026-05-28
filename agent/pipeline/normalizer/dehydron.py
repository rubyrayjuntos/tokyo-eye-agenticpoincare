"""Normalize dehydron detection results into fact_dehydron.

Write path:
    Dehydron  →  fact_dehydron

The fact table has a direct structure_id FK (added in 002_schema_extensions.sql)
and a nullable run_id for standalone Tier 1 runs that have no associated
validation-mining run.
"""
import uuid
from typing import Optional

import sqlalchemy as sa

from gosp.models.data_models import Dehydron


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

_INSERT = sa.text("""
    INSERT INTO fact_dehydron
        (dehydron_id, structure_id, run_id,
         donor_chain, donor_residue_index,
         acceptor_chain, acceptor_residue_index,
         midpoint_x, midpoint_y, midpoint_z,
         distance, wrapping_count,
         is_dehydron, is_interchain,
         source_type)
    VALUES
        (:dehydron_id, :structure_id, :run_id,
         :donor_chain, :donor_residue_index,
         :acceptor_chain, :acceptor_residue_index,
         :midpoint_x, :midpoint_y, :midpoint_z,
         :distance, :wrapping_count,
         :is_dehydron, :is_interchain,
         :source_type)
    ON CONFLICT (dehydron_id) DO NOTHING
""")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def normalize_dehydrons(
    conn,
    structure_id: str,
    dehydrons: list,
    source_type: str = "deterministic",
    run_id: Optional[str] = None,
) -> dict:
    """Write dehydron results to fact_dehydron.

    Parameters
    ----------
    conn:
        Async SQLAlchemy connection (or any object with an ``execute``
        coroutine).
    structure_id:
        UUID v4 identifying the parent structure in dim_structure.
    dehydrons:
        List of :class:`~gosp.models.data_models.Dehydron` objects produced
        by the dehydron-detection service.
    source_type:
        Provenance tag — ``"deterministic"`` for the physics kernel,
        ``"ml"`` for model-predicted results.
    run_id:
        Optional reference to a fact_validation_mining_run row.  Pass
        ``None`` for standalone Tier 1 runs.

    Returns
    -------
    dict[int, str]
        Mapping from 0-based insertion index to the generated UUID for that
        dehydron.  Callers (e.g. normalize_voids) use this map to resolve
        integer indices stored in Void.nearby_dehydrons to real FK values
        before writing bridge_void_dehydron rows.
    """
    id_map: dict[int, str] = {}
    for i, deh in enumerate(dehydrons):
        deh_uuid = str(uuid.uuid4())
        mx, my, mz = deh.midpoint
        await conn.execute(_INSERT, {
            "dehydron_id":          deh_uuid,
            "structure_id":         structure_id,
            "run_id":               run_id,
            "donor_chain":          deh.donor_chain_id,
            "donor_residue_index":  deh.donor_res_id,
            "acceptor_chain":       deh.acceptor_chain_id,
            "acceptor_residue_index": deh.acceptor_res_id,
            "midpoint_x":           mx,
            "midpoint_y":           my,
            "midpoint_z":           mz,
            "distance":             deh.distance,
            "wrapping_count":       deh.wrapping_count,
            "is_dehydron":          deh.is_dehydron,
            "is_interchain":        deh.is_interchain,
            "source_type":          source_type,
        })
        id_map[i] = deh_uuid
    return id_map
