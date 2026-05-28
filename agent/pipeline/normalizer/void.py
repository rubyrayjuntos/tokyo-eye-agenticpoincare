"""Normalize void detection results into fact_void and bridge_void_dehydron.

Write path:
    Void  →  fact_void
    Void.nearby_dehydrons  →  bridge_void_dehydron  (one row per entry)

The fact table has a direct structure_id FK (added in 002_schema_extensions.sql)
and a nullable run_id for standalone Tier 1 runs that have no associated
validation-mining run.
"""
import uuid
from typing import Optional

import sqlalchemy as sa

from gosp.models.data_models import Void


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

_INSERT_VOID = sa.text("""
    INSERT INTO fact_void
        (void_id, structure_id, run_id,
         center_x, center_y, center_z,
         volume, point_count,
         source_type)
    VALUES
        (:void_id, :structure_id, :run_id,
         :center_x, :center_y, :center_z,
         :volume, :point_count,
         :source_type)
    ON CONFLICT (void_id) DO NOTHING
""")

_INSERT_BRIDGE = sa.text("""
    INSERT INTO bridge_void_dehydron (void_id, dehydron_id)
    VALUES (:void_id, :dehydron_id)
    ON CONFLICT DO NOTHING
""")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def normalize_voids(
    conn,
    structure_id: str,
    voids: list,
    source_type: str = "deterministic",
    run_id: Optional[str] = None,
    dehydron_id_map: Optional[dict] = None,
) -> int:
    """Write void results to fact_void and bridge_void_dehydron.

    Parameters
    ----------
    conn:
        Async SQLAlchemy connection (or any object with an ``execute``
        coroutine).
    structure_id:
        UUID v4 identifying the parent structure in dim_structure.
    voids:
        List of :class:`~gosp.models.data_models.Void` objects produced by
        the void-detection service.
    source_type:
        Provenance tag — ``"deterministic"`` for the physics kernel.
    run_id:
        Optional reference to a fact_validation_mining_run row.  Pass
        ``None`` for standalone Tier 1 runs.
    dehydron_id_map:
        Mapping from 0-based integer index (as stored in
        ``Void.nearby_dehydrons``) to the UUID string used as
        ``fact_dehydron.dehydron_id``.  Returned by
        :func:`~gosp.normalizer.dehydron.normalize_dehydrons`.  When
        ``None`` or when a ref key is absent, the bridge row is silently
        skipped to avoid FK violations.

    Returns
    -------
    int
        Number of fact_void rows inserted.
    """
    resolved_map = dehydron_id_map or {}
    count = 0
    for void_result in voids:
        void_id = str(uuid.uuid4())
        cx, cy, cz = void_result.center

        await conn.execute(_INSERT_VOID, {
            "void_id":      void_id,
            "structure_id": structure_id,
            "run_id":       run_id,
            "center_x":     cx,
            "center_y":     cy,
            "center_z":     cz,
            "volume":       void_result.volume,
            "point_count":  void_result.point_count,
            "source_type":  source_type,
        })

        # Write one bridge row for each nearby dehydron.
        # nearby_dehydrons contains 0-based integer indices that are resolved
        # to UUIDs via dehydron_id_map.  Unresolvable refs are silently skipped
        # to prevent FK violations against fact_dehydron.dehydron_id.
        for deh_ref in getattr(void_result, "nearby_dehydrons", []):
            resolved_id = resolved_map.get(deh_ref)
            if resolved_id is None:
                continue
            await conn.execute(_INSERT_BRIDGE, {
                "void_id":     void_id,
                "dehydron_id": resolved_id,
            })

        count += 1
    return count
