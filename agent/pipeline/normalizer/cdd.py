"""Normalize CDD domain annotation results into fact_cdd_annotation.

Write path:
    list[CddDomain]  →  fact_cdd_annotation  (one row per domain per chain)

Schema columns (from 002_schema_extensions.sql §7d):
    annotation_id, structure_id, chain_label,
    domain_id, domain_name,
    start_residue, end_residue,
    e_value, bit_score, is_synthetic, source_type
"""
import uuid
from typing import List

import sqlalchemy as sa

from gosp.models.data_models import CddDomain


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

_INSERT = sa.text("""
    INSERT INTO fact_cdd_annotation
        (annotation_id, structure_id, chain_label,
         domain_id, domain_name,
         start_residue, end_residue,
         e_value, bit_score, is_synthetic, source_type)
    VALUES
        (:annotation_id, :structure_id, :chain_label,
         :domain_id, :domain_name,
         :start_residue, :end_residue,
         :e_value, :bit_score, :is_synthetic, :source_type)
    ON CONFLICT (annotation_id) DO NOTHING
""")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def normalize_cdd_annotations(
    conn,
    structure_id: str,
    annotations: List[CddDomain],
    chain_label: str = "A",
    source_type: str = "external",
    is_synthetic: bool = False,
) -> int:
    """Write CDD domain annotations to fact_cdd_annotation.

    Parameters
    ----------
    conn:
        Async SQLAlchemy connection (or any object with an ``execute``
        coroutine).
    structure_id:
        UUID v4 identifying the parent structure in dim_structure.
    annotations:
        List of :class:`~gosp.models.data_models.CddDomain` objects produced
        by the CDD annotation service.
    chain_label:
        Chain identifier these annotations belong to (e.g. ``"A"``).
    source_type:
        Provenance tag — ``"external"`` for CDD database results.
    is_synthetic:
        Whether these annotations were synthetically generated rather than
        fetched from a real CDD query.

    Returns
    -------
    int
        Number of fact_cdd_annotation rows inserted.
    """
    count = 0
    for domain in annotations:
        await conn.execute(_INSERT, {
            "annotation_id": str(uuid.uuid4()),
            "structure_id":  structure_id,
            "chain_label":   chain_label,
            "domain_id":     domain.cdd_id,
            "domain_name":   domain.name,
            "start_residue": domain.range.start,
            "end_residue":   domain.range.end,
            "e_value":       domain.evalue,
            "bit_score":     domain.bit_score,
            "is_synthetic":  is_synthetic,
            "source_type":   source_type,
        })
        count += 1
    return count
