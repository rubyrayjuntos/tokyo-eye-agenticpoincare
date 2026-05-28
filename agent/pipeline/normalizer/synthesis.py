"""Normalize synthesis feasibility results into fact_synthesis_feasibility.

Write path:
    SynthesisFeasibilityResult  →  fact_synthesis_feasibility  (one row per result)

Schema columns (001_base_schema.sql + 002_schema_extensions.sql):
    synthesis_id, sequence_id, structure_id,
    manufacturable, complexity_score, issues,
    gc_content, aa_length, dna_length,
    source_type, computed_at

Dual-publisher: after each successful write this module publishes TWO Pub/Sub
events via :func:`~gosp.normalizer.events.publish_event`:
  - ``synthesis-ready``      (for synthesis consumers)
  - ``autoprotocol-ready``   (for autoprotocol consumers, which depend on synthesis)
"""
import json
import uuid
from typing import List, Optional, Tuple

import sqlalchemy as sa

from gosp.models.data_models import SynthesisFeasibilityResult
from gosp.normalizer.events import NormalizerEvent, publish_event


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

_INSERT = sa.text("""
    INSERT INTO fact_synthesis_feasibility
        (synthesis_id, sequence_id, structure_id,
         manufacturable, complexity_score, issues,
         gc_content, aa_length, dna_length,
         source_type)
    VALUES
        (:synthesis_id, :sequence_id, :structure_id,
         :manufacturable, :complexity_score, :issues,
         :gc_content, :aa_length, :dna_length,
         :source_type)
    ON CONFLICT (synthesis_id) DO NOTHING
""")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def normalize_synthesis(
    conn,
    structure_id: str,
    result: SynthesisFeasibilityResult,
    sequence_id: str,
    source_type: str = "external",
) -> int:
    """Write a synthesis feasibility result to fact_synthesis_feasibility.

    Parameters
    ----------
    conn:
        Async SQLAlchemy connection (or any object with an ``execute``
        coroutine).
    structure_id:
        UUID v4 identifying the parent structure in dim_structure.
    result:
        :class:`~gosp.models.data_models.SynthesisFeasibilityResult` produced
        by the synthesis feasibility service.
    sequence_id:
        FK to ``dim_sequence.sequence_id``.
    source_type:
        Provenance tag — ``"external"`` for Twist Bioscience API results.

    Returns
    -------
    int
        Number of rows inserted (always 1 on success).

    Side effects
    ------------
    Publishes ``synthesis-ready`` and ``autoprotocol-ready`` events to Pub/Sub
    after each successful DB write.
    """
    details = result.details or {}
    gc_content = details.get("gc_content")
    aa_length = details.get("aa_length")
    dna_length = details.get("dna_length")
    issues_json = json.dumps(result.issues) if result.issues else json.dumps([])

    await conn.execute(_INSERT, {
        "synthesis_id":    str(uuid.uuid4()),
        "sequence_id":     sequence_id,
        "structure_id":    structure_id,
        "manufacturable":  result.manufacturable,
        "complexity_score": result.complexity_score,
        "issues":          issues_json,
        "gc_content":      gc_content,
        "aa_length":       aa_length,
        "dna_length":      dna_length,
        "source_type":     source_type,
    })

    # Dual-publish: synthesis consumers and autoprotocol consumers both need
    # to know that synthesis feasibility data is available for this structure.
    publish_event(NormalizerEvent(
        structure_id=structure_id,
        data_type="synthesis",
        status="ready",
    ))
    publish_event(NormalizerEvent(
        structure_id=structure_id,
        data_type="autoprotocol",
        status="ready",
    ))

    return 1
