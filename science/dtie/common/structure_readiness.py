"""Ensure structures are fully ingested with MASTER features before training/inference read.

Single writer path: ingest dimensions → compute MASTER features from dim_atom → persist.
Readers (training, GraphBuilder) load from DB only.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from science.dtie.common import ingest_master_features as imf
from science.dtie.common import residue_features as rf
from science.dtie.common.keys import make_structure_id

logger = logging.getLogger(__name__)

MIN_RESIDUES = 10
MIN_RESIDUES_FOR_SSE_DIVERSITY = 50


@dataclass(frozen=True)
class MasterFeaturesReadiness:
    structure_id: str
    chain_label: str
    residue_count: int
    with_sasa_sse: int
    with_ingestion_facts: int
    ready: bool
    reason: str


async def check_master_features_ready(
    db: Any,
    structure_id: str,
    chain_label: str,
) -> MasterFeaturesReadiness:
    """True when chain has dim rows + populated sasa/sse_code + current ingestion facts."""
    row = await db.fetch_one(
        """
        SELECT
            COUNT(*)::int AS residue_count,
            COUNT(*) FILTER (
                WHERE r.sasa IS NOT NULL AND r.sse_code IS NOT NULL
            )::int AS with_sasa_sse,
            COUNT(*) FILTER (
                WHERE EXISTS (
                    SELECT 1 FROM fact_ingestion_features f
                    WHERE f.residue_id = r.residue_id
                      AND f.structure_id = c.structure_id
                      AND f.condition = :condition
                      AND f.is_current = TRUE
                      AND f.rho IS NOT NULL
                )
            )::int AS with_ingestion_facts
        FROM dim_residue r
        JOIN dim_chain c ON c.chain_id = r.chain_id
        WHERE c.structure_id = :structure_id
          AND c.chain_label = :chain_label
          AND EXISTS (
              SELECT 1 FROM dim_atom a
              WHERE a.residue_id = r.residue_id
                AND a.atom_name = 'CA'
                AND a.x IS NOT NULL
          )
        """,
        {
            "structure_id": structure_id,
            "chain_label": chain_label,
            "condition": imf.DEFAULT_CONDITION,
        },
    )
    if not row or int(row["residue_count"]) == 0:
        return MasterFeaturesReadiness(
            structure_id=structure_id,
            chain_label=chain_label,
            residue_count=0,
            with_sasa_sse=0,
            with_ingestion_facts=0,
            ready=False,
            reason="no_dim_residue_rows",
        )

    total = int(row["residue_count"])
    with_feats = int(row["with_sasa_sse"])
    with_facts = int(row["with_ingestion_facts"])
    if total < MIN_RESIDUES:
        return MasterFeaturesReadiness(
            structure_id=structure_id,
            chain_label=chain_label,
            residue_count=total,
            with_sasa_sse=with_feats,
            with_ingestion_facts=with_facts,
            ready=False,
            reason=f"too_few_residues ({total} < {MIN_RESIDUES})",
        )
    if with_feats != total or with_facts != total:
        return MasterFeaturesReadiness(
            structure_id=structure_id,
            chain_label=chain_label,
            residue_count=total,
            with_sasa_sse=with_feats,
            with_ingestion_facts=with_facts,
            ready=False,
            reason="master_features_incomplete",
        )
    return MasterFeaturesReadiness(
        structure_id=structure_id,
        chain_label=chain_label,
        residue_count=total,
        with_sasa_sse=with_feats,
        with_ingestion_facts=with_facts,
        ready=True,
        reason="ok",
    )


async def fetch_atom_rows_for_structure(
    db: Any,
    structure_id: str,
    chain_label: str | None = None,
) -> list[dict[str, Any]]:
    chain_clause = ""
    params: dict[str, Any] = {"structure_id": structure_id}
    if chain_label:
        chain_clause = "AND c.chain_label = :chain_label"
        params["chain_label"] = chain_label

    return await db.fetch_all(
        f"""
        SELECT
            r.residue_id,
            r.residue_index,
            c.chain_label,
            r.residue_name,
            r.residue_name_3,
            a.atom_name,
            a.element,
            a.x,
            a.y,
            a.z
        FROM dim_residue r
        JOIN dim_chain c ON c.chain_id = r.chain_id
        LEFT JOIN dim_atom a ON a.residue_id = r.residue_id
        WHERE c.structure_id = :structure_id
          {chain_clause}
        ORDER BY c.chain_label, r.residue_index, a.atom_name
        """,
        params,
    )


async def compute_master_features_from_db(
    db: Any,
    structure_id: str,
    chain_label: str,
) -> list[rf.ResidueNodeFeatures]:
    """Compute MASTER features from governed dim_atom rows (single writer path)."""
    rows = await fetch_atom_rows_for_structure(db, structure_id, chain_label)
    records = rf.records_from_db_atom_rows(rows)
    feats = rf.build_node_features(records, mode=rf.FeatureMode.MASTER)
    if not feats:
        raise ValueError(f"No MASTER features computed for {structure_id}:{chain_label}")
    return imf.attach_canonical_residue_ids(feats, structure_id)


async def persist_master_features_for_chain(
    db: Any,
    structure_id: str,
    chain_label: str,
    *,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Compute from dim_atom + persist to dim_residue / fact_ingestion_features."""
    features = await compute_master_features_from_db(db, structure_id, chain_label)
    return await imf.persist_master_features(
        db,
        structure_id,
        features,
        run_id=run_id,
    )


async def run_ingest_dimensions(
    pdb_id: str,
    *,
    force_reingest: bool = False,
    requested_by: str = "structure_readiness",
) -> dict[str, Any]:
    """In-process ingest-full (structure dimensions)."""
    from fastapi import BackgroundTasks

    from science.api.routers.ingest import IngestRequest, ingest_full

    bg = BackgroundTasks()
    response = await ingest_full(
        IngestRequest(
            pdb_id=pdb_id.upper().strip(),
            force_reingest=force_reingest,
            requested_by=requested_by,
        ),
        bg,
    )
    payload = response.model_dump() if hasattr(response, "model_dump") else dict(response)
    for task in bg.tasks:
        try:
            if asyncio.iscoroutinefunction(task.func):
                await task.func(*task.args, **task.kwargs)
            else:
                task.func(*task.args, **task.kwargs)
        except Exception as exc:
            logger.warning("Background ingest task failed (non-fatal): %s", exc)
    return payload


async def ensure_structure_ready(
    db: Any,
    pdb_id: str,
    chain_label: str,
    pdb_dir: Path | None = None,
    *,
    force_reingest: bool = False,
    run_onboard_pipeline: bool | None = None,
) -> str:
    """Ensure structure dims + MASTER features exist; return structure_id.

    1. Ingest dimensions if missing (ingest-full).
    2. Compute + persist MASTER features from dim_atom if incomplete.
    3. Optionally queue onboard compute (env TRAINING_RUN_ONBOARD=1).
    """
    pdb_id = pdb_id.upper().strip()
    structure_id = make_structure_id(pdb_id=pdb_id, source="rcsb")

    dims_ok = await imf.structure_exists(db, structure_id)
    if not dims_ok or force_reingest:
        logger.info("Ingesting dimensions for %s (force=%s)", pdb_id, force_reingest)
        result = await run_ingest_dimensions(pdb_id, force_reingest=force_reingest)
        structure_id = str(result.get("structure_id") or structure_id)
        if result.get("audit_only"):
            logger.info("Ingest audit-only for %s — using existing %s", pdb_id, structure_id)

    readiness = await check_master_features_ready(db, structure_id, chain_label)
    if not readiness.ready:
        logger.info(
            "Persisting MASTER features for %s:%s (%s)",
            pdb_id,
            chain_label,
            readiness.reason,
        )
        await persist_master_features_for_chain(db, structure_id, chain_label)
        readiness = await check_master_features_ready(db, structure_id, chain_label)
        if not readiness.ready:
            raise RuntimeError(
                f"{pdb_id}:{chain_label} still not ready after persist: {readiness.reason}"
            )

    if run_onboard_pipeline is None:
        run_onboard_pipeline = os.environ.get("TRAINING_RUN_ONBOARD", "").lower() in (
            "1",
            "true",
            "yes",
        )
    if run_onboard_pipeline:
        from science.dtie.ingest.orchestrator import run_onboard_compute

        logger.info("Running onboard compute for %s", structure_id)
        await run_onboard_compute(structure_id)

    return structure_id
