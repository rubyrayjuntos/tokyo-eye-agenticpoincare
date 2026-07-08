"""Persist FeatureMode.MASTER outputs to governed tables.

Writes:
  - dim_residue.sasa (FreeSASA Å²)
  - dim_residue.sse_code (DSSP-class H/E/C)
  - fact_ingestion_features (rho, tau_flag, ss_type, sasa) for audit/replay

All feature *values* come from science.dtie.common.residue_features — this module
only handles DB I/O.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from science.dtie.common import residue_features as rf
from science.dtie.common.keys import make_residue_id, make_structure_id

logger = logging.getLogger(__name__)

DEFAULT_CONDITION = "apo"


def attach_canonical_residue_ids(
    features: Sequence[rf.ResidueNodeFeatures],
    structure_id: str,
) -> list[rf.ResidueNodeFeatures]:
    """Attach canonical residue_id keys for dim_residue updates."""
    out: list[rf.ResidueNodeFeatures] = []
    for feat in features:
        rid = make_residue_id(structure_id, feat.chain_label, feat.residue_index)
        out.append(replace(feat, residue_id=rid))
    return out


def compute_master_features_from_pdb(
    pdb_path: Path | str,
    chain_id: str,
    structure_id: str,
) -> list[rf.ResidueNodeFeatures]:
    """Compute MASTER features from PDB and attach canonical residue_ids."""
    feats = rf.build_from_pdb_chain(pdb_path, chain_id, mode=rf.FeatureMode.MASTER)
    if not feats:
        raise ValueError(f"No valid residues for {structure_id} chain {chain_id} in {pdb_path}")
    return attach_canonical_residue_ids(feats, structure_id)


async def structure_exists(db: Any, structure_id: str) -> bool:
    row = await db.fetch_one(
        "SELECT structure_id FROM dim_structure WHERE structure_id = :sid",
        {"sid": structure_id},
    )
    return row is not None


async def count_dim_residues_for_structure(db: Any, structure_id: str) -> int:
    row = await db.fetch_one(
        """
        SELECT COUNT(*) AS n
        FROM dim_residue r
        JOIN dim_chain c ON c.chain_id = r.chain_id
        WHERE c.structure_id = :sid
        """,
        {"sid": structure_id},
    )
    return int(row["n"]) if row else 0


async def persist_master_features(
    db: Any,
    structure_id: str,
    features: Sequence[rf.ResidueNodeFeatures],
    *,
    run_id: str | None = None,
    condition: str = DEFAULT_CONDITION,
    write_ingestion_facts: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Update dim_residue and optionally fact_ingestion_features."""
    if not features:
        raise ValueError(f"No features to persist for {structure_id}")

    effective_run_id = run_id or f"master_ingest_{uuid.uuid4().hex[:12]}"
    dim_updates = [
        {
            "residue_id": feat.residue_id,
            "sasa": float(feat.sasa),
            "sse_code": feat.sse_code,
        }
        for feat in features
        if feat.residue_id and feat.sse_code
    ]
    if len(dim_updates) != len(features):
        raise ValueError(f"{structure_id}: missing residue_id or sse_code on feature rows")

    if dry_run:
        return {
            "structure_id": structure_id,
            "run_id": effective_run_id,
            "residue_count": len(features),
            "dry_run": True,
        }

    missing: list[str] = []
    for row in dim_updates:
        exists = await db.fetch_one(
            "SELECT residue_id FROM dim_residue WHERE residue_id = :residue_id",
            {"residue_id": row["residue_id"]},
        )
        if not exists:
            missing.append(row["residue_id"])
    if missing:
        raise RuntimeError(
            f"{structure_id}: {len(missing)} residue_id(s) not found in dim_residue — "
            "ingest structure dimensions first (POST /api/ingest). "
            f"Example missing: {missing[:3]}"
        )

    await db.execute_many(
        """
        UPDATE dim_residue
        SET sasa = :sasa,
            sse_code = :sse_code,
            updated_at = NOW()
        WHERE residue_id = :residue_id
        """,
        dim_updates,
    )

    facts_written = 0
    if write_ingestion_facts:
        await db.execute(
            """
            UPDATE fact_ingestion_features
            SET is_current = FALSE
            WHERE structure_id = :structure_id
              AND condition = :condition
              AND is_current = TRUE
            """,
            {"structure_id": structure_id, "condition": condition},
        )
        fact_rows = [
            {
                "run_id": effective_run_id,
                "structure_id": structure_id,
                "condition": condition,
                "residue_id": feat.residue_id,
                "residue_index": feat.residue_index,
                "rho": float(feat.rho),
                "tau_flag": float(feat.tau_flag),
                "sasa": float(feat.sasa),
                "ss_type": float(feat.ss_type),
            }
            for feat in features
        ]
        await db.execute_many(
            """
            INSERT INTO fact_ingestion_features (
                run_id, structure_id, condition, residue_id, residue_index,
                rho, tau_flag, sasa, ss_type, is_current, computed_at
            ) VALUES (
                :run_id, :structure_id, :condition, :residue_id, :residue_index,
                :rho, :tau_flag, :sasa, :ss_type, TRUE, NOW()
            )
            """,
            fact_rows,
        )
        facts_written = len(fact_rows)

    return {
        "structure_id": structure_id,
        "run_id": effective_run_id,
        "residue_count": len(features),
        "dim_residue_updated": len(dim_updates),
        "fact_ingestion_features_written": facts_written,
        "dry_run": False,
    }


async def verify_persisted_master_features(
    db: Any,
    structure_id: str,
    chain_label: str,
    pdb_path: Path | str,
) -> list[rf.FeatureParityMismatch]:
    """Recompute from PDB and compare to dim_residue + fact_ingestion_features."""
    expected = compute_master_features_from_pdb(pdb_path, chain_label, structure_id)
    rows = await db.fetch_all(
        """
        SELECT r.residue_id, r.residue_index, c.chain_label,
               r.sasa, r.sse_code,
               f.rho, f.tau_flag, f.ss_type AS fact_ss_type
        FROM dim_residue r
        JOIN dim_chain c ON c.chain_id = r.chain_id
        LEFT JOIN fact_ingestion_features f
          ON f.residue_id = r.residue_id
         AND f.structure_id = c.structure_id
         AND f.is_current = TRUE
         AND f.condition = :condition
        WHERE c.structure_id = :structure_id
          AND c.chain_label = :chain_label
        ORDER BY r.residue_index
        """,
        {
            "structure_id": structure_id,
            "chain_label": chain_label,
            "condition": DEFAULT_CONDITION,
        },
    )
    db_feats: list[rf.ResidueNodeFeatures] = []
    for row in rows:
        sse = row.get("sse_code") or "C"
        db_feats.append(
            rf.ResidueNodeFeatures(
                chain_label=chain_label,
                residue_index=int(row["residue_index"]),
                residue_id=str(row["residue_id"]),
                rho=float(row.get("rho") or 0.0),
                tau_flag=float(row.get("tau_flag") or 0.0),
                ss_type=float(row["fact_ss_type"])
                if row.get("fact_ss_type") is not None
                else rf.encode_ss_type_from_sse_code(sse),
                sasa=float(row.get("sasa") or 0.0),
                sse_code=str(sse),
            )
        )
    return rf.compare_feature_parity(expected, db_feats)


def structure_id_for_pdb(pdb_id: str) -> str:
    return make_structure_id(pdb_id=pdb_id, source="rcsb")
