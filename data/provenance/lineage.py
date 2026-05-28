# Migrated from: new (Phase 1/2 implementation) on 2026-05-27
"""Provenance lineage traversal and querying.

Provides the ability to answer:
- "What is the full provenance DAG for this residue's embedding?"
- "What runs contributed to this asset?"
- "Show me all assets produced by this run and its children."
- "What is the lineage chain from this output back to raw input?"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol


class ReadOnlyDB(Protocol):
    """Read-only database protocol for provenance queries."""

    async def fetch_one(self, query: str, params: dict[str, Any]) -> dict[str, Any] | None: ...
    async def fetch_all(self, query: str, params: dict[str, Any]) -> list[dict[str, Any]]: ...


@dataclass
class ProvenanceNode:
    """A single node in the provenance DAG."""

    run_id: str
    model_version: str
    pipeline_name: str | None
    run_type: str
    source_type: str
    structure_id: str | None
    started_at: datetime | None
    completed_at: datetime | None
    parent_run_id: str | None
    parameters: dict[str, Any] | None = None
    children: list[ProvenanceNode] = field(default_factory=list)
    assets: list[AssetRecord] = field(default_factory=list)


@dataclass
class AssetRecord:
    """A governed asset linked to a provenance run."""

    asset_id: str
    asset_type: str
    structure_id: str | None
    residue_id: str | None
    created_at: datetime | None


@dataclass
class LineageResult:
    """Complete lineage result for a query."""

    target_id: str
    target_type: str  # "residue", "asset", "run"
    ancestors: list[ProvenanceNode]
    descendants: list[ProvenanceNode]
    total_runs: int
    total_assets: int


class ProvenanceTracer:
    """Query and traverse the provenance DAG.

    Usage:
        tracer = ProvenanceTracer(db=read_connection)
        lineage = await tracer.trace_residue("4obe:A:12")
        ancestors = await tracer.get_ancestors("run_001")
    """

    def __init__(self, db: ReadOnlyDB):
        self._db = db

    async def trace_residue(
        self, residue_id: str, max_depth: int = 10
    ) -> LineageResult:
        """Get the full provenance lineage for a residue's governed data.

        Answers: "What runs produced data for this residue, and what was
        their lineage?"

        Args:
            residue_id: Canonical residue_id to trace.
            max_depth: Maximum ancestor depth to traverse.

        Returns:
            LineageResult with all contributing runs and assets.
        """
        # Find all runs that produced data for this residue
        contributing_runs = await self._db.fetch_all(
            """
            SELECT DISTINCT e.run_id
            FROM fact_gnn_node_embedding e
            WHERE e.residue_id = :residue_id
            UNION
            SELECT DISTINCT p.run_id
            FROM fact_phase3_persistence p
            WHERE p.residue_id = :residue_id
            UNION
            SELECT DISTINCT d.run_id
            FROM fact_dehydron d
            WHERE d.donor_residue_id = :residue_id
               OR d.acceptor_residue_id = :residue_id
            UNION
            SELECT DISTINCT cp.run_id
            FROM fact_computed_property cp
            WHERE cp.residue_id = :residue_id
            """,
            {"residue_id": residue_id},
        )

        ancestors: list[ProvenanceNode] = []
        for row in contributing_runs:
            run_lineage = await self.get_ancestors(row["run_id"], max_depth=max_depth)
            ancestors.extend(run_lineage)

        # Get all assets for this residue
        assets = await self._get_residue_assets(residue_id)

        return LineageResult(
            target_id=residue_id,
            target_type="residue",
            ancestors=ancestors,
            descendants=[],
            total_runs=len(contributing_runs),
            total_assets=len(assets),
        )

    async def trace_asset(self, asset_id: str, max_depth: int = 10) -> LineageResult:
        """Get the full provenance lineage for a specific governed asset.

        Args:
            asset_id: The governed asset ID to trace.
            max_depth: Maximum ancestor depth.

        Returns:
            LineageResult for this asset.
        """
        asset = await self._db.fetch_one(
            "SELECT * FROM governed_asset WHERE asset_id = :asset_id",
            {"asset_id": asset_id},
        )
        if not asset:
            return LineageResult(
                target_id=asset_id,
                target_type="asset",
                ancestors=[],
                descendants=[],
                total_runs=0,
                total_assets=0,
            )

        ancestors = await self.get_ancestors(asset["run_id"], max_depth=max_depth)
        descendants = await self.get_descendants(asset["run_id"], max_depth=max_depth)

        return LineageResult(
            target_id=asset_id,
            target_type="asset",
            ancestors=ancestors,
            descendants=descendants,
            total_runs=len(ancestors) + len(descendants),
            total_assets=1,
        )

    async def get_ancestors(
        self, run_id: str, max_depth: int = 10
    ) -> list[ProvenanceNode]:
        """Traverse the provenance DAG upward (toward inputs/parents).

        Uses recursive CTE to walk parent_run_id links.
        """
        rows = await self._db.fetch_all(
            """
            WITH RECURSIVE lineage AS (
                SELECT run_id, model_version, pipeline_name, run_type,
                       source_type, structure_id, started_at, completed_at,
                       parent_run_id, parameters, 0 AS depth
                FROM provenance_run
                WHERE run_id = :run_id

                UNION ALL

                SELECT p.run_id, p.model_version, p.pipeline_name, p.run_type,
                       p.source_type, p.structure_id, p.started_at, p.completed_at,
                       p.parent_run_id, p.parameters, l.depth + 1
                FROM provenance_run p
                JOIN lineage l ON p.run_id = l.parent_run_id
                WHERE l.depth < :max_depth
            )
            SELECT * FROM lineage ORDER BY depth ASC
            """,
            {"run_id": run_id, "max_depth": max_depth},
        )

        return [self._row_to_node(row) for row in rows]

    async def get_descendants(
        self, run_id: str, max_depth: int = 10
    ) -> list[ProvenanceNode]:
        """Traverse the provenance DAG downward (toward derived outputs).

        Finds all runs that have this run as an ancestor.
        """
        rows = await self._db.fetch_all(
            """
            WITH RECURSIVE lineage AS (
                SELECT run_id, model_version, pipeline_name, run_type,
                       source_type, structure_id, started_at, completed_at,
                       parent_run_id, parameters, 0 AS depth
                FROM provenance_run
                WHERE run_id = :run_id

                UNION ALL

                SELECT c.run_id, c.model_version, c.pipeline_name, c.run_type,
                       c.source_type, c.structure_id, c.started_at, c.completed_at,
                       c.parent_run_id, c.parameters, l.depth + 1
                FROM provenance_run c
                JOIN lineage l ON c.parent_run_id = l.run_id
                WHERE l.depth < :max_depth
            )
            SELECT * FROM lineage ORDER BY depth ASC
            """,
            {"run_id": run_id, "max_depth": max_depth},
        )

        return [self._row_to_node(row) for row in rows]

    async def get_run_assets(self, run_id: str) -> list[AssetRecord]:
        """Get all governed assets produced by a specific run."""
        rows = await self._db.fetch_all(
            """
            SELECT asset_id, asset_type, structure_id, residue_id, created_at
            FROM governed_asset
            WHERE run_id = :run_id
            ORDER BY created_at
            """,
            {"run_id": run_id},
        )
        return [
            AssetRecord(
                asset_id=r["asset_id"],
                asset_type=r["asset_type"],
                structure_id=r.get("structure_id"),
                residue_id=r.get("residue_id"),
                created_at=r.get("created_at"),
            )
            for r in rows
        ]

    async def compare_lineages(
        self, run_id_a: str, run_id_b: str
    ) -> dict[str, Any]:
        """Compare the provenance of two runs (useful for v3 vs v4 comparison).

        Returns shared ancestors, divergence point, and differences.
        """
        ancestors_a = await self.get_ancestors(run_id_a)
        ancestors_b = await self.get_ancestors(run_id_b)

        ids_a = {n.run_id for n in ancestors_a}
        ids_b = {n.run_id for n in ancestors_b}

        return {
            "run_a": run_id_a,
            "run_b": run_id_b,
            "shared_ancestors": list(ids_a & ids_b),
            "unique_to_a": list(ids_a - ids_b),
            "unique_to_b": list(ids_b - ids_a),
            "a_depth": len(ancestors_a),
            "b_depth": len(ancestors_b),
        }

    async def _get_residue_assets(self, residue_id: str) -> list[AssetRecord]:
        """Get all governed assets associated with a residue."""
        rows = await self._db.fetch_all(
            """
            SELECT asset_id, asset_type, structure_id, residue_id, created_at
            FROM governed_asset
            WHERE residue_id = :residue_id
            ORDER BY created_at
            """,
            {"residue_id": residue_id},
        )
        return [
            AssetRecord(
                asset_id=r["asset_id"],
                asset_type=r["asset_type"],
                structure_id=r.get("structure_id"),
                residue_id=r.get("residue_id"),
                created_at=r.get("created_at"),
            )
            for r in rows
        ]

    def _row_to_node(self, row: dict[str, Any]) -> ProvenanceNode:
        return ProvenanceNode(
            run_id=row["run_id"],
            model_version=row["model_version"],
            pipeline_name=row.get("pipeline_name"),
            run_type=row["run_type"],
            source_type=row["source_type"],
            structure_id=row.get("structure_id"),
            started_at=row.get("started_at"),
            completed_at=row.get("completed_at"),
            parent_run_id=row.get("parent_run_id"),
            parameters=row.get("parameters"),
        )
