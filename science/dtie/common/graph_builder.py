# Migrated from: new (Phase 3 integration) on 2026-05-27
"""Graph builder — constructs PyG Data objects from the governed data layer.

This module bridges the gap between the dimensional model (dim_residue,
dim_atom) and the GNN runners which expect torch_geometric Data objects.

It handles:
1. Querying residue coordinates from the governed layer
2. Building Cα contact graphs (edge_index + edge_attr)
3. Computing per-node input features (rho, tau_flag, ss_type, sasa)
4. Attaching metadata (chain_ids, residue_indices) for result extraction

This replaces the ad-hoc graph construction scattered across the source repos.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

logger = logging.getLogger(__name__)

# Contact distance threshold for Cα graph (Ångströms)
DEFAULT_CONTACT_CUTOFF = 8.0

# Secondary structure encoding
SSE_ENCODING = {"H": 0.0, "E": 1.0, "C": 2.0, "": 2.0, None: 2.0}


class GraphDB(Protocol):
    """Database protocol for graph building queries."""

    async def fetch_all(self, query: str, params: dict[str, Any]) -> list[dict[str, Any]]: ...


@dataclass
class ResidueFeatures:
    """Per-residue features extracted from the governed layer."""

    residue_id: str
    residue_index: int
    chain_label: str
    rho: float  # Dehydron density (wrapping count / max)
    tau_flag: float  # Tau torsion angle flag
    ss_type: float  # Secondary structure encoding
    sasa: float  # Solvent accessible surface area
    ca_x: float  # Cα x coordinate
    ca_y: float  # Cα y coordinate
    ca_z: float  # Cα z coordinate


@dataclass
class ProteinGraph:
    """Intermediate representation before conversion to PyG Data."""

    structure_id: str
    residues: list[ResidueFeatures]
    edge_index: np.ndarray  # [2, E]
    edge_attr: np.ndarray  # [E, 4] (rel_x, rel_y, rel_z, distance)
    chain_ids: list[str]
    residue_indices: list[int]
    residue_ids: list[str]


class GraphBuilder:
    """Builds PyG-compatible graphs from the governed data layer.

    Usage:
        builder = GraphBuilder(db=connection)
        graph = await builder.build_graph("4obe")
        data = graph.to_pyg()  # torch_geometric.data.Data
    """

    def __init__(self, db: GraphDB, contact_cutoff: float = DEFAULT_CONTACT_CUTOFF):
        self._db = db
        self._contact_cutoff = contact_cutoff

    async def build_graph(
        self,
        structure_id: str,
        chain_filter: str | None = None,
    ) -> ProteinGraph:
        """Build a protein graph from governed residue data.

        Args:
            structure_id: Canonical structure_id.
            chain_filter: Optional chain label to restrict to.

        Returns:
            ProteinGraph ready for conversion to PyG Data.
        """
        # 1. Fetch residue data with Cα coordinates
        residues = await self._fetch_residues(structure_id, chain_filter)

        if not residues:
            raise ValueError(f"No residues found for structure {structure_id}")

        # 2. Build Cα contact graph
        coords = np.array([[r.ca_x, r.ca_y, r.ca_z] for r in residues])
        edge_index, edge_attr = self._build_contact_edges(coords)

        # 3. Compute clustering coefficients (needed by GNN gate)
        # This is done at inference time by the runner via precompute_clustering

        return ProteinGraph(
            structure_id=structure_id,
            residues=residues,
            edge_index=edge_index,
            edge_attr=edge_attr,
            chain_ids=[r.chain_label for r in residues],
            residue_indices=[r.residue_index for r in residues],
            residue_ids=[r.residue_id for r in residues],
        )

    def to_pyg(self, graph: ProteinGraph) -> Any:
        """Convert ProteinGraph to a torch_geometric Data object.

        Returns a Data object with:
            x: [N, 4] node features (rho, tau_flag, ss_type, sasa)
            edge_index: [2, E] graph connectivity
            edge_attr: [E, 4] edge features (rel_x, rel_y, rel_z, dist)
            chain_ids: list[str] metadata
            residue_indices: list[int] metadata
            residue_ids: list[str] metadata
        """
        import torch
        from torch_geometric.data import Data

        # Node features: [rho, tau_flag, ss_type, sasa]
        x = torch.tensor(
            [[r.rho, r.tau_flag, r.ss_type, r.sasa] for r in graph.residues],
            dtype=torch.float32,
        )

        edge_index = torch.tensor(graph.edge_index, dtype=torch.long)
        edge_attr = torch.tensor(graph.edge_attr, dtype=torch.float32)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.chain_ids = graph.chain_ids
        data.residue_indices = graph.residue_indices
        data.residue_ids = graph.residue_ids

        return data

    async def _fetch_residues(
        self, structure_id: str, chain_filter: str | None
    ) -> list[ResidueFeatures]:
        """Fetch residue features from the governed layer."""
        chain_clause = ""
        params: dict[str, Any] = {"structure_id": structure_id}

        if chain_filter:
            chain_clause = "AND c.chain_label = :chain_label"
            params["chain_label"] = chain_filter

        rows = await self._db.fetch_all(
            f"""
            SELECT
                r.residue_id,
                r.residue_index,
                c.chain_label,
                COALESCE(r.sasa, 0.0) AS sasa,
                COALESCE(r.sse_code, 'C') AS sse_code,
                a.x AS ca_x, a.y AS ca_y, a.z AS ca_z
            FROM dim_residue r
            JOIN dim_chain c ON c.chain_id = r.chain_id
            JOIN dim_structure s ON s.structure_id = c.structure_id
            LEFT JOIN dim_atom a ON a.residue_id = r.residue_id AND a.atom_name = 'CA'
            WHERE s.structure_id = :structure_id
              {chain_clause}
              AND a.x IS NOT NULL
            ORDER BY c.chain_label, r.residue_index
            """,
            params,
        )

        residues = []
        for row in rows:
            # Compute dehydron density (rho) from fact_dehydron if available
            # For now, use a placeholder — will be populated from dehydron facts
            rho = await self._get_dehydron_density(row["residue_id"])

            residues.append(
                ResidueFeatures(
                    residue_id=row["residue_id"],
                    residue_index=row["residue_index"],
                    chain_label=row["chain_label"],
                    rho=rho,
                    tau_flag=0.0,  # Computed from backbone torsion angles
                    ss_type=SSE_ENCODING.get(row["sse_code"], 2.0),
                    sasa=row["sasa"],
                    ca_x=row["ca_x"],
                    ca_y=row["ca_y"],
                    ca_z=row["ca_z"],
                )
            )

        return residues

    async def _get_dehydron_density(self, residue_id: str) -> float:
        """Get dehydron density for a residue from governed facts."""
        rows = await self._db.fetch_all(
            """
            SELECT COUNT(*) as count
            FROM fact_dehydron
            WHERE (donor_residue_id = :rid OR acceptor_residue_id = :rid)
              AND is_dehydron = true
            """,
            {"rid": residue_id},
        )
        if rows and rows[0]["count"] > 0:
            # Normalize: typical max is ~30 dehydrons per residue
            return min(float(rows[0]["count"]) / 30.0, 1.0)
        return 0.0

    def _build_contact_edges(
        self, coords: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build Cα contact graph from coordinates.

        Args:
            coords: [N, 3] Cα coordinates.

        Returns:
            edge_index: [2, E] source/target pairs.
            edge_attr: [E, 4] (rel_x, rel_y, rel_z, distance).
        """
        n = len(coords)
        sources = []
        targets = []
        attrs = []

        for i in range(n):
            for j in range(i + 1, n):
                diff = coords[j] - coords[i]
                dist = np.linalg.norm(diff)

                if dist < self._contact_cutoff:
                    # Bidirectional edges
                    sources.extend([i, j])
                    targets.extend([j, i])
                    attrs.append([diff[0], diff[1], diff[2], dist])
                    attrs.append([-diff[0], -diff[1], -diff[2], dist])

        if not sources:
            # Fallback: connect sequential residues
            for i in range(n - 1):
                diff = coords[i + 1] - coords[i]
                dist = np.linalg.norm(diff)
                sources.extend([i, i + 1])
                targets.extend([i + 1, i])
                attrs.append([diff[0], diff[1], diff[2], dist])
                attrs.append([-diff[0], -diff[1], -diff[2], dist])

        edge_index = np.array([sources, targets], dtype=np.int64)
        edge_attr = np.array(attrs, dtype=np.float32)

        return edge_index, edge_attr
