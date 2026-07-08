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

from science.dtie.common import ingest_master_features as imf
from science.dtie.common import residue_features as rf
from science.dtie.common.normalizer_payloads import GraphEdge

logger = logging.getLogger(__name__)

# Contact distance threshold for Cα graph (Ångströms)
DEFAULT_CONTACT_CUTOFF = 8.0

# H-bond distance heuristic: Cα–Cα distance below this threshold between
# sequential residues (|i-j| <= 5) suggests a backbone hydrogen bond.
HBOND_CA_DISTANCE_CUTOFF = 5.5

# Secondary structure one-hot mapping for geometric ss_type scalars (training path).
# 0.0 = helix-like, 0.5 = sheet-like, 1.0 = coil (default).
GEOMETRIC_SS_ONEHOT = {
    0.0: 0,
    0.5: 1,
    1.0: 2,
}


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

    def to_pyg(
        self,
        graph: ProteinGraph,
        *,
        gnn_input_mode: rf.GnnInputMode | None = None,
    ) -> Any:
        """Convert ProteinGraph to a torch_geometric Data object.

        Returns a Data object with:
            x: [N, 3|4] node features (ρ, τ, ss_type [, sasa legacy])
            sasa: [N] side-channel (always; for binding scan / training probes)
            edge_index: [2, E] graph connectivity
            edge_attr: [E, 4] edge features (rel_x, rel_y, rel_z, dist)
            degree: [N] node degree from contact graph
            ss_onehot: [N, 3] one-hot secondary structure [helix, sheet, coil]
            rho: [N] raw dehydron density for gate shortcut
            chain_ids: list[str] metadata
            residue_indices: list[int] metadata
            residue_ids: list[str] metadata
        """
        import torch
        from torch_geometric.data import Data

        num_nodes = len(graph.residues)

        rho = np.array([r.rho for r in graph.residues], dtype=np.float64)
        tau = np.array([r.tau_flag for r in graph.residues], dtype=np.float64)
        ss = np.array([r.ss_type for r in graph.residues], dtype=np.float64)
        sasa = np.array([r.sasa for r in graph.residues], dtype=np.float64)
        x_np = rf.stack_gnn_node_features(rho, tau, ss, sasa, mode=gnn_input_mode)
        x = torch.tensor(x_np, dtype=torch.float32)

        edge_index = torch.tensor(graph.edge_index, dtype=torch.long)
        edge_attr = torch.tensor(graph.edge_attr, dtype=torch.float32)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.sasa = torch.tensor(sasa, dtype=torch.float32)

        # --- V6 topological features ---

        # 1. Node degree: count edges per node from edge_index
        degree = torch.zeros(num_nodes, dtype=torch.long)
        if edge_index.numel() > 0:
            # Count outgoing edges per source node (graph is bidirectional,
            # so counting sources gives total degree)
            src_nodes = edge_index[0]
            degree.scatter_add_(0, src_nodes, torch.ones_like(src_nodes, dtype=torch.long))
        data.degree = degree

        # 2. SS one-hot: geometric ss_type scalar → [helix, sheet, coil]
        ss_onehot = torch.zeros(num_nodes, 3, dtype=torch.float32)
        for i, r in enumerate(graph.residues):
            ss_idx = GEOMETRIC_SS_ONEHOT.get(r.ss_type, 2)
            ss_onehot[i, ss_idx] = 1.0
        data.ss_onehot = ss_onehot

        # 3. Raw rho values for gate shortcut
        data.rho = torch.tensor(
            [r.rho for r in graph.residues], dtype=torch.float32
        )

        # --- Metadata ---
        data.chain_ids = graph.chain_ids
        data.residue_indices = graph.residue_indices
        data.residue_ids = graph.residue_ids

        return data

    def extract_edges_for_persistence(
        self,
        graph: ProteinGraph,
        hbond_cutoff: float = HBOND_CA_DISTANCE_CUTOFF,
    ) -> list[GraphEdge]:
        """Convert a ProteinGraph into a list of GraphEdge objects for persistence.

        Deduplicates bidirectional edges (keeps source_index < target_index)
        and classifies edges as 'h_bond' or 'contact' based on a Cα distance
        heuristic: sequential residues (sequence separation <= 5) with Cα
        distance below *hbond_cutoff* are classified as h_bond; all others
        are classified as contact.

        Args:
            graph: A ProteinGraph produced by build_graph.
            hbond_cutoff: Cα distance threshold for H-bond classification.

        Returns:
            List of GraphEdge objects ready for GraphTopologyPayload.
        """
        edge_index = graph.edge_index  # [2, E]
        edge_attr = graph.edge_attr  # [E, 4] (rel_x, rel_y, rel_z, distance)
        residue_ids = graph.residue_ids
        residue_indices = graph.residue_indices

        num_edges = edge_index.shape[1]
        seen: set[tuple[int, int]] = set()
        edges: list[GraphEdge] = []

        for e in range(num_edges):
            src_idx = int(edge_index[0, e])
            tgt_idx = int(edge_index[1, e])

            # Deduplicate: only keep the canonical direction (lower index first)
            key = (min(src_idx, tgt_idx), max(src_idx, tgt_idx))
            if key in seen:
                continue
            seen.add(key)

            distance = float(edge_attr[e, 3])
            src_residue_id = residue_ids[src_idx]
            tgt_residue_id = residue_ids[tgt_idx]

            # Classify edge type using distance + sequence separation heuristic
            seq_sep = abs(residue_indices[src_idx] - residue_indices[tgt_idx])
            if distance < hbond_cutoff and seq_sep <= 5:
                edge_type = "h_bond"
            else:
                edge_type = "contact"

            edges.append(
                GraphEdge(
                    source_residue_id=src_residue_id,
                    target_residue_id=tgt_residue_id,
                    edge_type=edge_type,
                    distance_angstrom=distance,
                    weight=1.0,
                )
            )

        return edges

    @staticmethod
    def build_node_features_from_pdb(
        pdb_path: str,
        chain_id: str,
        *,
        mode: rf.FeatureMode = rf.FeatureMode.TRAINING_CURRENT,
    ) -> list[rf.ResidueNodeFeatures]:
        """Inference-side feature extraction from PDB (P_FEATURE_01 parity path)."""
        return rf.build_from_pdb_chain(pdb_path, chain_id, mode=mode)

    async def _fetch_residues(
        self, structure_id: str, chain_filter: str | None
    ) -> list[ResidueFeatures]:
        """Fetch persisted MASTER features from governed tables (read-only)."""
        chain_clause = ""
        params: dict[str, Any] = {
            "structure_id": structure_id,
            "condition": imf.DEFAULT_CONDITION,
        }

        if chain_filter:
            chain_clause = "AND c.chain_label = :chain_label"
            params["chain_label"] = chain_filter

        rows = await self._db.fetch_all(
            f"""
            SELECT
                r.residue_id,
                r.residue_index,
                c.chain_label,
                r.sasa,
                r.sse_code,
                f.rho,
                f.tau_flag,
                f.ss_type,
                a.x AS ca_x,
                a.y AS ca_y,
                a.z AS ca_z
            FROM dim_residue r
            JOIN dim_chain c ON c.chain_id = r.chain_id
            JOIN fact_ingestion_features f
              ON f.residue_id = r.residue_id
             AND f.structure_id = c.structure_id
             AND f.condition = :condition
             AND f.is_current = TRUE
            LEFT JOIN dim_atom a
              ON a.residue_id = r.residue_id
             AND a.atom_name = 'CA'
            WHERE c.structure_id = :structure_id
              {chain_clause}
              AND r.sasa IS NOT NULL
              AND r.sse_code IS NOT NULL
              AND a.x IS NOT NULL
            ORDER BY c.chain_label, r.residue_index
            """,
            params,
        )

        if not rows:
            return []

        residues: list[ResidueFeatures] = []
        for row in rows:
            sse = row.get("sse_code") or "C"
            residues.append(
                ResidueFeatures(
                    residue_id=row["residue_id"],
                    residue_index=int(row["residue_index"]),
                    chain_label=row["chain_label"],
                    rho=float(row["rho"]),
                    tau_flag=float(row["tau_flag"]),
                    ss_type=float(row["ss_type"]),
                    sasa=float(row["sasa"]),
                    ca_x=float(row["ca_x"]),
                    ca_y=float(row["ca_y"]),
                    ca_z=float(row["ca_z"]),
                )
            )
        return residues

    def _compute_burial_depth(
        self, ca_coords: np.ndarray, radius: float = 10.0
    ) -> np.ndarray:
        """Legacy burial proxy — retained for diagnostics only, not feature path."""
        return rf.compute_burial_rho_skew(ca_coords, radius=radius).astype(np.float32)

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
