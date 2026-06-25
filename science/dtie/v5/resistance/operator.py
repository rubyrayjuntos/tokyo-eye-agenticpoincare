"""Mutation Operator — applies virtual mutations to PyG graph data.

This module implements the physicochemically-grounded virtual mutation engine.
It modifies node features and edge attributes to simulate amino acid substitutions
without altering graph connectivity (node count and edge count are preserved).

Key operations:
- compute_perturbation_factor: Steric + electrostatic scaling from AA properties
- apply_perturbation: In-place feature/edge modification on a PyG Data clone
- build_mutant_graph: End-to-end graph construction with mutation applied
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from science.dtie.common.graph_builder import GraphBuilder
from science.dtie.v5.resistance.models import (
    AMINO_ACID_PROPERTIES,
    MutationSpec,
    compute_perturbation_factor,
    mutant_structure_id,
)


class MutationOperator:
    """Applies virtual mutations to PyG graph data.

    The operator modifies:
    - Node features at the mutation site (volume-scaled rho)
    - Edge attributes for edges incident to the mutated node
      (scaled by combined perturbation factor)

    The operator preserves:
    - Graph connectivity (edge_index unchanged)
    - Node count and edge count
    - Features of non-mutated nodes
    """

    def __init__(self, graph_builder: GraphBuilder):
        self._builder = graph_builder

    async def build_mutant_graph(
        self,
        structure_id: str,
        mutation: MutationSpec,
    ) -> tuple[Any, int]:
        """Build a mutated PyG graph.

        Fetches the wild-type graph from the database, locates the target
        residue, and applies the virtual mutation.

        Args:
            structure_id: Canonical structure_id for the wild-type protein.
            mutation: The amino acid substitution to apply.

        Returns:
            Tuple of (mutated PyG Data, graph index of mutated residue).

        Raises:
            ValueError: If the specified residue_index does not exist in the
                target chain.
        """
        # Build the WT graph
        protein_graph = await self._builder.build_graph(structure_id)
        pyg_data = self._builder.to_pyg(protein_graph)

        # Find the node index for the target residue
        node_idx = self._find_residue_node(
            pyg_data, mutation.chain, mutation.residue_index
        )

        # Apply the mutation
        mutated_data = self.apply_perturbation(pyg_data, node_idx, mutation)

        return mutated_data, node_idx

    def apply_perturbation(
        self,
        pyg_data: Any,
        node_idx: int,
        mutation: MutationSpec,
    ) -> Any:
        """Apply feature and edge perturbations to simulate a mutation.

        Creates a deep copy of the graph data and modifies:
        - Node features at node_idx: rho scaled by volume ratio
        - Edge attributes for edges incident to node_idx: distance scaled
          by combined perturbation factor

        Does NOT modify:
        - Graph connectivity (edge_index unchanged)
        - Other nodes' features
        - Edge attributes for non-incident edges

        Args:
            pyg_data: PyG Data object with x, edge_index, edge_attr.
            node_idx: Graph index of the residue to mutate.
            mutation: The amino acid substitution specification.

        Returns:
            A new PyG Data object with perturbations applied.
        """
        import torch

        # Clone to avoid modifying the original
        mutated = pyg_data.clone()

        # Compute perturbation factors
        steric_factor, electrostatic_factor = compute_perturbation_factor(
            mutation.wild_type_aa, mutation.mutant_aa
        )
        combined_factor = steric_factor * electrostatic_factor

        # --- Modify node features at mutation site ---
        # Node features are [rho, tau_flag, ss_type, sasa]
        # Scale rho by volume ratio (steric factor) to reflect size change
        mutated.x[node_idx, 0] = mutated.x[node_idx, 0] * steric_factor

        # Update hydropathy in the feature vector if we have room
        # The node feature is [rho, tau_flag, ss_type, sasa] — we scale rho
        # which is the primary physicochemical signal at the node level.

        # --- Modify edge attributes for incident edges ---
        # edge_index is [2, E], edge_attr is [E, 4] (rel_x, rel_y, rel_z, dist)
        edge_index = mutated.edge_index  # [2, E]
        edge_attr = mutated.edge_attr  # [E, 4]

        # Find edges incident to the mutated node (either source or target)
        src_mask = edge_index[0] == node_idx
        tgt_mask = edge_index[1] == node_idx
        incident_mask = src_mask | tgt_mask

        # Scale the distance component (column 3) by combined factor
        # Also scale the relative position components (columns 0-2) to maintain
        # geometric consistency
        edge_attr[incident_mask] = edge_attr[incident_mask] * combined_factor

        mutated.edge_attr = edge_attr

        return mutated

    def _find_residue_node(
        self,
        pyg_data: Any,
        chain: str,
        residue_index: int,
    ) -> int:
        """Find the graph node index for a given chain + residue_index.

        Args:
            pyg_data: PyG Data object with chain_ids and residue_indices metadata.
            chain: Chain label (e.g., 'A').
            residue_index: Residue sequence number.

        Returns:
            The graph node index.

        Raises:
            ValueError: If the residue is not found in the graph.
        """
        chain_ids = getattr(pyg_data, "chain_ids", None)
        residue_indices = getattr(pyg_data, "residue_indices", None)

        if chain_ids is None or residue_indices is None:
            raise ValueError(
                "PyG Data object missing chain_ids or residue_indices metadata"
            )

        # Search for matching chain + residue_index
        for idx, (c, r) in enumerate(zip(chain_ids, residue_indices)):
            if c == chain and r == residue_index:
                return idx

        # Not found — build informative error message
        chain_residues = [
            r for c, r in zip(chain_ids, residue_indices) if c == chain
        ]
        if chain_residues:
            min_r, max_r = min(chain_residues), max(chain_residues)
            raise ValueError(
                f"Residue {chain}:{residue_index} not found in graph. "
                f"Chain {chain} has residues {min_r}-{max_r}."
            )
        else:
            available_chains = sorted(set(chain_ids))
            raise ValueError(
                f"Chain '{chain}' not found in graph. "
                f"Available chains: {available_chains}"
            )
