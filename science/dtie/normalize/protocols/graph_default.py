"""graph_default normalization protocol.

Produces a view suitable for GNN graph building:
- Scoped chains only (from ComputationScope)
- Protein atoms only (no ligands, no waters, no nucleic acid)
- Highest-occupancy altloc selected where alternates exist
- Modified residues harmonized to parent comp_id
- Unresolved residues excluded
- Residues with partial_backbone excluded
- Cα projection available (only CA atoms retained if requested)

Requirements: 5.2
"""

from __future__ import annotations

import logging

from science.dtie.ingest.chain_scorer import ComputationScope
from science.dtie.ingest.parser import ParsedChain, ParsedResidue, ParsedStructure
from science.dtie.normalize.protocols.base import (
    NormalizedResidue,
    NormalizedView,
    NormalizationProtocol,
    ProtocolName,
    _select_highest_occupancy_altloc,
    register_protocol,
)

logger = logging.getLogger(__name__)

# Entity types considered protein for filtering
_PROTEIN_ENTITY_TYPES = frozenset({
    "polymer", "polypeptide(l)", "polypeptide(d)", "protein",
})


class GraphDefaultProtocol(NormalizationProtocol):
    """Default normalization for GNN graph building.

    Parameters:
        atoms: "protein_only" — only protein polymer atoms
        altloc: "highest_occupancy" — select highest-occupancy alternate
        modified_residues: "harmonize_to_parent" — use parent comp_id
        unresolved: "exclude" — skip unresolved residues
        projection: "ca_only" — retain only Cα atoms for graph nodes
        partial_backbone: "exclude" — skip residues missing N/CA/C
    """

    name = ProtocolName.GRAPH_DEFAULT
    version = 1
    parameters = {
        "atoms": "protein_only",
        "altloc": "highest_occupancy",
        "modified_residues": "harmonize_to_parent",
        "unresolved": "exclude",
        "projection": "ca_only",
        "partial_backbone": "exclude",
    }

    def apply(
        self,
        parsed: ParsedStructure,
        scope: ComputationScope,
    ) -> NormalizedView:
        """Apply graph_default filtering to produce a GNN-ready view."""
        # Determine which chains are in scope
        included_chains = set(scope.primary_chain_ids)
        excluded_chains = set(scope.exclude_chain_ids)

        result_chains: dict[str, list[NormalizedResidue]] = {}

        for chain in parsed.chains:
            # Filter by scope
            if chain.auth_asym_id in excluded_chains:
                continue
            if included_chains and chain.auth_asym_id not in included_chains:
                continue

            # Protein only
            if chain.entity_type.lower() not in _PROTEIN_ENTITY_TYPES:
                continue

            normalized_residues = self._filter_chain_residues(chain)
            if normalized_residues:
                result_chains[chain.auth_asym_id] = normalized_residues

        return NormalizedView(
            structure_id=parsed.pdb_id,
            protocol_name=self.name.value,
            protocol_version=self.version,
            chains=result_chains,
            parameters=self.parameters.copy(),
        )

    def _filter_chain_residues(
        self, chain: ParsedChain
    ) -> list[NormalizedResidue]:
        """Filter residues for a single chain per graph_default rules."""
        result: list[NormalizedResidue] = []

        for residue in chain.residues:
            # Exclude unresolved
            if not residue.is_resolved:
                continue

            # Exclude partial backbone
            if residue.partial_backbone:
                continue

            # Select highest-occupancy altloc
            filtered_atoms = _select_highest_occupancy_altloc(residue.atoms)

            # Protein atoms only: exclude hetero atoms
            protein_atoms = [a for a in filtered_atoms if not a.is_hetero]

            # Cα projection: keep only CA
            ca_atoms = [a for a in protein_atoms if a.atom_name == "CA"]
            if not ca_atoms:
                # No Cα → skip this residue (can't build graph node)
                continue

            # Harmonize modified residues → use parent comp_id
            comp_id = residue.comp_id
            residue_name_3 = residue.residue_name_3
            if residue.is_modified and residue.parent_comp_id:
                comp_id = residue.parent_comp_id
                residue_name_3 = residue.parent_comp_id

            result.append(NormalizedResidue(
                residue_id=None,
                auth_seq_id=residue.auth_seq_id,
                label_seq_id=residue.label_seq_id,
                insertion_code=residue.insertion_code,
                residue_name=residue.residue_name,
                residue_name_3=residue_name_3,
                comp_id=comp_id,
                parent_comp_id=residue.parent_comp_id,
                is_modified=residue.is_modified,
                atoms=ca_atoms,
            ))

        return result


# Register the protocol
_instance = GraphDefaultProtocol()
register_protocol(_instance)
