"""interface normalization protocol.

Produces a view suitable for protein-protein interface analysis:
- Multi-chain: all protein chains included (no single-chain restriction)
- Biological assembly aware (uses assembly_id if available)
- Entity-instance collapse disabled (all duplicate chains retained)
- All protein atoms retained (not just Cα)

Requirements: 5.5
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

# Entity types considered protein
_PROTEIN_ENTITY_TYPES = frozenset({
    "polymer", "polypeptide(l)", "polypeptide(d)", "protein",
})


class InterfaceProtocol(NormalizationProtocol):
    """Normalization for protein-protein interface analysis.

    Parameters:
        chains: "multi_chain" — include all protein chains
        assembly: "biological" — use biological assembly context
        entity_collapse: False — do NOT collapse duplicate entity instances
    """

    name = ProtocolName.INTERFACE
    version = 1
    parameters = {
        "chains": "multi_chain",
        "assembly": "biological",
        "entity_collapse": False,
    }

    def apply(
        self,
        parsed: ParsedStructure,
        scope: ComputationScope,
    ) -> NormalizedView:
        """Apply interface filtering — multi-chain, no entity collapse."""
        # For interface: use all protein chains from scope's primary list
        # Entity collapse is disabled — all chains retained
        included_chains = set(scope.primary_chain_ids)

        result_chains: dict[str, list[NormalizedResidue]] = {}

        for chain in parsed.chains:
            # Only include protein chains
            if chain.entity_type.lower() not in _PROTEIN_ENTITY_TYPES:
                continue

            # If scope specifies primary chains, use them; otherwise include all protein
            if included_chains and chain.auth_asym_id not in included_chains:
                continue

            normalized = self._filter_chain_residues(chain)
            if normalized:
                result_chains[chain.auth_asym_id] = normalized

        return NormalizedView(
            structure_id=parsed.pdb_id,
            protocol_name=self.name.value,
            protocol_version=self.version,
            chains=result_chains,
            parameters=self.parameters.copy(),
            metadata={
                "assembly_id": parsed.assembly_id,
                "chain_count": len(result_chains),
                "entity_collapse": False,
            },
        )

    def _filter_chain_residues(
        self, chain: ParsedChain
    ) -> list[NormalizedResidue]:
        """Filter residues for interface analysis.

        Keeps all resolved protein residues with all atoms (not just Cα).
        Partial backbone residues are retained for interface detection.
        """
        result: list[NormalizedResidue] = []

        for residue in chain.residues:
            # Exclude unresolved
            if not residue.is_resolved:
                continue

            # Select best altloc
            filtered_atoms = _select_highest_occupancy_altloc(residue.atoms)

            # Keep all non-hetero atoms
            protein_atoms = [a for a in filtered_atoms if not a.is_hetero]
            if not protein_atoms:
                continue

            # Harmonize modified residues
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
                atoms=protein_atoms,
            ))

        return result


# Register
_instance = InterfaceProtocol()
register_protocol(_instance)
