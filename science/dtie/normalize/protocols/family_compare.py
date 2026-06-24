"""family_compare normalization protocol.

Produces a view suitable for cross-structure family comparison:
- SIFTS-mapped residues only (requires UniProt alignment)
- UniProt position intersection between query and reference
- Reference superposition applied (uses stored rotation/translation)
- Tags and tails excluded (residues outside UniProt-mapped range)
- Mutation sites retained (for comparison purposes)

Requirements: 5.3
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


class FamilyCompareProtocol(NormalizationProtocol):
    """Normalization for cross-structure family comparison.

    Parameters:
        residues: "sifts_mapped_only" — only residues with UniProt mapping
        intersection: "uniprot_positions" — filter to shared UniProt positions
        superposition: "apply_reference" — apply stored rotation/translation
        tags_tails: "exclude" — remove N/C-terminal tags
        mutations: "retain" — keep engineered mutation sites
    """

    name = ProtocolName.FAMILY_COMPARE
    version = 1
    parameters = {
        "residues": "sifts_mapped_only",
        "intersection": "uniprot_positions",
        "superposition": "apply_reference",
        "tags_tails": "exclude",
        "mutations": "retain",
    }

    def apply(
        self,
        parsed: ParsedStructure,
        scope: ComputationScope,
        *,
        sifts_mapped_residue_ids: set[str] | None = None,
        tag_residue_ids: set[str] | None = None,
    ) -> NormalizedView:
        """Apply family_compare filtering.

        Args:
            parsed: Raw parsed structure.
            scope: Computation scope.
            sifts_mapped_residue_ids: Set of residue auth_seq_ids that have
                SIFTS UniProt mapping. If None, all resolved residues pass.
            tag_residue_ids: Set of residue auth_seq_ids identified as
                tags/tails (N/C-terminal affinity tags, etc.). These are excluded.
        """
        included_chains = set(scope.primary_chain_ids)
        excluded_chains = set(scope.exclude_chain_ids)
        mapped_ids = sifts_mapped_residue_ids or set()
        tag_ids = tag_residue_ids or set()

        result_chains: dict[str, list[NormalizedResidue]] = {}

        for chain in parsed.chains:
            if chain.auth_asym_id in excluded_chains:
                continue
            if included_chains and chain.auth_asym_id not in included_chains:
                continue
            if chain.entity_type.lower() not in _PROTEIN_ENTITY_TYPES:
                continue

            normalized = self._filter_chain_residues(chain, mapped_ids, tag_ids)
            if normalized:
                result_chains[chain.auth_asym_id] = normalized

        return NormalizedView(
            structure_id=parsed.pdb_id,
            protocol_name=self.name.value,
            protocol_version=self.version,
            chains=result_chains,
            parameters=self.parameters.copy(),
            metadata={
                "sifts_mapped_count": sum(len(v) for v in result_chains.values()),
                "tags_excluded": len(tag_ids),
            },
        )

    def _filter_chain_residues(
        self,
        chain: ParsedChain,
        mapped_ids: set[str],
        tag_ids: set[str],
    ) -> list[NormalizedResidue]:
        """Filter residues for family comparison."""
        result: list[NormalizedResidue] = []

        for residue in chain.residues:
            # Exclude unresolved
            if not residue.is_resolved:
                continue

            # Exclude tags/tails
            residue_key = f"{chain.auth_asym_id}:{residue.auth_seq_id}"
            if residue_key in tag_ids:
                continue

            # SIFTS-mapped only (if mapping data provided)
            if mapped_ids and residue_key not in mapped_ids:
                continue

            # Mutations retained (no filtering on is_modified)

            # Select best altloc
            filtered_atoms = _select_highest_occupancy_altloc(residue.atoms)

            # Keep all protein atoms (not just Cα) for displacement vectors
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
_instance = FamilyCompareProtocol()
register_protocol(_instance)
