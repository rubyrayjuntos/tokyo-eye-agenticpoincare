"""binding_site normalization protocol.

Produces a view suitable for binding site analysis:
- Preserve ligands and cofactors within distance cutoff
- Exclude waters unless they are bridging (between protein and ligand)
- Local pocket coordinate frame
- Protonation states untouched (no hydrogen manipulation)

Requirements: 5.4
"""

from __future__ import annotations

import logging
import math

from science.dtie.ingest.chain_scorer import ComputationScope
from science.dtie.ingest.parser import ParsedAtom, ParsedChain, ParsedResidue, ParsedStructure
from science.dtie.normalize.protocols.base import (
    NormalizedResidue,
    NormalizedView,
    NormalizationProtocol,
    ProtocolName,
    _select_highest_occupancy_altloc,
    register_protocol,
)

logger = logging.getLogger(__name__)

# Default distance cutoff for binding site residue inclusion (Angstroms)
DEFAULT_CUTOFF_ANGSTROM = 5.0

# Entity types considered protein
_PROTEIN_ENTITY_TYPES = frozenset({
    "polymer", "polypeptide(l)", "polypeptide(d)", "protein",
})


class BindingSiteProtocol(NormalizationProtocol):
    """Normalization for binding site analysis.

    Parameters:
        ligands: "within_cutoff" — include ligands within distance threshold
        cutoff_angstrom: 5.0 — distance threshold for pocket residues
        waters: "exclude_unless_bridging" — remove waters except bridging
        frame: "local_pocket" — use local pocket coordinate frame
        protonation: "untouched" — do not modify protonation states
    """

    name = ProtocolName.BINDING_SITE
    version = 1
    parameters = {
        "ligands": "within_cutoff",
        "cutoff_angstrom": DEFAULT_CUTOFF_ANGSTROM,
        "waters": "exclude_unless_bridging",
        "frame": "local_pocket",
        "protonation": "untouched",
    }

    def apply(
        self,
        parsed: ParsedStructure,
        scope: ComputationScope,
        *,
        ligand_atoms: list[ParsedAtom] | None = None,
        bridging_water_residues: set[int] | None = None,
    ) -> NormalizedView:
        """Apply binding_site filtering.

        Args:
            parsed: Raw parsed structure.
            scope: Computation scope.
            ligand_atoms: Ligand atom coordinates for distance calculation.
                If None, hetero atoms from the structure are used as ligand reference.
            bridging_water_residues: Set of auth_seq_ids for waters identified
                as bridging. These are retained.
        """
        cutoff = self.parameters.get("cutoff_angstrom", DEFAULT_CUTOFF_ANGSTROM)
        included_chains = set(scope.primary_chain_ids)
        excluded_chains = set(scope.exclude_chain_ids)
        bridging = bridging_water_residues or set()

        # Collect ligand atom positions for distance filtering
        ligand_coords = self._collect_ligand_coords(parsed, ligand_atoms)

        result_chains: dict[str, list[NormalizedResidue]] = {}

        for chain in parsed.chains:
            if chain.auth_asym_id in excluded_chains:
                continue
            if included_chains and chain.auth_asym_id not in included_chains:
                continue

            normalized = self._filter_chain_residues(
                chain, ligand_coords, cutoff, bridging
            )
            if normalized:
                result_chains[chain.auth_asym_id] = normalized

        return NormalizedView(
            structure_id=parsed.pdb_id,
            protocol_name=self.name.value,
            protocol_version=self.version,
            chains=result_chains,
            parameters=self.parameters.copy(),
            metadata={
                "cutoff_angstrom": cutoff,
                "ligand_atom_count": len(ligand_coords),
                "pocket_residue_count": sum(len(v) for v in result_chains.values()),
            },
        )

    def _collect_ligand_coords(
        self,
        parsed: ParsedStructure,
        explicit_ligand_atoms: list[ParsedAtom] | None,
    ) -> list[tuple[float, float, float]]:
        """Collect ligand atom coordinates for proximity filtering."""
        if explicit_ligand_atoms:
            return [(a.x, a.y, a.z) for a in explicit_ligand_atoms]

        # Use hetero atoms from all chains as ligand reference
        coords: list[tuple[float, float, float]] = []
        for chain in parsed.chains:
            for residue in chain.residues:
                for atom in residue.atoms:
                    if atom.is_hetero:
                        coords.append((atom.x, atom.y, atom.z))
        return coords

    def _filter_chain_residues(
        self,
        chain: ParsedChain,
        ligand_coords: list[tuple[float, float, float]],
        cutoff: float,
        bridging_waters: set[int],
    ) -> list[NormalizedResidue]:
        """Filter residues by proximity to ligand atoms."""
        result: list[NormalizedResidue] = []

        for residue in chain.residues:
            if not residue.is_resolved:
                continue

            # Check if residue is within cutoff of any ligand atom
            if ligand_coords and not self._residue_within_cutoff(residue, ligand_coords, cutoff):
                # Exception: bridging waters are kept
                if residue.auth_seq_id not in bridging_waters:
                    continue

            # Water handling: exclude unless bridging
            if residue.residue_name_3 == "HOH":
                if residue.auth_seq_id not in bridging_waters:
                    continue

            # Keep all atoms (protein + ligand); protonation untouched
            filtered_atoms = _select_highest_occupancy_altloc(residue.atoms)
            if not filtered_atoms:
                continue

            result.append(NormalizedResidue(
                residue_id=None,
                auth_seq_id=residue.auth_seq_id,
                label_seq_id=residue.label_seq_id,
                insertion_code=residue.insertion_code,
                residue_name=residue.residue_name,
                residue_name_3=residue.residue_name_3,
                comp_id=residue.comp_id,
                parent_comp_id=residue.parent_comp_id,
                is_modified=residue.is_modified,
                atoms=filtered_atoms,
            ))

        return result

    @staticmethod
    def _residue_within_cutoff(
        residue: ParsedResidue,
        ligand_coords: list[tuple[float, float, float]],
        cutoff: float,
    ) -> bool:
        """Check if any atom in the residue is within cutoff of any ligand atom."""
        cutoff_sq = cutoff * cutoff
        for atom in residue.atoms:
            for lx, ly, lz in ligand_coords:
                dx = atom.x - lx
                dy = atom.y - ly
                dz = atom.z - lz
                dist_sq = dx * dx + dy * dy + dz * dz
                if dist_sq <= cutoff_sq:
                    return True
        return False


# Register
_instance = BindingSiteProtocol()
register_protocol(_instance)
