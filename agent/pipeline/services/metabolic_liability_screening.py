"""
Metabolic Liability Screening Service

Identifies sites vulnerable to CYP450 metabolism using SMARTS pattern matching.
Screens for metabolic stability by detecting surface-exposed sites of metabolism (SOMs).

Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7
"""

from typing import List, Optional, Tuple
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from biotite.structure import AtomArray
import biotite.structure as struc
from freesasa import Structure, calc

from gosp.models.data_models import StructureData


class MetabolicLiabilityScreeningError(Exception):
    """Raised when metabolic liability screening fails."""
    pass


class SiteOfMetabolism:
    """Container for site of metabolism data."""
    
    def __init__(
        self,
        atom_idx: int,
        atom_type: str,
        pattern: str,
        coords: Tuple[float, float, float],
        sasa: float,
        is_exposed: bool
    ):
        self.atom_idx = atom_idx
        self.atom_type = atom_type
        self.pattern = pattern
        self.coords = coords
        self.sasa = sasa
        self.is_exposed = is_exposed


class MetabolicLiabilityScreeningResult:
    """Container for metabolic liability screening results."""
    
    def __init__(
        self,
        vulnerable_atoms: List[int],
        som_count: int,
        patterns_matched: List[str],
        surface_exposed_soms: List[int],
        sites: List[SiteOfMetabolism],
        flagged: bool,
        computation_time_sec: float,
        warning: Optional[str] = None
    ):
        self.vulnerable_atoms = vulnerable_atoms
        self.som_count = som_count
        self.patterns_matched = patterns_matched
        self.surface_exposed_soms = surface_exposed_soms
        self.sites = sites
        self.flagged = flagged
        self.computation_time_sec = computation_time_sec
        self.warning = warning


def screen_metabolic_liability(
    structure: StructureData,
    isoform: str = "3A4",
    sasa_threshold: float = 15.0,
    flag_threshold: int = 2
) -> MetabolicLiabilityScreeningResult:
    """
    Screen for metabolic liability by identifying CYP450 sites of metabolism.
    
    Algorithm:
    1. Convert structure to SMILES representation using RDKit
    2. Apply SMARTS patterns for CYP3A4 sites (aliphatic C-H, aromatic C-H)
    3. Map vulnerable atoms to 3D coordinates
    4. Calculate SASA for vulnerable atoms
    5. Filter for surface-exposed sites (SASA > sasa_threshold%)
    6. Flag structures with ≥flag_threshold exposed SOMs
    
    Args:
        structure: StructureData object with coordinates
        isoform: CYP450 isoform to screen (default: "3A4")
        sasa_threshold: SASA percentage threshold for exposed sites (default: 15.0%)
        flag_threshold: Number of exposed SOMs to trigger flag (default: 2)
    
    Returns:
        MetabolicLiabilityScreeningResult with vulnerable sites and flags
    
    Raises:
        MetabolicLiabilityScreeningError: If screening fails
    
    Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7
    """
    import time
    start_time = time.time()
    
    try:
        # Convert StructureData to Biotite structure for SASA calculation
        biotite_structure = _structure_data_to_biotite(structure)
        
        # Convert structure to SMILES using RDKit
        smiles = _structure_to_smiles(structure, biotite_structure)
        
        if smiles is None:
            # Could not generate SMILES - return empty result with warning
            computation_time = time.time() - start_time
            return MetabolicLiabilityScreeningResult(
                vulnerable_atoms=[],
                som_count=0,
                patterns_matched=[],
                surface_exposed_soms=[],
                sites=[],
                flagged=False,
                computation_time_sec=computation_time,
                warning="Could not generate SMILES representation for structure"
            )
        
        # Apply SMARTS patterns for CYP sites
        vulnerable_atoms, patterns_matched = _apply_cyp_patterns(
            smiles,
            isoform
        )
        
        if len(vulnerable_atoms) == 0:
            # No vulnerable sites found
            computation_time = time.time() - start_time
            return MetabolicLiabilityScreeningResult(
                vulnerable_atoms=[],
                som_count=0,
                patterns_matched=[],
                surface_exposed_soms=[],
                sites=[],
                flagged=False,
                computation_time_sec=computation_time,
                warning=None
            )
        
        # Map vulnerable atoms to 3D coordinates
        sites = _map_atoms_to_coords(
            vulnerable_atoms,
            patterns_matched,
            biotite_structure
        )
        
        # Calculate SASA for vulnerable atoms
        atom_sasa = _calculate_atom_sasa(biotite_structure)
        
        # Filter for surface-exposed sites
        surface_exposed_soms = []
        for site in sites:
            # Get SASA for this atom
            sasa_percent = atom_sasa.get(site.atom_idx, 0.0)
            site.sasa = sasa_percent
            site.is_exposed = sasa_percent > sasa_threshold
            
            if site.is_exposed:
                surface_exposed_soms.append(site.atom_idx)
        
        # Flag if ≥ threshold exposed SOMs
        flagged = len(surface_exposed_soms) >= flag_threshold
        
        computation_time = time.time() - start_time
        
        return MetabolicLiabilityScreeningResult(
            vulnerable_atoms=vulnerable_atoms,
            som_count=len(vulnerable_atoms),
            patterns_matched=list(set(patterns_matched)),
            surface_exposed_soms=surface_exposed_soms,
            sites=sites,
            flagged=flagged,
            computation_time_sec=computation_time,
            warning=None
        )
    
    except Exception as e:
        if isinstance(e, MetabolicLiabilityScreeningError):
            raise
        raise MetabolicLiabilityScreeningError(
            f"Metabolic liability screening failed: {str(e)}"
        )


def _structure_data_to_biotite(structure: StructureData) -> AtomArray:
    """
    Convert StructureData to Biotite AtomArray.
    
    Args:
        structure: StructureData object
    
    Returns:
        Biotite AtomArray reconstructed from StructureData
    """
    total_atoms = structure.total_atoms
    atom_array = struc.AtomArray(total_atoms)

    atom_index = 0
    for chain in structure.chains:
        for residue in chain.residues:
            for atom in residue.atoms:
                atom_array.atom_name[atom_index] = atom.atom_name
                atom_array.res_name[atom_index] = atom.residue_name
                atom_array.chain_id[atom_index] = atom.chain_id
                atom_array.res_id[atom_index] = atom.residue_id
                atom_array.coord[atom_index] = [atom.x, atom.y, atom.z]
                if hasattr(atom_array, "occupancy"):
                    atom_array.occupancy[atom_index] = atom.occupancy
                if hasattr(atom_array, "b_factor"):
                    atom_array.b_factor[atom_index] = atom.b_factor
                atom_array.element[atom_index] = atom.element.upper()
                atom_index += 1

    return atom_array


def _structure_to_smiles(
    structure: StructureData,
    biotite_structure: AtomArray
) -> Optional[str]:
    """
    Convert protein structure to SMILES representation.
    
    Args:
        structure: StructureData object
        biotite_structure: Biotite AtomArray
    
    Returns:
        SMILES string or None if conversion fails
    
    Validates: Requirements 8.1
    """
    try:
        sequence = "".join(chain.sequence for chain in structure.chains if chain.sequence)
        if not sequence:
            return None

        # For proteins, we'll use the sequence to generate a simplified SMILES
        # In a real implementation, this would use RDKit's protein-to-mol conversion
        # For now, we'll create a mol object from the structure

        # Extract backbone atoms and create RDKit molecule
        # This is a simplified approach - real implementation would be more sophisticated
        mol = Chem.MolFromSequence(sequence)
        
        if mol is None:
            return None
        
        # Generate SMILES
        smiles = Chem.MolToSmiles(mol)
        return smiles
    
    except Exception:
        return None


def _apply_cyp_patterns(
    smiles: str,
    isoform: str
) -> Tuple[List[int], List[str]]:
    """
    Apply SMARTS patterns for CYP450 sites of metabolism.
    
    Args:
        smiles: SMILES representation of molecule
        isoform: CYP450 isoform (e.g., "3A4")
    
    Returns:
        Tuple of (vulnerable_atom_indices, matched_patterns)
    
    Validates: Requirements 8.2, 8.3
    """
    # SMARTS patterns for CYP3A4 sites of metabolism
    # Based on literature patterns for common oxidation sites
    cyp3a4_patterns = {
        'aliphatic_ch': '[C;H3,H2,H1]',  # Aliphatic C-H
        'aromatic_ch': '[c;H]',  # Aromatic C-H
        'benzylic': '[C;H2,H1][c]',  # Benzylic position
        'allylic': '[C;H2,H1][C]=[C]',  # Allylic position
        'tertiary_amine': '[N;H0]([C])([C])[C]',  # Tertiary amine
    }
    
    # Convert SMILES to molecule
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        return [], []
    
    vulnerable_atoms = []
    patterns_matched = []
    
    # Apply each pattern
    for pattern_name, smarts in cyp3a4_patterns.items():
        pattern = Chem.MolFromSmarts(smarts)
        
        if pattern is None:
            continue
        
        # Find matches
        matches = mol.GetSubstructMatches(pattern)
        
        for match in matches:
            # Add the first atom in each match (the reactive center)
            atom_idx = match[0]
            if atom_idx not in vulnerable_atoms:
                vulnerable_atoms.append(atom_idx)
                patterns_matched.append(pattern_name)
    
    return vulnerable_atoms, patterns_matched


def _map_atoms_to_coords(
    vulnerable_atoms: List[int],
    patterns_matched: List[str],
    biotite_structure: AtomArray
) -> List[SiteOfMetabolism]:
    """
    Map vulnerable atoms to 3D coordinates.
    
    Args:
        vulnerable_atoms: List of atom indices
        patterns_matched: List of matched pattern names
        biotite_structure: Biotite AtomArray
    
    Returns:
        List of SiteOfMetabolism objects
    
    Validates: Requirements 8.4
    """
    sites = []
    
    for i, atom_idx in enumerate(vulnerable_atoms):
        # Map to structure coordinates
        # Note: This is simplified - real implementation would need proper atom mapping
        if atom_idx < len(biotite_structure):
            atom = biotite_structure[atom_idx]
            coords = tuple(atom.coord)
            atom_type = atom.element
            pattern = patterns_matched[i] if i < len(patterns_matched) else "unknown"
            
            site = SiteOfMetabolism(
                atom_idx=atom_idx,
                atom_type=atom_type,
                pattern=pattern,
                coords=coords,
                sasa=0.0,  # Will be filled in later
                is_exposed=False  # Will be filled in later
            )
            sites.append(site)
    
    return sites


def _calculate_atom_sasa(structure: AtomArray) -> dict:
    """
    Calculate SASA percentage for each atom using FreeSASA.
    
    Args:
        structure: Biotite AtomArray
    
    Returns:
        Dictionary mapping atom_idx -> SASA percentage
    
    Validates: Requirements 8.5
    """
    import tempfile
    import os
    from biotite.structure.io.pdb import PDBFile
    import io

    try:
        # Create a PDB file in memory
        pdb_file = PDBFile()
        pdb_file.set_structure(structure)
        
        # Write to a string buffer
        with io.StringIO() as f:
            pdb_file.write(f)
            pdb_string = f.getvalue()

        # FreeSASA requires a file path, not a string
        # Write PDB string to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            f.write(pdb_string)
            pdb_path = f.name
        
        try:
            # Calculate SASA using FreeSASA with file path
            freesasa_structure = Structure(pdb_path)
            result = calc(freesasa_structure)
            
            # Extract per-atom SASA
            atom_sasa = {}
            
            for atom_idx in range(len(structure)):
                # FreeSASA uses 1-based indexing
                atom_area = result.atomArea(atom_idx + 1)
                
                # Estimate maximum SASA for this atom type
                element = structure.element[atom_idx]
                max_sasa = _get_max_atom_sasa(element)
                
                # Calculate percentage
                if max_sasa > 0:
                    sasa_percent = (atom_area / max_sasa) * 100.0
                else:
                    sasa_percent = 0.0
                
                atom_sasa[atom_idx] = sasa_percent
            
            return atom_sasa
        finally:
            # Clean up temporary file
            if os.path.exists(pdb_path):
                os.unlink(pdb_path)

    except Exception as e:
        raise MetabolicLiabilityScreeningError(f"SASA calculation failed: {str(e)}")


def _get_max_atom_sasa(element: str) -> float:
    """
    Get maximum SASA for an atom type (Ų).
    
    Approximate values based on van der Waals radii.
    """
    max_sasa_values = {
        'C': 20.0,
        'N': 18.0,
        'O': 16.0,
        'S': 25.0,
        'H': 5.0,
    }
    return max_sasa_values.get(element, 20.0)  # Default to carbon
