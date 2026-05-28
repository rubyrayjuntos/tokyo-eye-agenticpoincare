"""
Immunogenicity Screening Service

Identifies immunogenic epitopes on solvent-exposed surfaces using NetMHCIIpan 4.1 API.
Screens for anti-drug antibody (ADA) risk by predicting MHC-II binding affinities.

Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7
"""

from typing import List, Optional, Tuple
import numpy as np
import requests
from biotite.structure import AtomArray
import biotite.structure as struc
from freesasa import Structure, calc

from gosp.models.data_models import StructureData, RedZoneFlag, RedZoneFlagDetails


class ImmunogenicityScreeningError(Exception):
    """Raised when immunogenicity screening fails."""
    pass


class Epitope:
    """Container for epitope data."""
    
    def __init__(
        self,
        sequence: str,
        start_res: int,
        end_res: int,
        allele: str,
        ic50: float,
        sasa_percent: float,
        flagged: bool
    ):
        self.sequence = sequence
        self.start_res = start_res
        self.end_res = end_res
        self.allele = allele
        self.ic50 = ic50
        self.sasa_percent = sasa_percent
        self.flagged = flagged


class ImmunogenicityScreeningResult:
    """Container for immunogenicity screening results."""
    
    def __init__(
        self,
        epitopes: List[Epitope],
        total_peptides: int,
        flagged_count: int,
        computation_time_sec: float,
        warning: Optional[str] = None
    ):
        self.epitopes = epitopes
        self.total_peptides = total_peptides
        self.flagged_count = flagged_count
        self.computation_time_sec = computation_time_sec
        self.warning = warning


def screen_immunogenicity(
    structure: StructureData,
    alleles: List[str] = None,
    ic50_threshold: float = 500.0,
    sasa_threshold: float = 20.0,
    peptide_length: int = 15,
    netmhciipan_url: Optional[str] = None
) -> ImmunogenicityScreeningResult:
    """
    Screen for immunogenic epitopes on solvent-exposed surfaces.
    
    Algorithm:
    1. Calculate SASA for all residues using FreeSASA
    2. Identify solvent-accessible residues (SASA > sasa_threshold%)
    3. Extract 15-mer peptide sequences from exposed regions
    4. Submit peptides to NetMHCIIpan 4.1 for HLA binding prediction
    5. Flag peptides with IC50 < ic50_threshold nM as Red Zone violations
    
    Args:
        structure: StructureData object with coordinates
        alleles: List of HLA alleles to test (default: DRB1*01:01, DRB1*15:01, DRB1*03:01)
        ic50_threshold: IC50 threshold for flagging (default: 500.0 nM)
        sasa_threshold: SASA percentage threshold for exposed residues (default: 20.0%)
        peptide_length: Length of peptide sequences to extract (default: 15)
        netmhciipan_url: Optional URL for NetMHCIIpan API (for testing)
    
    Returns:
        ImmunogenicityScreeningResult with epitopes and flags
    
    Raises:
        ImmunogenicityScreeningError: If screening fails
    
    Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7
    """
    import time
    start_time = time.time()
    
    if alleles is None:
        alleles = ["DRB1*01:01", "DRB1*15:01", "DRB1*03:01"]
    
    try:
        # Convert StructureData to Biotite structure
        biotite_structure = _build_biotite_structure_from_data(structure)
        
        # Calculate SASA for all residues
        residue_sasa = _calculate_residue_sasa(biotite_structure)
        
        # Identify solvent-accessible residues (SASA > threshold%)
        exposed_residues = _identify_exposed_residues(
            residue_sasa, 
            sasa_threshold
        )
        
        if len(exposed_residues) == 0:
            # No exposed residues - return empty result
            computation_time = time.time() - start_time
            return ImmunogenicityScreeningResult(
                epitopes=[],
                total_peptides=0,
                flagged_count=0,
                computation_time_sec=computation_time,
                warning="No exposed residues found (SASA > {}%)".format(sasa_threshold)
            )
        
        # Extract 15-mer peptide sequences from exposed regions
        full_sequence = "".join([c.sequence for c in structure.chains])
        peptides = _extract_peptides(
            full_sequence,
            exposed_residues,
            peptide_length
        )
        
        if len(peptides) == 0:
            # No peptides extracted - return empty result
            computation_time = time.time() - start_time
            return ImmunogenicityScreeningResult(
                epitopes=[],
                total_peptides=0,
                flagged_count=0,
                computation_time_sec=computation_time,
                warning="No peptides could be extracted from exposed regions"
            )
        
        # Submit peptides to NetMHCIIpan for HLA binding prediction
        epitopes = []
        warning = None
        
        try:
            predictions = _predict_mhc_binding(
                peptides,
                alleles,
                netmhciipan_url
            )
            
            # Process predictions and create epitopes
            for peptide_seq, start_res, end_res in peptides:
                # Get SASA percentage for this region
                region_sasa = np.mean([
                    residue_sasa.get(res_id, 0.0) 
                    for res_id in range(start_res, end_res + 1)
                ])
                
                # Check predictions for each allele
                for allele in alleles:
                    ic50 = predictions.get((peptide_seq, allele), None)
                    
                    if ic50 is not None:
                        # Flag if IC50 < threshold
                        flagged = ic50 < ic50_threshold
                        
                        epitope = Epitope(
                            sequence=peptide_seq,
                            start_res=start_res,
                            end_res=end_res,
                            allele=allele,
                            ic50=ic50,
                            sasa_percent=region_sasa,
                            flagged=flagged
                        )
                        epitopes.append(epitope)
        
        except Exception as e:
            # Graceful degradation if NetMHCIIpan is unavailable
            warning = f"NetMHCIIpan unavailable - immunogenicity screening skipped: {str(e)}"
            epitopes = []
        
        # Calculate statistics
        total_peptides = len(peptides)
        flagged_count = sum(1 for e in epitopes if e.flagged)
        
        computation_time = time.time() - start_time
        
        return ImmunogenicityScreeningResult(
            epitopes=epitopes,
            total_peptides=total_peptides,
            flagged_count=flagged_count,
            computation_time_sec=computation_time,
            warning=warning
        )
    
    except Exception as e:
        if isinstance(e, ImmunogenicityScreeningError):
            raise
        raise ImmunogenicityScreeningError(f"Immunogenicity screening failed: {str(e)}")


def _build_biotite_structure_from_data(structure_data: StructureData) -> AtomArray:
    """
    Constructs a Biotite AtomArray from the detailed StructureData model.
    """
    total_atoms = structure_data.total_atoms
    atom_array = struc.AtomArray(total_atoms)

    atom_index = 0
    for chain in structure_data.chains:
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


def _calculate_residue_sasa(structure: AtomArray) -> dict:
    """
    Calculate SASA percentage for each residue using FreeSASA.
    
    Args:
        structure: Biotite AtomArray
    
    Returns:
        Dictionary mapping residue_id -> SASA percentage
    
    Validates: Requirements 7.1
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
            
            # Extract per-residue SASA
            residue_sasa = {}
            
            # Get unique residues
            unique_residues = np.unique(structure.res_id)
            
            for res_id in unique_residues:
                # Get atoms for this residue
                res_mask = structure.res_id == res_id
                res_atoms = structure[res_mask]
                
                # Calculate total SASA for this residue
                total_sasa = 0.0
                max_sasa = 0.0
                
                for i, atom_idx in enumerate(np.where(res_mask)[0]):
                    # FreeSASA uses 1-based indexing
                    atom_sasa = result.atomArea(int(atom_idx) + 1)
                    total_sasa += atom_sasa
                
                # Estimate maximum SASA based on residue type
                # Using standard values from literature
                res_name = res_atoms.res_name[0]
                max_sasa = _get_max_sasa(res_name)
                
                # Calculate percentage
                if max_sasa > 0:
                    sasa_percent = (total_sasa / max_sasa) * 100.0
                else:
                    sasa_percent = 0.0
                
                residue_sasa[int(res_id)] = sasa_percent
            
            return residue_sasa
        finally:
            # Clean up temporary file
            if os.path.exists(pdb_path):
                os.unlink(pdb_path)
    
    except Exception as e:
        raise ImmunogenicityScreeningError(f"SASA calculation failed: {str(e)}")


def _get_max_sasa(res_name: str) -> float:
    """
    Get maximum SASA for a residue type (Ų).
    
    Values from Tien et al. (2013) PLoS ONE 8(11): e80635
    """
    max_sasa_values = {
        'ALA': 121.0, 'ARG': 265.0, 'ASN': 187.0, 'ASP': 187.0,
        'CYS': 148.0, 'GLN': 214.0, 'GLU': 214.0, 'GLY': 97.0,
        'HIS': 216.0, 'ILE': 195.0, 'LEU': 191.0, 'LYS': 230.0,
        'MET': 203.0, 'PHE': 228.0, 'PRO': 154.0, 'SER': 143.0,
        'THR': 163.0, 'TRP': 264.0, 'TYR': 255.0, 'VAL': 165.0
    }
    return max_sasa_values.get(res_name, 200.0)  # Default to average


def _identify_exposed_residues(
    residue_sasa: dict,
    threshold: float
) -> List[int]:
    """
    Identify residues with SASA > threshold%.
    
    Args:
        residue_sasa: Dictionary mapping residue_id -> SASA percentage
        threshold: SASA percentage threshold
    
    Returns:
        List of exposed residue IDs
    
    Validates: Requirements 7.1
    """
    exposed = []
    for res_id, sasa_percent in residue_sasa.items():
        if sasa_percent > threshold:
            exposed.append(res_id)
    
    return sorted(exposed)


def _extract_peptides(
    sequence: str,
    exposed_residues: List[int],
    peptide_length: int
) -> List[Tuple[str, int, int]]:
    """
    Extract peptide sequences from exposed regions.
    
    Args:
        sequence: Full amino acid sequence
        exposed_residues: List of exposed residue IDs (1-indexed)
        peptide_length: Length of peptides to extract
    
    Returns:
        List of (peptide_sequence, start_res, end_res) tuples
    
    Validates: Requirements 7.2
    """
    peptides = []
    seq_length = len(sequence)
    
    # Extract all possible peptides of specified length
    for i in range(seq_length - peptide_length + 1):
        start_res = i + 1  # 1-indexed
        end_res = start_res + peptide_length - 1
        
        # Check if this peptide overlaps with exposed residues
        peptide_residues = set(range(start_res, end_res + 1))
        exposed_set = set(exposed_residues)
        
        # Include peptide if at least 50% of residues are exposed
        overlap = len(peptide_residues & exposed_set)
        if overlap >= peptide_length * 0.5:
            peptide_seq = sequence[i:i + peptide_length]
            peptides.append((peptide_seq, start_res, end_res))
    
    return peptides


def _predict_mhc_binding(
    peptides: List[Tuple[str, int, int]],
    alleles: List[str],
    netmhciipan_url: Optional[str] = None
) -> dict:
    """
    Predict MHC-II binding affinities using NetMHCIIpan 4.1 API.
    
    Args:
        peptides: List of (peptide_sequence, start_res, end_res) tuples
        alleles: List of HLA alleles to test
        netmhciipan_url: Optional URL for NetMHCIIpan API
    
    Returns:
        Dictionary mapping (peptide_seq, allele) -> IC50 (nM)
    
    Raises:
        Exception: If API call fails
    
    Validates: Requirements 7.3, 7.4
    """
    if netmhciipan_url is None:
        # Default NetMHCIIpan 4.1 API endpoint
        # Note: This is a placeholder - actual API endpoint would be configured
        netmhciipan_url = "http://tools.iedb.org/mhcii/api/predict/"
    
    predictions = {}
    
    # Prepare request data
    peptide_sequences = [p[0] for p in peptides]
    
    for allele in alleles:
        try:
            # Make API request
            # Note: This is a simplified implementation
            # Real implementation would handle batching, rate limiting, etc.
            response = requests.post(
                netmhciipan_url,
                data={
                    'method': 'netmhciipan',
                    'sequence_text': '\n'.join(peptide_sequences),
                    'allele': allele,
                    'length': '15'
                },
                timeout=60
            )
            
            if response.status_code == 200:
                # Parse response
                # Note: Actual parsing would depend on API response format
                results = _parse_netmhciipan_response(response.text)
                
                for peptide_seq, ic50 in results.items():
                    predictions[(peptide_seq, allele)] = ic50
            else:
                raise Exception(f"API returned status code {response.status_code}")
        
        except requests.exceptions.RequestException as e:
            raise Exception(f"NetMHCIIpan API request failed: {str(e)}")
    
    return predictions


def _parse_netmhciipan_response(response_text: str) -> dict:
    """
    Parse NetMHCIIpan API response.
    
    Args:
        response_text: Raw API response text
    
    Returns:
        Dictionary mapping peptide_sequence -> IC50 (nM)
    """
    # This is a simplified parser
    # Real implementation would parse the actual NetMHCIIpan output format
    results = {}
    
    lines = response_text.strip().split('\n')
    for line in lines:
        if line.startswith('#') or not line.strip():
            continue
        
        parts = line.split()
        if len(parts) >= 3:
            peptide = parts[0]
            try:
                ic50 = float(parts[2])
                results[peptide] = ic50
            except (ValueError, IndexError):
                continue
    
    return results
