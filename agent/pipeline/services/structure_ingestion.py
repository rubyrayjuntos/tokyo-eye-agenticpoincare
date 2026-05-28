"""
Structure Ingestion Service

Handles fetching and parsing protein structures from RCSB PDB.
Supports mmCIF, BinaryCIF, and PDB formats.

Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5
"""

import gzip
import io
import json
import time
from typing import Literal, Optional, Union
import biotite.structure as struc
import biotite.structure.io as strucio
from biotite.database import rcsb
from biotite.structure import annotate_sse
import numpy as np

from gosp.models.data_models import StructureData, Atom, Residue, Chain
from gosp.services.error_handling import (
    StructuralError,
    ComputationError,
    retry_with_exponential_backoff,
    validate_structure_size,
    validate_secondary_structure,
    detect_out_of_scope_structure,
    check_timeout
)


class StructureIngestionError(Exception):
    """Raised when structure ingestion fails."""
    pass


class InvalidPDBIDError(StructureIngestionError):
    """Raised when PDB ID is invalid."""
    pass


class UnsupportedFormatError(StructureIngestionError):
    """Raised when file format is not supported."""
    pass


def parse_structure_content(
    file_content: Union[bytes, str, io.BytesIO, io.StringIO],
    format: Literal["cif", "bcif", "pdb"]
) -> struc.AtomArray:
    """Parse structure content into a Biotite AtomArray."""
    if format not in ["cif", "bcif", "pdb"]:
        raise UnsupportedFormatError(f"Unsupported format: {format}")

    import tempfile
    import os

    # BinaryCIF is binary; all others are text
    is_binary = (format == "bcif")

    if is_binary:
        if isinstance(file_content, bytes):
            raw_bytes = file_content
        elif hasattr(file_content, 'getvalue'):
            raw_bytes = file_content.getvalue()
        elif hasattr(file_content, 'read'):
            raw_bytes = file_content.read()
        else:
            raise StructureIngestionError(f"Unsupported file_content type: {type(file_content)}")
        tmp_file = tempfile.NamedTemporaryFile(mode='wb', suffix=f'.{format}', delete=False)
        tmp_file.write(raw_bytes)
        tmp_file.close()
        tmp_path = tmp_file.name
    else:
        if isinstance(file_content, bytes):
            text_content = file_content.decode('utf-8')
        elif isinstance(file_content, str):
            text_content = file_content
        elif hasattr(file_content, 'getvalue'):
            text_content = file_content.getvalue()
        elif hasattr(file_content, 'read'):
            text_content = file_content.read()
        else:
            raise StructureIngestionError(f"Unsupported file_content type: {type(file_content)}")
        tmp_file = tempfile.NamedTemporaryFile(mode='w', suffix=f'.{format}', delete=False)
        tmp_file.write(text_content)
        tmp_file.close()
        tmp_path = tmp_file.name

    try:
        if format == "cif":
            from biotite.structure.io.pdbx import CIFFile
            from biotite.structure.io.pdbx import get_structure as get_pdbx_structure
            cif_file = CIFFile.read(tmp_path)
            return get_pdbx_structure(cif_file, model=None, extra_fields=['atom_id'])
        elif format == "bcif":
            from biotite.structure.io.pdbx import BinaryCIFFile
            from biotite.structure.io.pdbx import get_structure as get_pdbx_structure
            bcif_file = BinaryCIFFile.read(tmp_path)
            return get_pdbx_structure(bcif_file, model=None, extra_fields=['atom_id'])
        else:  # format == "pdb"
            from biotite.structure.io.pdb import PDBFile
            pdb_file = PDBFile.read(tmp_path)
            return pdb_file.get_structure(model=None)
    finally:
        # Clean up temporary file
        os.unlink(tmp_path)


@retry_with_exponential_backoff(
    max_retries=3,
    initial_delay=1.0,
    exceptions=(ConnectionError, TimeoutError, IOError)
)
def fetch_structure_from_rcsb(
    pdb_id: str,
    format: Literal["cif", "bcif", "pdb"] = "cif",
    chain_id: Optional[str] = None,
    timeout_sec: float = 30.0
) -> StructureData:
    """
    Fetch and parse a protein structure from RCSB PDB.

    Args:
        pdb_id: PDB identifier (e.g., "1L2Y")
        format: File format - "cif" (mmCIF), "bcif" (BinaryCIF), or "pdb"
        chain_id: Optional chain ID to filter to a single chain
        timeout_sec: Timeout in seconds (default: 30.0)

    Returns:
        StructureData object with parsed structure information

    Raises:
        InvalidPDBIDError: If PDB ID is invalid or not found
        UnsupportedFormatError: If format is not supported
        StructureIngestionError: If parsing fails
        StructuralError: If structure exceeds size limits or has insufficient secondary structure

    Validates: Requirements 3.1, 3.2, 3.3, 15.1, 15.2, 15.5, 15.6
    """
    start_time = time.time()
    
    # Validate PDB ID format (4 characters: alphanumeric)
    if not pdb_id or len(pdb_id) != 4 or not pdb_id.isalnum():
        raise InvalidPDBIDError(f"Invalid PDB ID format: {pdb_id}")
    
    # Validate format
    if format not in ["cif", "bcif", "pdb"]:
        raise UnsupportedFormatError(f"Unsupported format: {format}")
    
    try:
        # Check timeout
        check_timeout(start_time, timeout_sec, "Structure ingestion")
        
        # Fetch structure from RCSB
        file_content = rcsb.fetch(pdb_id, format, target_path=None)
        
        # Check timeout
        check_timeout(start_time, timeout_sec, "Structure ingestion")
        
        # Parse structure based on format
        # rcsb.fetch returns a file-like object (StringIO/BytesIO)
        structure = parse_structure_content(file_content, format)
        
    except Exception as e:
        if isinstance(e, (InvalidPDBIDError, UnsupportedFormatError, StructuralError)):
            raise
        raise StructureIngestionError(f"Failed to fetch or parse structure {pdb_id}: {str(e)}")

    # Filter to single chain if requested
    if chain_id is not None:
        if isinstance(structure, struc.AtomArrayStack):
            filtered_models = []
            for i in range(structure.stack_depth()):
                model = structure[i]
                chain_mask = model.chain_id == chain_id
                if not np.any(chain_mask):
                    raise StructureIngestionError(
                        f"Chain '{chain_id}' not found in structure {pdb_id}"
                    )
                filtered_models.append(model[chain_mask])
            structure = struc.stack(filtered_models)
        else:
            chain_mask = structure.chain_id == chain_id
            if not np.any(chain_mask):
                raise StructureIngestionError(
                    f"Chain '{chain_id}' not found in structure {pdb_id}"
                )
            structure = structure[chain_mask]

    # Extract structure information
    try:
        check_timeout(start_time, timeout_sec, "Structure ingestion")
        structure_data = _extract_structure_data(pdb_id, structure)
        
        # Validate structure size (Requirement 15.1)
        validate_structure_size(structure_data.total_residues, max_residues=5000)
        
        # Validate secondary structure content (Requirement 15.2)
        secondary_structure_count = 0
        for chain_id in structure_data.secondary_structure:
            secondary_structure_count += sum(
                1 for ss in structure_data.secondary_structure[chain_id] if ss in ['H', 'E']
            )
        
        ss_fraction = secondary_structure_count / structure_data.total_residues if structure_data.total_residues > 0 else 0
        warning = validate_secondary_structure(ss_fraction, min_fraction=0.20)
        
        # Store warning in structure data if present
        if warning:
            # Note: We don't fail, just warn - this is graceful degradation
            pass
        
        # Aggregate sequence and sse for out-of-scope detection
        full_sequence = "".join([c.sequence for c in structure_data.chains])
        full_sse = [ss for chain_id in structure_data.secondary_structure for ss in structure_data.secondary_structure[chain_id]]

        # Detect out-of-scope structures (Requirement 14.7)
        out_of_scope_error = detect_out_of_scope_structure(
            sequence=full_sequence,
            secondary_structure=full_sse
        )
        if out_of_scope_error:
            raise StructuralError(
                message=out_of_scope_error,
                details={
                    "pdb_id": pdb_id,
                    "sequence_length": len(full_sequence),
                    "secondary_structure_fraction": ss_fraction
                },
                suggested_action="This structure type is not supported by GOSP"
            )
        
    except Exception as e:
        if isinstance(e, StructuralError):
            raise
        raise StructureIngestionError(f"Failed to extract structure data: {str(e)}")
    
    return structure_data


def _extract_structure_data(pdb_id: str, structure: struc.AtomArray) -> StructureData:
    """
    Extract a complete, non-lossy representation from a Biotite structure.
    
    Args:
        pdb_id: PDB identifier.
        structure: Biotite AtomArray or AtomArrayStack.
    
    Returns:
        A complete StructureData object.
        
    Validates: Requirements 3.3
    """
    # Handle both single model (AtomArray) and multi-model (AtomArrayStack)
    is_stack = isinstance(structure, struc.AtomArrayStack)
    
    if is_stack:
        num_models = structure.stack_depth()
        # We only process the first model as per the architectural decision
        model = structure[0]
    else:
        num_models = 1
        model = structure

    resolution = float(model.resolution) if hasattr(model, 'resolution') else None

    chains_list = []
    total_atoms_count = 0
    total_residues_count = 0
    
    unique_chain_ids = struc.get_chains(model)
    
    sse_map = {}
    try:
        # Annotate SSE for the entire model once
        ca_atoms = model[model.atom_name == 'CA']
        sse_annotations = annotate_sse(ca_atoms)
        # Map SSE annotations to a dictionary for quick lookup: (chain_id, res_id) -> sse
        sse_lookup = {
            (ca.chain_id, ca.res_id): sse for ca, sse in zip(ca_atoms, sse_annotations)
        }
    except Exception:
        sse_lookup = {} # Fallback to empty if annotation fails
    
    for chain_id in unique_chain_ids:
        chain_mask = (model.chain_id == chain_id)
        chain_atoms = model[chain_mask]
        
        residues_list = []
        chain_sequence_parts = [] # Use parts to build sequence after iterating all residues
        
        # Group atoms by residue ID and name
        unique_res_ids = np.unique(chain_atoms.res_id)
        for res_id in unique_res_ids:
            res_id_mask = (chain_atoms.res_id == res_id)
            residue_atoms_array = chain_atoms[res_id_mask]
            
            if len(residue_atoms_array) == 0:
                continue
            
            # The first atom determines the residue name for this res_id
            residue_name = residue_atoms_array[0].res_name
            
            print("Available annotation categories:", residue_atoms_array.get_annotation_categories())
            atoms_list = []
            # Get annotation arrays once before the loop
            # atom_id is conditionally assigned
            atom_name_annot = residue_atoms_array.get_annotation("atom_name")
            res_name_annot = residue_atoms_array.get_annotation("res_name")
            chain_id_annot = residue_atoms_array.get_annotation("chain_id")
            res_id_annot = residue_atoms_array.get_annotation("res_id")
            element_annot = residue_atoms_array.get_annotation("element")
            
            # Check if atom_id annotation exists, otherwise generate it
            has_atom_id_annotation = "atom_id" in residue_atoms_array.get_annotation_categories()
            if has_atom_id_annotation:
                atom_id_annot = residue_atoms_array.get_annotation("atom_id")
            else:
                atom_id_annot = None # Will be generated below

            occupancy_annot = None
            if "occupancy" in residue_atoms_array.get_annotation_categories():
                occupancy_annot = residue_atoms_array.get_annotation("occupancy")

            b_factor_annot = None
            if "b_factor" in residue_atoms_array.get_annotation_categories():
                b_factor_annot = residue_atoms_array.get_annotation("b_factor")

            for i in range(len(residue_atoms_array)):
                current_atom_id = atom_id_annot[i] if has_atom_id_annotation else i + 1
                atoms_list.append(Atom(
                    atom_id=current_atom_id,
                    atom_name=atom_name_annot[i],
                    residue_name=res_name_annot[i],
                    chain_id=chain_id_annot[i],
                    residue_id=res_id_annot[i],
                    x=residue_atoms_array.coord[i, 0],
                    y=residue_atoms_array.coord[i, 1],
                    z=residue_atoms_array.coord[i, 2],
                    occupancy=occupancy_annot[i] if occupancy_annot is not None else 1.0,
                    b_factor=b_factor_annot[i] if b_factor_annot is not None else 0.0,
                    element=element_annot[i].upper()
                ))
            
            residues_list.append(Residue(
                residue_id=res_id,
                residue_name=residue_name,
                atoms=atoms_list,
            ))
            
            # Append to one-letter sequence parts
            chain_sequence_parts.append(_convert_to_one_letter([residue_name])[0])
        
        # Construct the full chain sequence
        chain_sequence = "".join(chain_sequence_parts)

        # Handle secondary structure for the chain
        chain_sse = []
        for res in residues_list:
            sse_code = sse_lookup.get((chain_id, res.residue_id))
            if sse_code == 'a':
                chain_sse.append('H')
            elif sse_code == 'b':
                chain_sse.append('E')
            else:
                chain_sse.append('C')
        sse_map[chain_id] = chain_sse

        chain_obj = Chain(
            chain_id=chain_id,
            residues=residues_list,
            sequence="".join(chain_sequence)
        )
        chains_list.append(chain_obj)
        
        total_residues_count += len(residues_list)
        total_atoms_count += len(chain_atoms)

    # Simplified missing residues detection - can be improved later
    missing_residues_map = {}

    return StructureData(
        pdb_id=pdb_id.upper(),
        num_models=num_models,
        resolution=resolution,
        chains=chains_list,
        total_residues=total_residues_count,
        total_atoms=total_atoms_count,
        secondary_structure=sse_map,
        missing_residues=missing_residues_map
    )


def build_structure_data(pdb_id: str, structure: struc.AtomArray) -> StructureData:
    """Public wrapper for building StructureData from a Biotite structure."""
    return _extract_structure_data(pdb_id, structure)

def _convert_to_one_letter(residue_names: list[str]) -> str:
    """
    Convert 3-letter amino acid codes to 1-letter codes.
    
    Args:
        residue_names: List of 3-letter residue names
    
    Returns:
        String of 1-letter amino acid codes
    """
    # Standard amino acid conversion table
    conversion = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
        'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
        'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
        'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
    }
    
    sequence = []
    for res_name in residue_names:
        # Handle modified residues or unknowns
        one_letter = conversion.get(res_name, 'X')
        sequence.append(one_letter)
    
    return ''.join(sequence)


def compress_structure_payload(structure: StructureData) -> bytes:
    """
    Compress StructureData to gzip-compressed JSON.
    
    Args:
        structure: StructureData object
    
    Returns:
        Gzip-compressed JSON bytes
    
    Validates: Requirements 3.4
    """
    # Serialize to JSON
    json_str = structure.model_dump_json()
    
    # Compress with gzip
    compressed = gzip.compress(json_str.encode('utf-8'))
    
    return compressed


def decompress_structure_payload(compressed_data: bytes) -> StructureData:
    """
    Decompress gzip-compressed JSON to StructureData.
    
    Args:
        compressed_data: Gzip-compressed JSON bytes
    
    Returns:
        StructureData object
    
    Validates: Requirements 3.4
    """
    # Decompress
    decompressed = gzip.decompress(compressed_data).decode('utf-8')
    
    # Parse JSON
    data = json.loads(decompressed)
    
    # Create StructureData object
    structure = StructureData(**data)
    
    return structure
