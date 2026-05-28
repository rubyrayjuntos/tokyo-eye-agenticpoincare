"""
Dehydron Detection Service

Identifies under-wrapped backbone hydrogen bonds (dehydrons) in protein structures.
Uses spatial indexing (k-d tree) for efficient O(N log N) wrapping atom search.

Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9
"""

from typing import List, Tuple
import numpy as np
from scipy.spatial import cKDTree
import biotite.structure as struc

from gosp.models.data_models import Dehydron, StructureData
from gosp.services.biotite_utils import build_biotite_structure


class DehydronDetectionError(Exception):
    """Raised when dehydron detection fails."""
    pass


class DehydronDetectionResult:
    """Container for dehydron detection results."""
    
    def __init__(
        self,
        dehydrons: List[Dehydron],
        total_hbonds: int,
        dehydron_count: int,
        dehydron_fraction: float,
        computation_time_sec: float
    ):
        self.dehydrons = dehydrons
        self.total_hbonds = total_hbonds
        self.dehydron_count = dehydron_count
        self.dehydron_fraction = dehydron_fraction
        self.computation_time_sec = computation_time_sec


def detect_dehydrons(
    structure: StructureData,
    wrapping_threshold: int = 13,
    hbond_distance: float = 3.5,
    search_radius: float = 6.5,
    min_sequence_separation: int = 2
) -> DehydronDetectionResult:
    """
    Detect dehydrons (under-wrapped hydrogen bonds) in a protein structure.
    
    Algorithm:
    1. Filter backbone N (donor) and O (acceptor) atoms
    2. Build k-d tree from non-polar side-chain carbons
    3. For each potential H-bond (N-O distance ≤ hbond_distance):
       - Verify sequence separation > min_sequence_separation
       - Calculate midpoint
       - Query k-d tree for wrapping atoms within search_radius
       - Exclude donor/acceptor residues from wrapping count
       - Classify as dehydron if wrapping_count < wrapping_threshold
    
    Args:
        structure: StructureData object with coordinates
        wrapping_threshold: Threshold for dehydron classification (default: 13)
        hbond_distance: Maximum N-O distance for H-bond (default: 3.5Å)
        search_radius: Radius for wrapping atom search (default: 6.5Å)
        min_sequence_separation: Minimum residue separation (default: 2)
    
    Returns:
        DehydronDetectionResult with detected dehydrons and statistics
    
    Raises:
        DehydronDetectionError: If detection fails
    
    Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9
    """
    import time
    start_time = time.time()
    
    try:
        # Convert StructureData to Biotite structure for processing
        biotite_structure = build_biotite_structure(structure)
        
        # Filter backbone N (donor) and O (acceptor) atoms
        donors, acceptors = _extract_backbone_atoms(biotite_structure)
        
        if len(donors) == 0 or len(acceptors) == 0:
            # No backbone atoms found - return empty result
            computation_time = time.time() - start_time
            return DehydronDetectionResult(
                dehydrons=[],
                total_hbonds=0,
                dehydron_count=0,
                dehydron_fraction=0.0,
                computation_time_sec=computation_time
            )
        
        # Build k-d tree from non-polar side-chain carbons
        wrapping_atoms, wrapping_chain_ids, wrapping_residues = _extract_wrapping_atoms(biotite_structure)

        if len(wrapping_atoms) == 0:
            # No wrapping atoms - all bonds are dehydrons
            computation_time = time.time() - start_time
            hbonds = _find_hydrogen_bonds(
                donors, acceptors, hbond_distance, min_sequence_separation
            )
            dehydrons_list = [
                Dehydron(
                    donor_res_id=int(donor_res),
                    acceptor_res_id=int(acceptor_res),
                    wrapping_count=0,
                    midpoint=tuple(midpoint.tolist()),
                    distance=float(distance),
                    is_dehydron=True,
                    donor_chain_id=donor_chain,
                    acceptor_chain_id=acceptor_chain,
                    is_interchain=is_interchain,
                )
                for donor_chain, donor_res, acceptor_chain, acceptor_res, midpoint, distance, is_interchain in hbonds
            ]
            return DehydronDetectionResult(
                dehydrons=dehydrons_list,
                total_hbonds=len(hbonds),
                dehydron_count=len(hbonds),
                dehydron_fraction=1.0,
                computation_time_sec=computation_time
            )

        kdtree = cKDTree(wrapping_atoms)

        # Find potential hydrogen bonds
        hbonds = _find_hydrogen_bonds(
            donors, acceptors, hbond_distance, min_sequence_separation
        )

        # Calculate wrapping counts and classify dehydrons
        dehydrons_list = []
        for donor_chain, donor_res, acceptor_chain, acceptor_res, midpoint, distance, is_interchain in hbonds:
            # Query k-d tree for atoms within search_radius
            indices = kdtree.query_ball_point(midpoint, search_radius)

            # Count wrapping atoms excluding donor/acceptor residues (using chain+res_id)
            wrapping_count = 0
            for idx in indices:
                wrap_chain = wrapping_chain_ids[idx]
                wrap_res = wrapping_residues[idx]
                if (wrap_chain, wrap_res) != (donor_chain, donor_res) and \
                   (wrap_chain, wrap_res) != (acceptor_chain, acceptor_res):
                    wrapping_count += 1

            # Classify as dehydron if wrapping_count < threshold
            is_dehydron = wrapping_count < wrapping_threshold

            dehydron = Dehydron(
                donor_res_id=int(donor_res),
                acceptor_res_id=int(acceptor_res),
                wrapping_count=wrapping_count,
                midpoint=tuple(midpoint.tolist()),
                distance=float(distance),
                is_dehydron=is_dehydron,
                donor_chain_id=donor_chain,
                acceptor_chain_id=acceptor_chain,
                is_interchain=is_interchain,
            )
            dehydrons_list.append(dehydron)
        
        # Calculate statistics
        total_hbonds = len(dehydrons_list)
        dehydron_count = sum(1 for d in dehydrons_list if d.is_dehydron)
        dehydron_fraction = dehydron_count / total_hbonds if total_hbonds > 0 else 0.0
        
        computation_time = time.time() - start_time
        
        return DehydronDetectionResult(
            dehydrons=dehydrons_list,
            total_hbonds=total_hbonds,
            dehydron_count=dehydron_count,
            dehydron_fraction=dehydron_fraction,
            computation_time_sec=computation_time
        )
    
    except Exception as e:
        raise DehydronDetectionError(f"Dehydron detection failed: {str(e)}") from e


def _extract_backbone_atoms(
    structure: struc.AtomArray
) -> Tuple[List[Tuple[str, int, np.ndarray]], List[Tuple[str, int, np.ndarray]]]:
    """
    Extract backbone N (donor) and O (acceptor) atoms.

    Args:
        structure: Biotite AtomArray

    Returns:
        Tuple of (donors, acceptors) where each is a list of (chain_id, res_id, coord) tuples

    Validates: Requirements 4.1
    """
    donors = []
    acceptors = []

    # Filter backbone nitrogen atoms (donors)
    n_mask = (structure.atom_name == 'N') & (structure.element == 'N')
    for i in np.where(n_mask)[0]:
        donors.append((structure.chain_id[i], structure.res_id[i], structure.coord[i]))

    # Filter backbone oxygen atoms (acceptors)
    o_mask = (structure.atom_name == 'O') & (structure.element == 'O')
    for i in np.where(o_mask)[0]:
        acceptors.append((structure.chain_id[i], structure.res_id[i], structure.coord[i]))

    return donors, acceptors


def _extract_wrapping_atoms(
    structure: struc.AtomArray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract non-polar side-chain carbon atoms for wrapping calculation.

    Matches the GNN's is_apolar_carbon() filter: excludes backbone C/CA
    AND excludes all carbons in polar residues (ARG, ASN, ASP, GLN, GLU,
    HIS, LYS, SER, THR, TYR, TRP). This ensures rho values computed here
    are consistent with what the GNN sees in phase2_vulnerability_scan.

    Args:
        structure: Biotite AtomArray

    Returns:
        Tuple of (coordinates, chain_ids, residue_ids) arrays

    Validates: Requirements 4.2
    """
    _POLAR_RESIDUES = {
        "ARG", "ASN", "ASP", "GLN", "GLU",
        "HIS", "LYS", "SER", "THR", "TYR", "TRP",
    }

    # Filter: element is C, not backbone C or CA, and not in a polar residue
    carbon_mask = (
        (structure.element == 'C')
        & (structure.atom_name != 'C')
        & (structure.atom_name != 'CA')
    )

    # Build polar residue mask
    polar_mask = np.array([
        rn.strip() in _POLAR_RESIDUES for rn in structure.res_name
    ])

    apolar_carbon_mask = carbon_mask & ~polar_mask

    coords = structure.coord[apolar_carbon_mask]
    chain_ids = structure.chain_id[apolar_carbon_mask]
    res_ids = structure.res_id[apolar_carbon_mask]

    return coords, chain_ids, res_ids


def _find_hydrogen_bonds(
    donors: List[Tuple[str, int, np.ndarray]],
    acceptors: List[Tuple[str, int, np.ndarray]],
    max_distance: float,
    min_sequence_separation: int
) -> List[Tuple[str, int, str, int, np.ndarray, float, bool]]:
    """
    Find potential hydrogen bonds based on distance and sequence separation.

    Uses a KD-tree on acceptor coordinates for O(N log N) instead of O(N²).

    Args:
        donors: List of (chain_id, res_id, coord) for donor atoms
        acceptors: List of (chain_id, res_id, coord) for acceptor atoms
        max_distance: Maximum N-O distance for H-bond
        min_sequence_separation: Minimum residue separation

    Returns:
        List of (donor_chain, donor_res, acceptor_chain, acceptor_res, midpoint, distance, is_interchain) tuples

    Validates: Requirements 4.3, 4.4, 4.5
    """
    if not donors or not acceptors:
        return []

    acceptor_coords = np.array([a[2] for a in acceptors])
    tree = cKDTree(acceptor_coords)

    hbonds = []
    for donor_chain, donor_res, donor_coord in donors:
        indices = tree.query_ball_point(donor_coord, max_distance)
        for j in indices:
            acceptor_chain, acceptor_res, acceptor_coord = acceptors[j]

            # Sequence separation only applies within the same chain
            if donor_chain == acceptor_chain:
                if abs(donor_res - acceptor_res) <= min_sequence_separation:
                    continue

            distance = np.linalg.norm(donor_coord - acceptor_coord)
            midpoint = (donor_coord + acceptor_coord) / 2.0
            is_interchain = donor_chain != acceptor_chain

            hbonds.append((donor_chain, donor_res, acceptor_chain, acceptor_res, midpoint, distance, is_interchain))

    return hbonds

