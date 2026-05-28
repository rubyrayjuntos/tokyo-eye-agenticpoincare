"""
Void Detection Service

Identifies volumetric cavities and packing defects in protein structures.
Uses 3D grid generation, spatial indexing (KDTree), and DBSCAN clustering.

Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8
"""

from typing import List, Tuple, Optional
import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
import biotite.structure as struc

from gosp.models.data_models import Void, StructureData, Dehydron


class VoidDetectionError(Exception):
    """Raised when void detection fails."""
    pass


class VoidDetectionResult:
    """Container for void detection results."""
    
    def __init__(
        self,
        voids: List[Void],
        total_voids: int,
        total_volume: float,
        computation_time_sec: float
    ):
        self.voids = voids
        self.total_voids = total_voids
        self.total_volume = total_volume
        self.computation_time_sec = computation_time_sec


# Van der Waals radii for common elements (in Angstroms)
VDW_RADII = {
    'H': 1.20,
    'C': 1.70,
    'N': 1.55,
    'O': 1.52,
    'S': 1.80,
    'P': 1.80,
    'F': 1.47,
    'CL': 1.75,
    'BR': 1.85,
    'I': 1.98
}


def detect_voids(
    structure: StructureData,
    dehydrons: Optional[List[Dehydron]] = None,
    grid_spacing: float = 0.5,
    vdw_tolerance: float = 1.09,
    min_volume: float = 10.0,
    dehydron_proximity_threshold: float = 6.0,
    dbscan_eps: float = 0.866,
    dbscan_min_samples: int = 3
) -> VoidDetectionResult:
    """
    Detect voids (volumetric cavities) in a protein structure.
    
    Algorithm:
    1. Generate 3D grid over protein bounding box (grid_spacing)
    2. Build KDTree from atomic coordinates
    3. Exclude grid points inside van der Waals radii (+ tolerance)
    4. Cluster void points using DBSCAN
    5. Filter clusters by minimum volume threshold
    6. Map voids to nearby dehydrons (within dehydron_proximity_threshold)
    
    Args:
        structure: StructureData object with coordinates
        dehydrons: Optional list of Dehydron objects for proximity mapping
        grid_spacing: Grid spacing in Angstroms (default: 0.5Å)
        vdw_tolerance: Tolerance added to vdW radii (default: 1.09Å)
        min_volume: Minimum void volume in Ų (default: 10.0Ų)
        dehydron_proximity_threshold: Distance threshold for void-dehydron mapping (default: 6.0Å)
        dbscan_eps: DBSCAN epsilon parameter (default: 0.866Å)
        dbscan_min_samples: DBSCAN minimum samples (default: 3)
    
    Returns:
        VoidDetectionResult with detected voids and statistics
    
    Raises:
        VoidDetectionError: If detection fails
    
    Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8
    """
    import time
    start_time = time.time()
    
    try:
        # Convert StructureData to Biotite structure for processing
        biotite_structure = _build_biotite_structure_from_data(structure)
        
        # Get atomic coordinates and elements
        atom_coords = biotite_structure.coord
        atom_elements = biotite_structure.element
        
        if len(atom_coords) == 0:
            # No atoms - return empty result
            computation_time = time.time() - start_time
            return VoidDetectionResult(
                voids=[],
                total_voids=0,
                total_volume=0.0,
                computation_time_sec=computation_time
            )
        
        # Generate 3D grid over protein bounding box (Requirement 5.1)
        grid_points = _generate_grid(atom_coords, grid_spacing)
        
        if len(grid_points) == 0:
            # No grid points - return empty result
            computation_time = time.time() - start_time
            return VoidDetectionResult(
                voids=[],
                total_voids=0,
                total_volume=0.0,
                computation_time_sec=computation_time
            )
        
        # Build KDTree from atomic coordinates (Requirement 5.2)
        kdtree = KDTree(atom_coords)
        
        # Exclude grid points inside van der Waals radii (Requirement 5.3)
        void_points = _filter_void_points(
            grid_points, kdtree, atom_coords, atom_elements, vdw_tolerance
        )
        
        if len(void_points) == 0:
            # No void points - return empty result
            computation_time = time.time() - start_time
            return VoidDetectionResult(
                voids=[],
                total_voids=0,
                total_volume=0.0,
                computation_time_sec=computation_time
            )
        
        # Cluster void points using DBSCAN (Requirement 5.4)
        clusters = _cluster_void_points(void_points, dbscan_eps, dbscan_min_samples)
        
        # Filter clusters by minimum volume and create Void objects (Requirement 5.5)
        # Exclude clusters that touch the sampling boundary (outside solvent region)
        grid_min = np.min(grid_points, axis=0)
        grid_max = np.max(grid_points, axis=0)
        voids_list = _create_voids_from_clusters(
            clusters, void_points, grid_spacing, min_volume, grid_min, grid_max
        )
        
        # Map voids to nearby dehydrons (Requirement 5.7)
        if dehydrons:
            _map_voids_to_dehydrons(voids_list, dehydrons, dehydron_proximity_threshold)
        
        # Calculate statistics
        total_voids = len(voids_list)
        total_volume = sum(v.volume for v in voids_list)
        
        computation_time = time.time() - start_time
        
        return VoidDetectionResult(
            voids=voids_list,
            total_voids=total_voids,
            total_volume=total_volume,
            computation_time_sec=computation_time
        )
    
    except Exception as e:
        raise VoidDetectionError(f"Void detection failed: {str(e)}")


def _build_biotite_structure_from_data(structure_data: StructureData) -> struc.AtomArray:
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


def _generate_grid(
    atom_coords: np.ndarray,
    spacing: float
) -> np.ndarray:
    """
    Generate 3D grid over protein bounding box.
    
    Args:
        atom_coords: Atomic coordinates (N x 3)
        spacing: Grid spacing in Angstroms
    
    Returns:
        Grid points (M x 3)
    
    Validates: Requirement 5.1
    """
    # Calculate bounding box
    min_coords = np.min(atom_coords, axis=0)
    max_coords = np.max(atom_coords, axis=0)
    
    # Add padding to bounding box
    padding = 2.0  # Angstroms
    min_coords -= padding
    max_coords += padding
    
    # Generate grid
    x_range = np.arange(min_coords[0], max_coords[0], spacing)
    y_range = np.arange(min_coords[1], max_coords[1], spacing)
    z_range = np.arange(min_coords[2], max_coords[2], spacing)
    
    # Create meshgrid
    xx, yy, zz = np.meshgrid(x_range, y_range, z_range, indexing='ij')
    
    # Flatten to list of points
    grid_points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    
    return grid_points


def _filter_void_points(
    grid_points: np.ndarray,
    kdtree: KDTree,
    atom_coords: np.ndarray,
    atom_elements: np.ndarray,
    tolerance: float
) -> np.ndarray:
    """
    Exclude grid points inside van der Waals radii.
    
    Args:
        grid_points: Grid points (M x 3)
        kdtree: KDTree of atomic coordinates
        atom_coords: Atomic coordinates (N x 3)
        atom_elements: Element symbols (N,)
        tolerance: Tolerance added to vdW radii
    
    Returns:
        Filtered void points (K x 3)
    
    Validates: Requirement 5.3
    """
    void_mask = np.ones(len(grid_points), dtype=bool)
    
    # For each grid point, check if it's inside any atom's vdW radius
    for i, point in enumerate(grid_points):
        # Find nearest atom
        distance, nearest_idx = kdtree.query(point)
        
        # Get vdW radius for nearest atom
        element = atom_elements[nearest_idx].upper()
        vdw_radius = VDW_RADII.get(element, 1.70)  # Default to carbon radius
        
        # Check if point is inside vdW radius + tolerance
        if distance < (vdw_radius + tolerance):
            void_mask[i] = False
    
    return grid_points[void_mask]


def _cluster_void_points(
    void_points: np.ndarray,
    eps: float,
    min_samples: int
) -> np.ndarray:
    """
    Cluster void points using DBSCAN.
    
    Args:
        void_points: Void points (K x 3)
        eps: DBSCAN epsilon parameter
        min_samples: DBSCAN minimum samples
    
    Returns:
        Cluster labels (K,) where -1 indicates noise
    
    Validates: Requirement 5.4
    """
    if len(void_points) == 0:
        return np.array([])
    
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(void_points)
    
    return labels


def _create_voids_from_clusters(
    labels: np.ndarray,
    void_points: np.ndarray,
    grid_spacing: float,
    min_volume: float,
    grid_min: np.ndarray,
    grid_max: np.ndarray
) -> List[Void]:
    """
    Create Void objects from clusters, filtering by minimum volume.
    
    Args:
        labels: Cluster labels (K,)
        void_points: Void points (K x 3)
        grid_spacing: Grid spacing in Angstroms
        min_volume: Minimum void volume in Ų
    
    Returns:
        List of Void objects
    
    Validates: Requirements 5.5, 5.6
    """
    voids_list = []
    
    # Calculate volume per grid point
    volume_per_point = grid_spacing ** 3
    
    # Get unique cluster labels (excluding noise label -1)
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != -1]
    
    void_id = 1
    boundary_tol = grid_spacing * 1.5

    for label in unique_labels:
        # Get points in this cluster
        cluster_mask = labels == label
        cluster_points = void_points[cluster_mask]

        # Skip clusters that touch the boundary (exterior solvent region)
        touches_min = np.any(cluster_points <= (grid_min + boundary_tol))
        touches_max = np.any(cluster_points >= (grid_max - boundary_tol))
        if touches_min or touches_max:
            continue
        
        # Calculate volume
        point_count = len(cluster_points)
        volume = point_count * volume_per_point
        
        # Filter by minimum volume (Requirement 5.5)
        if volume >= min_volume:
            # Calculate center
            center = np.mean(cluster_points, axis=0)
            
            # Create Void object (Requirement 5.6)
            void = Void(
                void_id=void_id,
                center=tuple(center.tolist()),
                volume=float(volume),
                point_count=point_count,
                nearby_dehydrons=[],
                points=cluster_points.tolist()
            )
            voids_list.append(void)
            void_id += 1
    
    return voids_list


def _map_voids_to_dehydrons(
    voids: List[Void],
    dehydrons: List[Dehydron],
    threshold: float
) -> None:
    """
    Map voids to nearby dehydrons within threshold distance.
    
    Modifies voids in-place by populating nearby_dehydrons field.
    
    Args:
        voids: List of Void objects
        dehydrons: List of Dehydron objects
        threshold: Distance threshold in Angstroms
    
    Validates: Requirement 5.7
    """
    for void in voids:
        void_center = np.array(void.center)
        
        for i, dehydron in enumerate(dehydrons):
            dehydron_midpoint = np.array(dehydron.midpoint)
            
            # Calculate distance
            distance = np.linalg.norm(void_center - dehydron_midpoint)
            
            # Add to nearby_dehydrons if within threshold
            if distance <= threshold:
                void.nearby_dehydrons.append(i)
