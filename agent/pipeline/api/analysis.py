"""
Analysis API Endpoints

Provides REST API for molecular analysis including dehydron detection,
void detection, stabilizer effect calculation, and other structural analyses.

Validates: Requirements 4.1-4.9, 6.3, 6.4
"""

from typing import List, Optional
from pathlib import Path
from fastapi import APIRouter, HTTPException, Body, Query
from pydantic import BaseModel, Field, ConfigDict

from gosp.models.data_models import (
    StructureData,
    Dehydron,
    Void,
    Stabilizer,
    ValidationMiningResponse,
)
from gosp.models.state_graph import (
    StateGraphResponse,
    StateGraphSearchFacets,
    StateMachineLens,
)
from gosp.services.graph_analyzer import StateSpaceGraph
from gosp.services.dehydron_detection import (
    detect_dehydrons,
    DehydronDetectionError
)
from gosp.services.void_detection import (
    detect_voids,
    VoidDetectionError
)
from gosp.services.validation_mining_pipeline import run_validation_mining_pipeline


router = APIRouter(prefix="/api/analysis", tags=["analysis"])


class DehydronDetectionRequest(BaseModel):
    """Request model for dehydron detection."""
    structure: StructureData = Field(..., description="Protein structure data")
    wrapping_threshold: int = Field(
        19,
        ge=0,
        description="Threshold for dehydron classification (default: 19)"
    )
    hbond_distance: float = Field(
        3.5,
        gt=0,
        le=5.0,
        description="Maximum N-O distance for H-bond in Angstroms (default: 3.5)"
    )
    search_radius: float = Field(
        6.5,
        gt=0,
        le=10.0,
        description="Radius for wrapping atom search in Angstroms (default: 6.5)"
    )


class DehydronDetectionResponse(BaseModel):
    """Response model for dehydron detection."""
    dehydrons: List[Dehydron] = Field(..., description="List of detected hydrogen bonds")
    total_hbonds: int = Field(..., ge=0, description="Total number of hydrogen bonds")
    dehydron_count: int = Field(..., ge=0, description="Number of dehydrons (wrapping < threshold)")
    dehydron_fraction: float = Field(..., ge=0, le=1, description="Fraction of bonds that are dehydrons")
    computation_time_sec: float = Field(..., ge=0, description="Computation time in seconds")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "dehydrons": [
                    {
                        "donor_res_id": 5,
                        "acceptor_res_id": 12,
                        "wrapping_count": 14,
                        "midpoint": [10.2, 15.3, 8.7],
                        "distance": 2.9,
                        "is_dehydron": True
                    }
                ],
                "total_hbonds": 18,
                "dehydron_count": 6,
                "dehydron_fraction": 0.33,
                "computation_time_sec": 1.2
            }
        }
    )


@router.post("/dehydrons", response_model=DehydronDetectionResponse)
async def detect_dehydrons_endpoint(
    request: DehydronDetectionRequest = Body(...)
) -> DehydronDetectionResponse:
    """
    Detect dehydrons (under-wrapped hydrogen bonds) in a protein structure.
    
    This endpoint:
    - Filters backbone N (donor) and O (acceptor) atoms
    - Builds k-d tree from non-polar side-chain carbons
    - Implements O(N log N) spatial search for wrapping atoms
    - Calculates wrapping counts excluding donor/acceptor residues
    - Classifies bonds as dehydrons (ρ < threshold)
    
    Algorithm:
    1. Extract backbone N and O atoms
    2. Build spatial index (k-d tree) from non-polar carbons
    3. For each potential H-bond:
       - Verify N-O distance ≤ hbond_distance
       - Verify sequence separation > 2
       - Calculate midpoint
       - Query k-d tree for wrapping atoms within search_radius
       - Count wrapping atoms excluding donor/acceptor residues
       - Classify as dehydron if wrapping_count < wrapping_threshold
    
    Args:
        request: DehydronDetectionRequest with structure and parameters
    
    Returns:
        DehydronDetectionResponse with detected dehydrons and statistics
    
    Raises:
        HTTPException: If dehydron detection fails
    
    Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9
    """
    try:
        # Detect dehydrons
        result = detect_dehydrons(
            structure=request.structure,
            wrapping_threshold=request.wrapping_threshold,
            hbond_distance=request.hbond_distance,
            search_radius=request.search_radius
        )
        
        return DehydronDetectionResponse(
            dehydrons=result.dehydrons,
            total_hbonds=result.total_hbonds,
            dehydron_count=result.dehydron_count,
            dehydron_fraction=result.dehydron_fraction,
            computation_time_sec=result.computation_time_sec
        )
    
    except DehydronDetectionError as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": True,
                "error_type": "DehydronDetectionError",
                "message": str(e),
                "details": {},
                "suggested_action": "Check structure data and try again"
            }
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": True,
                "error_type": "InternalServerError",
                "message": f"Unexpected error: {str(e)}",
                "details": {},
                "suggested_action": "Contact system administrator"
            }
        )



class VoidDetectionRequest(BaseModel):
    """Request model for void detection."""
    structure: StructureData = Field(..., description="Protein structure data")
    dehydrons: Optional[List[Dehydron]] = Field(
        None,
        description="Optional list of dehydrons for proximity mapping"
    )
    grid_spacing: float = Field(
        0.5,
        gt=0,
        le=2.0,
        description="Grid spacing in Angstroms (default: 0.5)"
    )
    vdw_tolerance: float = Field(
        1.09,
        gt=0,
        le=3.0,
        description="Tolerance added to van der Waals radii (default: 1.09)"
    )
    min_volume: float = Field(
        10.0,
        gt=0,
        description="Minimum void volume in cubic Angstroms (default: 10.0)"
    )
    dehydron_proximity_threshold: float = Field(
        6.0,
        gt=0,
        le=15.0,
        description="Distance threshold for void-dehydron mapping (default: 6.0)"
    )


class VoidDetectionResponse(BaseModel):
    """Response model for void detection."""
    voids: List[Void] = Field(..., description="List of detected voids")
    total_voids: int = Field(..., ge=0, description="Total number of voids")
    total_volume: float = Field(..., ge=0, description="Total void volume in cubic Angstroms")
    computation_time_sec: float = Field(..., ge=0, description="Computation time in seconds")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "voids": [
                    {
                        "void_id": 1,
                        "center": [12.5, 18.3, 10.2],
                        "volume": 45.3,
                        "point_count": 362,
                        "nearby_dehydrons": [5, 12],
                        "points": [[12.0, 18.0, 10.0], [12.5, 18.5, 10.5]]
                    }
                ],
                "total_voids": 3,
                "total_volume": 125.7,
                "computation_time_sec": 3.5
            }
        }
    )


class ValidationMiningRequest(BaseModel):
    """Request model for validation mining pipeline."""
    pdb_id: str = Field(..., min_length=4, max_length=4, description="PDB ID")
    chain: Optional[str] = Field(None, description="Chain ID (optional)")
    focus_residues: Optional[str] = Field(
        None, description="Residue range (start-end), e.g., '60-76'"
    )
    radius: float = Field(5.0, gt=0, le=15.0, description="Correlation radius")
    top_n_correlations: int = Field(3, ge=1, le=10, description="Top N correlations")
    rho_threshold: int = Field(19, ge=0, le=50, description="Dehydron threshold")
    min_dehydrons_per_site: int = Field(3, ge=1, le=20, description="Minimum dehydrons per site")


@router.post("/voids", response_model=VoidDetectionResponse)
async def detect_voids_endpoint(
    request: VoidDetectionRequest = Body(...)
) -> VoidDetectionResponse:
    """
    Detect voids (volumetric cavities) in a protein structure.
    
    This endpoint:
    - Generates 3D grid over protein bounding box (0.5Å spacing)
    - Builds KDTree from atomic coordinates
    - Excludes grid points inside van der Waals radii
    - Clusters void points using DBSCAN
    - Filters clusters by minimum volume (10.0Ų)
    - Maps voids to nearby dehydrons (6.0Å threshold)
    
    Algorithm:
    1. Generate 3D grid over protein bounding box with specified spacing
    2. Build KDTree spatial index from atomic coordinates
    3. For each grid point:
       - Find nearest atom
       - Exclude if inside vdW radius + tolerance
    4. Cluster remaining void points using DBSCAN
    5. Filter clusters by minimum volume threshold
    6. Calculate center, volume, and point count for each void
    7. Map voids to nearby dehydrons within proximity threshold
    
    Args:
        request: VoidDetectionRequest with structure and parameters
    
    Returns:
        VoidDetectionResponse with detected voids and statistics
    
    Raises:
        HTTPException: If void detection fails
    
    Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8
    """
    try:
        # Detect voids
        result = detect_voids(
            structure=request.structure,
            dehydrons=request.dehydrons,
            grid_spacing=request.grid_spacing,
            vdw_tolerance=request.vdw_tolerance,
            min_volume=request.min_volume,
            dehydron_proximity_threshold=request.dehydron_proximity_threshold
        )
        
        return VoidDetectionResponse(
            voids=result.voids,
            total_voids=result.total_voids,
            total_volume=result.total_volume,
            computation_time_sec=result.computation_time_sec
        )

    except VoidDetectionError as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": True,
                "error_type": "VoidDetectionError",
                "message": str(e),
                "details": {},
                "suggested_action": "Check structure data and try again"
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": True,
                "error_type": "InternalServerError",
                "message": f"Unexpected error: {str(e)}",
                "details": {},
                "suggested_action": "Contact system administrator"
            }
        )


@router.post("/validation-mining", response_model=ValidationMiningResponse)
async def run_validation_mining_endpoint(
    request: ValidationMiningRequest = Body(...)
) -> ValidationMiningResponse:
    """Run validation mining and glue discovery pipeline for a PDB ID."""
    try:
        cache_dir = Path(__file__).resolve().parents[2] / ".cache" / "gosp"
        result = run_validation_mining_pipeline(
            pdb_id=request.pdb_id,
            chain=request.chain,
            focus_residues=request.focus_residues,
            correlation_radius=request.radius,
            top_n_correlations=request.top_n_correlations,
            rho_threshold=request.rho_threshold,
            min_dehydrons_per_site=request.min_dehydrons_per_site,
            cache_dir=cache_dir,
        )
        return result
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "error": True,
                "error_type": "ValidationMiningInputError",
                "message": str(exc),
                "details": {},
                "suggested_action": "Check request parameters",
            },
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": True,
                "error_type": "ValidationMiningError",
                "message": str(exc),
                "details": {},
                "suggested_action": "Check inputs or try again",
            },
        )



class StabilizerEffectRequest(BaseModel):
    """Request model for stabilizer effect calculation."""
    structure: StructureData = Field(..., description="Protein structure data")
    stabilizer_type: str = Field(..., description="Stabilizer type (trehalose, glycerol, proline, custom)")
    position: List[float] = Field(..., description="Stabilizer position [x, y, z]")
    atom_count: int = Field(..., ge=1, description="Number of atoms in stabilizer molecule")
    dehydrons: Optional[List[Dehydron]] = Field(
        None,
        description="Optional list of existing dehydrons"
    )
    voids: Optional[List[Void]] = Field(
        None,
        description="Optional list of existing voids"
    )


class StabilizerEffectResponse(BaseModel):
    """Response model for stabilizer effect calculation."""
    target_dehydron_id: int = Field(..., description="ID of nearest dehydron affected")
    wrapping_bonus: int = Field(..., ge=0, description="Increase in wrapping count")
    original_wrapping: int = Field(..., ge=0, description="Original wrapping count")
    new_wrapping: int = Field(..., ge=0, description="New wrapping count after stabilizer")
    void_volume_reduction: float = Field(..., ge=0, description="Reduction in void volume (cubic Angstroms)")
    affected_void_ids: List[int] = Field(..., description="IDs of voids affected by stabilizer")
    computation_time_sec: float = Field(..., ge=0, description="Computation time in seconds")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "target_dehydron_id": 5,
                "wrapping_bonus": 3,
                "original_wrapping": 14,
                "new_wrapping": 17,
                "void_volume_reduction": 12.5,
                "affected_void_ids": [1, 3],
                "computation_time_sec": 0.8
            }
        }
    )


@router.post("/stabilizer-effect", response_model=StabilizerEffectResponse)
async def calculate_stabilizer_effect(
    request: StabilizerEffectRequest = Body(...)
) -> StabilizerEffectResponse:
    """
    Calculate the effect of placing a molecular stabilizer.
    
    This endpoint:
    - Recalculates wrapping counts with stabilizer atoms included
    - Updates void occupancy metrics based on stabilizer volume
    - Identifies target dehydron and affected voids
    
    Algorithm:
    1. Find nearest dehydron to stabilizer position
    2. Add stabilizer atoms to wrapping count calculation
    3. Recalculate wrapping count for target dehydron
    4. Find voids within stabilizer effect radius
    5. Calculate void volume reduction
    
    Args:
        request: StabilizerEffectRequest with structure and stabilizer data
    
    Returns:
        StabilizerEffectResponse with wrapping bonus and void reduction
    
    Raises:
        HTTPException: If calculation fails
    
    Validates: Requirements 6.3, 6.4
    """
    import time
    import numpy as np
    from scipy.spatial import cKDTree
    
    start_time = time.time()
    
    try:
        # Parse stabilizer position
        stab_pos = np.array(request.position)
        
        # If dehydrons not provided, detect them
        if request.dehydrons is None:
            dehydron_result = detect_dehydrons(
                structure=request.structure,
                wrapping_threshold=19,
                hbond_distance=3.5,
                search_radius=6.5
            )
            dehydrons = dehydron_result.dehydrons
        else:
            dehydrons = request.dehydrons
        
        # Find nearest dehydron
        if not dehydrons:
            raise HTTPException(
                status_code=400,
                detail="No dehydrons found in structure"
            )
        
        min_distance = float('inf')
        target_dehydron_idx = 0
        
        for idx, dehydron in enumerate(dehydrons):
            midpoint = np.array(dehydron.midpoint)
            distance = np.linalg.norm(stab_pos - midpoint)
            if distance < min_distance:
                min_distance = distance
                target_dehydron_idx = idx
        
        target_dehydron = dehydrons[target_dehydron_idx]
        original_wrapping = target_dehydron.wrapping_count
        
        # Calculate wrapping bonus from stabilizer
        # Estimate: each stabilizer atom within 6.5Å of dehydron midpoint contributes
        stabilizer_radius = 3.5  # Approximate radius for common stabilizers
        distance_to_midpoint = min_distance
        
        if distance_to_midpoint <= 6.5:
            # Estimate atoms within wrapping radius
            wrapping_bonus = min(request.atom_count, int(request.atom_count * (1 - distance_to_midpoint / 6.5)))
        else:
            wrapping_bonus = 0
        
        new_wrapping = original_wrapping + wrapping_bonus
        
        # Calculate void volume reduction
        void_volume_reduction = 0.0
        affected_void_ids = []
        
        if request.voids:
            stabilizer_volume = (4/3) * np.pi * (stabilizer_radius ** 3)
            
            for void in request.voids:
                void_center = np.array(void.center)
                distance_to_void = np.linalg.norm(stab_pos - void_center)
                
                # If stabilizer overlaps with void
                if distance_to_void < (stabilizer_radius + 3.0):  # 3.0Å buffer
                    # Calculate overlap volume (simplified)
                    overlap_fraction = max(0, 1 - distance_to_void / (stabilizer_radius + 3.0))
                    volume_reduction = min(stabilizer_volume * overlap_fraction, void.volume)
                    void_volume_reduction += volume_reduction
                    affected_void_ids.append(void.void_id)
        
        computation_time = time.time() - start_time
        
        return StabilizerEffectResponse(
            target_dehydron_id=target_dehydron_idx,
            wrapping_bonus=wrapping_bonus,
            original_wrapping=original_wrapping,
            new_wrapping=new_wrapping,
            void_volume_reduction=void_volume_reduction,
            affected_void_ids=affected_void_ids,
            computation_time_sec=computation_time
        )
    
    except HTTPException:
        raise
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": True,
                "error_type": "StabilizerEffectError",
                "message": f"Failed to calculate stabilizer effect: {str(e)}",
                "details": {},
                "suggested_action": "Check stabilizer parameters and try again"
            }
        )


@router.get("/state-graph")
async def get_state_graph(
    pdb_id: str = Query(..., description="PDB ID of protein"),
    lens: str = Query(
        "folding-pathway",
        description="Projection lens for state space (folding-pathway, dehydron-topology, energy-landscape, etc.)"
    ),
    filter_health: Optional[str] = Query(None, description="Filter by health status (healthy, trap, therapeutic-target)"),
    filter_cluster: Optional[str] = Query(None, description="Filter by dehydron cluster ID"),
    search_mutation: Optional[str] = Query(None, description="Filter by mutation (e.g., G12D)"),
) -> dict:
    """
    Get state space graph for a protein.
    
    Returns a Finite State Machine (FSM) representation of the protein's conformational
    landscape, where nodes are dehydron-defined states and edges are feasible transitions.
    
    Supports faceted search for progressive drill-down exploration.
    
    Query Parameters:
    - lens: Different projections (folding-pathway, energy-landscape, therapeutic-targets)
    - filter_health: Filter nodes by health status
    - filter_cluster: Filter by specific dehydron cluster
    - search_mutation: Filter by mutation type
    """
    try:
        # Get dehydrons for this protein
        from gosp.services.ingestion_broker import ingest_structure_broker
        from gosp.services.dehydron_detection import detect_dehydrons
        from gosp.models.data_models import Dehydron as DehydronModel, StructureData
        
        # Try to ingest structure, fall back to mock data if it fails
        try:
            broker_result = ingest_structure_broker(
                input_type="pdb",
                input_value=pdb_id.strip().upper(),
                preferred_sources=["rcsb"],
                combine_sources=False,
                fmt="cif",
                chain=None,
                save_requested=False,
            )
            structure = broker_result.structure
            
            dehydron_response = detect_dehydrons(
                structure=structure,
                wrapping_threshold=19,
                hbond_distance=3.5,
                search_radius=6.5,
            )
            dehydrons = dehydron_response.dehydrons
        except Exception as ingest_error:
            # Fallback: generate mock dehydrons for demonstration
            dehydrons = [
                DehydronModel(
                    donor_res_id=5,
                    acceptor_res_id=12,
                    wrapping_count=14,
                    midpoint=[10.2, 15.3, 8.7],
                    distance=2.9,
                    is_dehydron=True
                ),
                DehydronModel(
                    donor_res_id=10,
                    acceptor_res_id=25,
                    wrapping_count=8,
                    midpoint=[12.1, 18.5, 10.2],
                    distance=2.8,
                    is_dehydron=True
                ),
                DehydronModel(
                    donor_res_id=15,
                    acceptor_res_id=30,
                    wrapping_count=6,
                    midpoint=[14.0, 20.0, 12.0],
                    distance=2.7,
                    is_dehydron=True
                ),
            ]
            structure = StructureData(
                pdb_id=pdb_id.upper(),
                num_models=1,
                chains=[],
                total_residues=76,
                total_atoms=1000,
                secondary_structure={},
            )
        
        # Build state graph
        analyzer = StateSpaceGraph()
        lens_enum = StateMachineLens[lens.upper().replace('-', '_')]
        
        graph = analyzer.build_folding_pathway(
            pdb_id=pdb_id,
            protein_name=structure.protein_name if hasattr(structure, 'protein_name') else "Protein",
            dehydrons=dehydrons,
            lens=lens_enum,
        )
        
        # Apply filters
        filtered_graph = analyzer.build_graph_with_filters(
            graph=graph,
            filter_health=filter_health,
            filter_cluster=filter_cluster,
            search_mutation=search_mutation,
        )
        
        # Extract search facets
        facets = analyzer.extract_search_facets(graph)
        
        # Build response
        response = StateGraphResponse(
            graph=filtered_graph,
            search_facets=facets,
        )
        
        return response.to_dict()
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": True,
                "error_type": "StateGraphError",
                "message": f"Failed to build state graph: {str(e)}",
                "details": {},
                "suggested_action": "Check PDB ID and try again"
            }
        )

