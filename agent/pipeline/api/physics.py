"""
Physics Kernel API Endpoints

Provides REST API for energy calculations and thermodynamic analysis.

Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5
"""

from typing import Literal, Dict, Tuple
from datetime import datetime
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field

from gosp.models.data_models import (
    StructureData,
    EnergyCalculationResult,
    DehydronBonusRequest,
    DehydronBonusResponse,
    DehydronBonusItem,
)
from gosp.services.physics_kernel import (
    calculate_energy,
    ForceFieldParameterizationError,
    EnergyMinimizationError,
    PhysicsKernelError
)
from gosp.services.dehydron_bonus_calculator import calculate_dehydron_bonus


router = APIRouter(prefix="/api/physics", tags=["physics"])


class EnergyCalculationRequest(BaseModel):
    """Request model for energy calculation."""
    structure: StructureData = Field(..., description="Protein structure data")
    force_field: Literal["MMFF94", "UFF"] = Field(
        "MMFF94",
        description="Force field to use for minimization"
    )
    max_iterations: int = Field(
        500,
        ge=1,
        le=10000,
        description="Maximum iterations for energy minimization"
    )
    timeout_sec: float = Field(
        5.0,
        gt=0,
        le=300,
        description="Timeout in seconds"
    )


class ErrorResponse(BaseModel):
    """Error response model."""
    error: bool = True
    error_type: str
    message: str
    details: dict = Field(default_factory=dict)
    suggested_action: str | None = None


@router.post("/energy", response_model=EnergyCalculationResult)
async def compute_energy(
    request: EnergyCalculationRequest = Body(
        ...,
        description="Energy calculation request with structure and parameters"
    )
) -> EnergyCalculationResult:
    """
    Calculate Gibbs free energy for a protein structure.
    
    This endpoint:
    - Performs force field minimization using RDKit (MMFF94 or UFF)
    - Computes SASA using FreeSASA
    - Calculates ΔG = potential_energy + (0.005 * SASA)
    - Returns within specified timeout (default 5 seconds)
    
    The calculation follows the formula:
    ΔG = E_potential + (0.005 kcal/mol/Ų × SASA)
    
    Args:
        request: EnergyCalculationRequest with structure and parameters
    
    Returns:
        EnergyCalculationResult with ΔG and component energies
    
    Raises:
        HTTPException: If calculation fails
    
    Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5
    """
    try:
        result = calculate_energy(
            structure=request.structure,
            force_field=request.force_field,
            max_iterations=request.max_iterations,
            timeout_sec=request.timeout_sec
        )
        
        return result
    
    except ForceFieldParameterizationError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": True,
                "error_type": "ForceFieldParameterizationError",
                "message": str(e),
                "details": {
                    "force_field": request.force_field,
                    "pdb_id": request.structure.pdb_id
                },
                "suggested_action": (
                    "Try using UFF force field as fallback, or check if structure "
                    "contains non-standard residues or atoms"
                )
            }
        )
    
    except EnergyMinimizationError as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": True,
                "error_type": "EnergyMinimizationError",
                "message": str(e),
                "details": {
                    "force_field": request.force_field,
                    "max_iterations": request.max_iterations,
                    "pdb_id": request.structure.pdb_id
                },
                "suggested_action": (
                    "Increase max_iterations or timeout, or try a different force field"
                )
            }
        )
    
    except PhysicsKernelError as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": True,
                "error_type": "PhysicsKernelError",
                "message": str(e),
                "details": {
                    "pdb_id": request.structure.pdb_id
                },
                "suggested_action": "Check structure validity and try again"
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


@router.post("/dehydron-bonus", response_model=DehydronBonusResponse)
async def compute_dehydron_bonus(
    request: DehydronBonusRequest = Body(
        ...,
        description="Dehydron bonus calculation request with dehydron list and optional baseline"
    )
) -> DehydronBonusResponse:
    """
    Calculate dehydron stability bonuses from wrapping count changes.
    
    This endpoint:
    - Computes bonus = wrapping_bound - wrapping_unbound for each dehydron
    - Identifies active dehydrons (wrapping_unbound < 19)
    - Calculates normalized stability weights
    - Returns detailed results for UI visualization and scoring
    
    The dehydron bonus reflects a stabilizing interaction when wrapping increases
    in the bound state, indicating burial of additional polar groups.
    
    Spec Reference: Section 6.6 "Dehydron Bonus"
    
    Args:
        request: DehydronBonusRequest with dehydrons list and optional baseline
    
    Returns:
        DehydronBonusResponse with individual bonuses and summary metrics
    
    Raises:
        HTTPException: If bonus calculation fails
    """
    try:
        # Parse baseline wrapping map if provided
        baseline_wrapping: Dict[Tuple[int, int], int] | None = None
        if request.baseline_wrapping:
            baseline_wrapping = {}
            for key, value in request.baseline_wrapping.items():
                # Key format: "res_a-res_b" or "res_a_res_b"
                parts = key.replace("-", "_").split("_")
                if len(parts) == 2:
                    try:
                        res_a, res_b = int(parts[0]), int(parts[1])
                        baseline_wrapping[(res_a, res_b)] = value
                    except (ValueError, IndexError):
                        pass  # Skip malformed entries, just log as warning

        # Calculate bonuses
        bonus_results, aggregates = calculate_dehydron_bonus(
            dehydrons=request.dehydrons,
            baseline_wrapping=baseline_wrapping
        )

        # Convert internal results to response items
        bonus_items = [
            DehydronBonusItem(
                dehydron_id=result.dehydron_id,
                residue_pair=result.residue_pair,
                wrapping_unbound=result.wrapping_unbound,
                wrapping_bound=result.wrapping_bound,
                bonus=result.bonus,
                is_dehydron=result.is_dehydron,
                stability_weight=result.stability_weight,
            )
            for result in bonus_results
        ]

        warnings = []
        if not request.dehydrons:
            warnings.append("Input dehydrons list is empty")

        return DehydronBonusResponse(
            status="success" if bonus_items else "partial",
            timestamp=datetime.utcnow(),
            bonuses=bonus_items,
            summary=aggregates,
            warnings=warnings,
            error=None,
        )

    except ValueError as e:
        return DehydronBonusResponse(
            status="error",
            timestamp=datetime.utcnow(),
            bonuses=[],
            summary={},
            warnings=[],
            error=f"Validation error: {str(e)}",
        )

    except Exception as e:
        return DehydronBonusResponse(
            status="error",
            timestamp=datetime.utcnow(),
            bonuses=[],
            summary={},
            warnings=[],
            error=f"Internal error: {str(e)}",
        )

