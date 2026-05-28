"""
Empirical Validation API Endpoints

Provides REST API for empirical validation including HDX-MS correlation
with predicted dehydron wrapping counts.

Validates: Requirements 9.1-9.7
"""

from typing import List
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field, ConfigDict

from gosp.models.data_models import Dehydron
from gosp.services.hdx_ms_validation import (
    compute_hdx_correlation,
    HDXCorrelationResult,
    HDXMSError
)


router = APIRouter(prefix="/api/empirical", tags=["empirical"])


class HDXCorrelationRequest(BaseModel):
    """Request model for HDX-MS correlation analysis."""
    csv_content: str = Field(
        ...,
        description="HDX-MS data in DynamX CSV format"
    )
    sequence: str = Field(
        ...,
        min_length=1,
        description="Full protein amino acid sequence"
    )
    dehydrons: List[Dehydron] = Field(
        ...,
        description="List of detected dehydrons with wrapping counts"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "csv_content": "Start,End,Sequence,Deut,MaxUptake\n1,15,NLYIQWLKDGGPSSG,2.5,15.0\n",
                "sequence": "NLYIQWLKDGGPSSGRPPPS",
                "dehydrons": [
                    {
                        "donor_res_id": 5,
                        "acceptor_res_id": 12,
                        "wrapping_count": 14,
                        "midpoint": [10.2, 15.3, 8.7],
                        "distance": 2.9,
                        "is_dehydron": True
                    }
                ]
            }
        }
    )


@router.post("/hdx-correlation", response_model=HDXCorrelationResult)
async def hdx_correlation_endpoint(
    request: HDXCorrelationRequest = Body(...)
) -> HDXCorrelationResult:
    """
    Compute Spearman correlation between HDX-MS data and wrapping counts.
    
    This endpoint:
    - Parses HDX-MS data in DynamX CSV format
    - Maps peptide ranges to residue IDs
    - Calculates fractional deuterium uptake per residue
    - Averages overlapping peptide segments
    - Computes Spearman correlation with wrapping counts
    - Returns R², p-value, and pass/fail status
    
    Algorithm:
    1. Parse DynamX CSV format (Start, End, Sequence, Deut, MaxUptake columns)
    2. Validate peptide ranges against protein sequence
    3. Calculate fractional uptake = Deut / MaxUptake for each peptide
    4. Map peptide data to individual residues
    5. Average uptake values for residues covered by multiple peptides
    6. Map dehydron wrapping counts to residues
    7. Compute Spearman correlation between wrapping and uptake
    8. Calculate R² = rho²
    9. Determine pass if R² ≥ 0.50 AND p < 0.05
    
    Expected correlation: Negative (more wrapping → less solvent exposure → less uptake)
    
    Args:
        request: HDXCorrelationRequest with CSV data, sequence, and dehydrons
    
    Returns:
        HDXCorrelationResult with correlation statistics and per-residue data
    
    Raises:
        HTTPException: If correlation computation fails
    
    Validates: Requirements 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7
    """
    try:
        result = compute_hdx_correlation(
            csv_content=request.csv_content,
            sequence=request.sequence,
            dehydrons=request.dehydrons
        )
        
        return result
    
    except HDXMSError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": True,
                "error_type": "HDXMSError",
                "message": str(e),
                "details": {},
                "suggested_action": "Check CSV format and ensure peptide ranges match sequence"
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
