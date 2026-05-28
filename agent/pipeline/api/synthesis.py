"""
Synthesis API endpoints.

Provides endpoints for synthesis feasibility checking and Autoprotocol generation.

Validates: Requirements 10.1-10.6, 11.1-11.7
"""

from fastapi import APIRouter, HTTPException, status
from typing import Dict
import logging

from gosp.models.data_models import (
    SynthesisFeasibilityRequest,
    SynthesisFeasibilityResponse,
    SynthesisFeasibilityResult,
    AutoprotocolRequest,
    AutoprotocolResponse,
)
from gosp.services.synthesis_feasibility import (
    check_synthesis_feasibility,
    SynthesisFeasibilityError,
    SequenceExtractionError,
    ReverseTranslationError,
    TwistAPIError,
)
from gosp.services.autoprotocol_generation import (
    generate_cfps_protocol,
    validate_autoprotocol_schema,
    AutoprotocolGenerationError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/synthesis", tags=["synthesis"])


@router.post(
    "/feasibility",
    response_model=SynthesisFeasibilityResponse,
    status_code=status.HTTP_200_OK,
    summary="Check synthesis feasibility",
    description="""
    Check synthesis feasibility for amino acid sequences.
    
    This endpoint:
    1. Extracts amino acid sequences from input
    2. Reverse-translates to DNA with E. coli codon optimization
    3. Optionally integrates with Twist Bioscience API for manufacturability assessment
    4. Returns complexity scores and synthesis issues
    
    Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.5, 10.6
    """,
)
async def check_feasibility(
    request: SynthesisFeasibilityRequest,
) -> SynthesisFeasibilityResponse:
    """
    Check synthesis feasibility for amino acid sequences.
    
    Args:
        request: SynthesisFeasibilityRequest containing sequences and optional API key
        
    Returns:
        SynthesisFeasibilityResponse with results for each sequence
        
    Raises:
        HTTPException: If feasibility check fails
    """
    try:
        logger.info(f"Checking feasibility for {len(request.sequences)} sequences")
        
        # Check synthesis feasibility
        results = check_synthesis_feasibility(
            sequences=request.sequences,
            api_key=request.api_key
        )
        
        logger.info(f"Feasibility check complete: {len(results)} results")
        
        return SynthesisFeasibilityResponse(results=results)
        
    except SequenceExtractionError as e:
        logger.error(f"Sequence extraction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Sequence extraction failed: {str(e)}"
        )
    except ReverseTranslationError as e:
        logger.error(f"Reverse translation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Reverse translation failed: {str(e)}"
        )
    except TwistAPIError as e:
        logger.error(f"Twist API error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Twist Bioscience API unavailable: {str(e)}"
        )
    except SynthesisFeasibilityError as e:
        logger.error(f"Synthesis feasibility error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Synthesis feasibility check failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error in feasibility check: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )


@router.post(
    "/autoprotocol",
    response_model=AutoprotocolResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate Autoprotocol for CFPS",
    description="""
    Generate Autoprotocol JSON for cell-free protein synthesis (CFPS) and purification.
    
    This endpoint:
    1. Generates CFPS protocol with reagent mixing
    2. Includes Ni-NTA bead purification steps
    3. Specifies wash cycles and elution conditions
    4. Validates against Autoprotocol schema
    
    Validates: Requirements 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7
    """,
)
async def generate_autoprotocol(
    request: AutoprotocolRequest,
) -> AutoprotocolResponse:
    """
    Generate Autoprotocol JSON for cell-free protein synthesis.
    
    Args:
        request: AutoprotocolRequest containing sequence and parameters
        
    Returns:
        AutoprotocolResponse with protocol JSON and metadata
        
    Raises:
        HTTPException: If protocol generation fails
    """
    try:
        logger.info(f"Generating Autoprotocol for sequence length {len(request.sequence)}")
        
        # Generate CFPS protocol
        result = generate_cfps_protocol(
            sequence=request.sequence,
            template_volume_ul=request.template_volume_ul,
            extract_volume_ul=request.extract_volume_ul,
            incubation_time_hours=request.incubation_time_hours,
            temperature_celsius=request.temperature_celsius,
        )
        
        # Validate schema
        is_valid = validate_autoprotocol_schema(result["protocol"])
        
        logger.info(f"Autoprotocol generated: {result['instruction_count']} instructions, valid={is_valid}")
        
        return AutoprotocolResponse(
            protocol=result["protocol"],
            instruction_count=result["instruction_count"],
            schema_valid=is_valid,
            metadata=result["metadata"]
        )
        
    except AutoprotocolGenerationError as e:
        logger.error(f"Autoprotocol generation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Autoprotocol generation failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error in autoprotocol generation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )
