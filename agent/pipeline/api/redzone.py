"""
Red Zone API Endpoints

Provides REST API for safety screening including immunogenicity and metabolic liability.

Validates: Requirements 7.1-7.7, 8.1-8.7
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field, ConfigDict

from gosp.models.data_models import StructureData, RedZoneFlag, RedZoneFlagDetails
from gosp.services.immunogenicity_screening import (
    screen_immunogenicity,
    ImmunogenicityScreeningError,
    Epitope
)
from gosp.services.metabolic_liability_screening import (
    screen_metabolic_liability,
    MetabolicLiabilityScreeningError,
    SiteOfMetabolism
)


router = APIRouter(prefix="/api/redzone", tags=["redzone"])


class ImmunogenicityScreeningRequest(BaseModel):
    """Request model for immunogenicity screening."""
    structure: StructureData = Field(..., description="Protein structure data")
    alleles: Optional[List[str]] = Field(
        None,
        description="HLA alleles to test (default: DRB1*01:01, DRB1*15:01, DRB1*03:01)"
    )
    ic50_threshold: float = Field(
        500.0,
        gt=0,
        description="IC50 threshold for flagging in nM (default: 500.0)"
    )
    sasa_threshold: float = Field(
        20.0,
        gt=0,
        le=100,
        description="SASA percentage threshold for exposed residues (default: 20.0)"
    )
    peptide_length: int = Field(
        15,
        ge=9,
        le=25,
        description="Length of peptide sequences to extract (default: 15)"
    )


class EpitopeResponse(BaseModel):
    """Response model for individual epitope."""
    sequence: str = Field(..., description="Peptide sequence")
    start_res: int = Field(..., ge=1, description="Start residue ID")
    end_res: int = Field(..., ge=1, description="End residue ID")
    allele: str = Field(..., description="HLA allele")
    ic50: float = Field(..., gt=0, description="Predicted IC50 in nM")
    sasa_percent: float = Field(..., ge=0, le=100, description="SASA percentage")
    flagged: bool = Field(..., description="True if IC50 < threshold")


class ImmunogenicityScreeningResponse(BaseModel):
    """Response model for immunogenicity screening."""
    epitopes: List[EpitopeResponse] = Field(..., description="List of predicted epitopes")
    total_peptides: int = Field(..., ge=0, description="Total number of peptides screened")
    flagged_count: int = Field(..., ge=0, description="Number of flagged epitopes")
    computation_time_sec: float = Field(..., ge=0, description="Computation time in seconds")
    warning: Optional[str] = Field(None, description="Warning message if API unavailable")
    red_zone_flags: List[RedZoneFlag] = Field(
        default_factory=list,
        description="Red Zone flags for flagged epitopes"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "epitopes": [
                    {
                        "sequence": "AKFQSEEQQQTEDEL",
                        "start_res": 45,
                        "end_res": 59,
                        "allele": "DRB1*01:01",
                        "ic50": 340.0,
                        "sasa_percent": 65.3,
                        "flagged": True
                    }
                ],
                "total_peptides": 156,
                "flagged_count": 8,
                "computation_time_sec": 12.5,
                "warning": None,
                "red_zone_flags": [
                    {
                        "flag_type": "immunogenicity",
                        "severity": "high",
                        "details": {
                            "sequence": "AKFQSEEQQQTEDEL",
                            "start_res": 45,
                            "end_res": 59,
                            "allele": "DRB1*01:01",
                            "ic50": 340.0
                        }
                    }
                ]
            }
        }
    )


@router.post("/immunogenicity", response_model=ImmunogenicityScreeningResponse)
async def screen_immunogenicity_endpoint(
    request: ImmunogenicityScreeningRequest = Body(...)
) -> ImmunogenicityScreeningResponse:
    """
    Screen for immunogenic epitopes on solvent-exposed surfaces.
    
    This endpoint:
    - Identifies solvent-accessible residues (SASA > 20%)
    - Extracts 15-mer peptide sequences from exposed regions
    - Integrates NetMHCIIpan 4.1 API for HLA binding prediction
    - Flags peptides with IC50 < 500 nM as Red Zone violations
    
    Algorithm:
    1. Calculate SASA for all residues using FreeSASA
    2. Identify solvent-accessible residues (SASA > sasa_threshold%)
    3. Extract peptide sequences of specified length from exposed regions
    4. Submit peptides to NetMHCIIpan 4.1 for HLA binding prediction
    5. Evaluate binding affinities for specified HLA alleles
    6. Flag peptides with IC50 < ic50_threshold nM
    7. Create Red Zone flags for flagged epitopes
    
    Args:
        request: ImmunogenicityScreeningRequest with structure and parameters
    
    Returns:
        ImmunogenicityScreeningResponse with epitopes and Red Zone flags
    
    Raises:
        HTTPException: If screening fails
    
    Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7
    """
    try:
        # Screen for immunogenicity
        result = screen_immunogenicity(
            structure=request.structure,
            alleles=request.alleles,
            ic50_threshold=request.ic50_threshold,
            sasa_threshold=request.sasa_threshold,
            peptide_length=request.peptide_length
        )
        
        # Convert epitopes to response format
        epitope_responses = []
        red_zone_flags = []
        
        for epitope in result.epitopes:
            epitope_response = EpitopeResponse(
                sequence=epitope.sequence,
                start_res=epitope.start_res,
                end_res=epitope.end_res,
                allele=epitope.allele,
                ic50=epitope.ic50,
                sasa_percent=epitope.sasa_percent,
                flagged=epitope.flagged
            )
            epitope_responses.append(epitope_response)
            
            # Create Red Zone flag for flagged epitopes
            if epitope.flagged:
                # Determine severity based on IC50
                if epitope.ic50 < 100:
                    severity = "high"
                elif epitope.ic50 < 300:
                    severity = "medium"
                else:
                    severity = "low"
                
                flag = RedZoneFlag(
                    flag_type="immunogenicity",
                    severity=severity,
                    details=RedZoneFlagDetails(
                        sequence=epitope.sequence,
                        start_res=epitope.start_res,
                        end_res=epitope.end_res,
                        allele=epitope.allele,
                        ic50=epitope.ic50
                    )
                )
                red_zone_flags.append(flag)
        
        return ImmunogenicityScreeningResponse(
            epitopes=epitope_responses,
            total_peptides=result.total_peptides,
            flagged_count=result.flagged_count,
            computation_time_sec=result.computation_time_sec,
            warning=result.warning,
            red_zone_flags=red_zone_flags
        )
    
    except ImmunogenicityScreeningError as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": True,
                "error_type": "ImmunogenicityScreeningError",
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



class MetabolicLiabilityScreeningRequest(BaseModel):
    """Request model for metabolic liability screening."""
    structure: StructureData = Field(..., description="Protein structure data")
    isoform: str = Field(
        "3A4",
        description="CYP450 isoform to screen (default: 3A4)"
    )
    sasa_threshold: float = Field(
        15.0,
        gt=0,
        le=100,
        description="SASA percentage threshold for exposed sites (default: 15.0)"
    )
    flag_threshold: int = Field(
        2,
        ge=1,
        description="Number of exposed SOMs to trigger flag (default: 2)"
    )


class SiteOfMetabolismResponse(BaseModel):
    """Response model for individual site of metabolism."""
    atom_idx: int = Field(..., ge=0, description="Atom index")
    atom_type: str = Field(..., description="Atom element type")
    pattern: str = Field(..., description="Matched SMARTS pattern name")
    coords: tuple[float, float, float] = Field(..., description="3D coordinates")
    sasa: float = Field(..., ge=0, le=100, description="SASA percentage")
    is_exposed: bool = Field(..., description="True if SASA > threshold")


class MetabolicLiabilityScreeningResponse(BaseModel):
    """Response model for metabolic liability screening."""
    vulnerable_atoms: List[int] = Field(..., description="List of vulnerable atom indices")
    som_count: int = Field(..., ge=0, description="Total number of sites of metabolism")
    patterns_matched: List[str] = Field(..., description="List of matched SMARTS patterns")
    surface_exposed_soms: List[int] = Field(
        ..., 
        description="List of surface-exposed SOM atom indices"
    )
    sites: List[SiteOfMetabolismResponse] = Field(..., description="Detailed site information")
    flagged: bool = Field(..., description="True if ≥ threshold exposed SOMs")
    computation_time_sec: float = Field(..., ge=0, description="Computation time in seconds")
    warning: Optional[str] = Field(None, description="Warning message if conversion fails")
    red_zone_flags: List[RedZoneFlag] = Field(
        default_factory=list,
        description="Red Zone flags if metabolic liability detected"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "vulnerable_atoms": [3, 7, 15, 22],
                "som_count": 4,
                "patterns_matched": ["aliphatic_ch", "aromatic_ch"],
                "surface_exposed_soms": [7, 15],
                "sites": [
                    {
                        "atom_idx": 7,
                        "atom_type": "C",
                        "pattern": "aromatic_ch",
                        "coords": [10.5, 15.2, 8.9],
                        "sasa": 18.5,
                        "is_exposed": True
                    }
                ],
                "flagged": True,
                "computation_time_sec": 3.2,
                "warning": None,
                "red_zone_flags": [
                    {
                        "flag_type": "metabolism",
                        "severity": "high",
                        "details": {
                            "vulnerable_atoms": [7, 15],
                            "isoform": "3A4",
                            "som_count": 2
                        }
                    }
                ]
            }
        }
    )


@router.post("/metabolism", response_model=MetabolicLiabilityScreeningResponse)
async def screen_metabolic_liability_endpoint(
    request: MetabolicLiabilityScreeningRequest = Body(...)
) -> MetabolicLiabilityScreeningResponse:
    """
    Screen for metabolic liability by identifying CYP450 sites of metabolism.
    
    This endpoint:
    - Converts structures to SMILES using RDKit
    - Applies SMARTS patterns for CYP3A4 sites
    - Maps vulnerable atoms to 3D coordinates
    - Filters for surface-exposed sites (SASA > 15%)
    - Flags structures with ≥2 exposed SOMs
    
    Algorithm:
    1. Convert structure to SMILES representation using RDKit
    2. Apply SMARTS patterns for CYP450 sites (aliphatic C-H, aromatic C-H, etc.)
    3. Identify vulnerable atoms from pattern matches
    4. Map vulnerable atoms to 3D coordinates
    5. Calculate SASA for vulnerable atoms
    6. Filter for surface-exposed sites (SASA > sasa_threshold%)
    7. Flag structures with ≥ flag_threshold exposed SOMs
    8. Create Red Zone flags for flagged structures
    
    Args:
        request: MetabolicLiabilityScreeningRequest with structure and parameters
    
    Returns:
        MetabolicLiabilityScreeningResponse with vulnerable sites and Red Zone flags
    
    Raises:
        HTTPException: If screening fails
    
    Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7
    """
    try:
        # Screen for metabolic liability
        result = screen_metabolic_liability(
            structure=request.structure,
            isoform=request.isoform,
            sasa_threshold=request.sasa_threshold,
            flag_threshold=request.flag_threshold
        )
        
        # Convert sites to response format
        site_responses = []
        for site in result.sites:
            site_response = SiteOfMetabolismResponse(
                atom_idx=site.atom_idx,
                atom_type=site.atom_type,
                pattern=site.pattern,
                coords=site.coords,
                sasa=site.sasa,
                is_exposed=site.is_exposed
            )
            site_responses.append(site_response)
        
        # Create Red Zone flag if flagged
        red_zone_flags = []
        if result.flagged:
            # Determine severity based on number of exposed SOMs
            exposed_count = len(result.surface_exposed_soms)
            if exposed_count >= 5:
                severity = "high"
            elif exposed_count >= 3:
                severity = "medium"
            else:
                severity = "low"
            
            flag = RedZoneFlag(
                flag_type="metabolism",
                severity=severity,
                details=RedZoneFlagDetails(
                    vulnerable_atoms=result.surface_exposed_soms,
                    isoform=request.isoform,
                    som_count=exposed_count
                )
            )
            red_zone_flags.append(flag)
        
        return MetabolicLiabilityScreeningResponse(
            vulnerable_atoms=result.vulnerable_atoms,
            som_count=result.som_count,
            patterns_matched=result.patterns_matched,
            surface_exposed_soms=result.surface_exposed_soms,
            sites=site_responses,
            flagged=result.flagged,
            computation_time_sec=result.computation_time_sec,
            warning=result.warning,
            red_zone_flags=red_zone_flags
        )
    
    except MetabolicLiabilityScreeningError as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": True,
                "error_type": "MetabolicLiabilityScreeningError",
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
