"""
HDX-MS Empirical Validation Service

Provides functionality for correlating predicted dehydron wrapping counts
with experimental HDX-MS (Hydrogen-Deuterium Exchange Mass Spectrometry) data.

Validates: Requirements 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7
"""

import csv
import time
from typing import List, Dict, Optional
from io import StringIO
import numpy as np
from scipy.stats import spearmanr
from pydantic import BaseModel, Field, ConfigDict

from gosp.models.data_models import Dehydron


class HDXMSError(Exception):
    """Base exception for HDX-MS validation errors."""
    pass


class HDXPeptide(BaseModel):
    """Represents a single HDX-MS peptide measurement."""
    start_res: int = Field(..., ge=1, description="Start residue ID")
    end_res: int = Field(..., ge=1, description="End residue ID")
    sequence: str = Field(..., min_length=1, description="Peptide sequence")
    deuterium_uptake: float = Field(..., ge=0, description="Deuterium uptake in Daltons")
    max_uptake: float = Field(..., gt=0, description="Maximum possible uptake")
    timepoint: Optional[float] = Field(None, ge=0, description="Timepoint in seconds")
    
    @property
    def fractional_uptake(self) -> float:
        """Calculate fractional deuterium uptake."""
        return self.deuterium_uptake / self.max_uptake if self.max_uptake > 0 else 0.0
    
    @property
    def length(self) -> int:
        """Calculate peptide length."""
        return self.end_res - self.start_res + 1


class ResidueHDXData(BaseModel):
    """HDX-MS data mapped to a single residue."""
    residue_id: int = Field(..., ge=1, description="Residue ID")
    fractional_uptake: float = Field(..., ge=0, le=1, description="Averaged fractional uptake")
    wrapping_count: Optional[int] = Field(None, ge=0, description="Predicted wrapping count")
    num_peptides: int = Field(..., ge=1, description="Number of overlapping peptides")


class HDXCorrelationResult(BaseModel):
    """Result of HDX-MS correlation analysis."""
    r_squared: float = Field(..., ge=0, le=1, description="R² coefficient of determination")
    p_value: float = Field(..., ge=0, le=1, description="Statistical p-value")
    spearman_rho: float = Field(..., ge=-1, le=1, description="Spearman correlation coefficient")
    num_residues: int = Field(..., ge=1, description="Number of residues in correlation")
    passed: bool = Field(..., description="True if R² ≥ 0.50 and p < 0.05")
    residue_data: List[ResidueHDXData] = Field(
        default_factory=list,
        description="Per-residue HDX and wrapping data"
    )
    computation_time_sec: float = Field(..., ge=0, description="Computation time in seconds")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "r_squared": 0.65,
                "p_value": 0.001,
                "spearman_rho": -0.78,
                "num_residues": 150,
                "passed": True,
                "residue_data": [],
                "computation_time_sec": 0.5
            }
        }
    )


def parse_dynamx_csv(csv_content: str, sequence: str) -> List[HDXPeptide]:
    """
    Parse HDX-MS data in DynamX CSV format.
    
    DynamX CSV format typically contains columns:
    - Start: Start residue number
    - End: End residue number
    - Sequence: Peptide sequence
    - Deut: Deuterium uptake (Da)
    - MaxUptake: Maximum possible uptake (Da)
    - Time: Timepoint (optional)
    
    Args:
        csv_content: CSV file content as string
        sequence: Full protein sequence for validation
    
    Returns:
        List of HDXPeptide objects
    
    Raises:
        HDXMSError: If CSV parsing fails or data is invalid
    
    Validates: Requirements 9.1
    """
    try:
        peptides = []
        csv_file = StringIO(csv_content)
        reader = csv.DictReader(csv_file)
        
        for row in reader:
            # Parse required fields
            start_res = int(row.get('Start', row.get('start', 0)))
            end_res = int(row.get('End', row.get('end', 0)))
            pep_sequence = row.get('Sequence', row.get('sequence', ''))
            deut_uptake = float(row.get('Deut', row.get('deut', row.get('Deuterium', 0))))
            max_uptake = float(row.get('MaxUptake', row.get('max_uptake', row.get('Max', 1))))
            
            # Parse optional timepoint
            timepoint = None
            if 'Time' in row or 'time' in row:
                timepoint = float(row.get('Time', row.get('time', 0)))
            
            # Validate residue IDs
            if start_res < 1 or end_res < start_res:
                raise HDXMSError(f"Invalid residue range: {start_res}-{end_res}")
            
            if end_res > len(sequence):
                raise HDXMSError(
                    f"Peptide end residue {end_res} exceeds sequence length {len(sequence)}"
                )
            
            peptides.append(HDXPeptide(
                start_res=start_res,
                end_res=end_res,
                sequence=pep_sequence,
                deuterium_uptake=deut_uptake,
                max_uptake=max_uptake,
                timepoint=timepoint
            ))
        
        if not peptides:
            raise HDXMSError("No valid peptides found in CSV")
        
        return peptides
    
    except (KeyError, ValueError) as e:
        raise HDXMSError(f"Failed to parse DynamX CSV: {str(e)}")


def map_peptides_to_residues(
    peptides: List[HDXPeptide],
    sequence_length: int
) -> Dict[int, List[float]]:
    """
    Map peptide HDX data to individual residues.
    
    For residues covered by multiple peptides, stores all fractional uptake values
    for later averaging.
    
    Args:
        peptides: List of HDX peptides
        sequence_length: Total number of residues in protein
    
    Returns:
        Dictionary mapping residue_id -> list of fractional uptake values
    
    Validates: Requirements 9.2, 9.3
    """
    residue_uptakes: Dict[int, List[float]] = {}
    
    for peptide in peptides:
        fractional_uptake = peptide.fractional_uptake
        
        # Map to all residues in peptide range
        for res_id in range(peptide.start_res, peptide.end_res + 1):
            if res_id < 1 or res_id > sequence_length:
                continue  # Skip invalid residue IDs
            
            if res_id not in residue_uptakes:
                residue_uptakes[res_id] = []
            
            residue_uptakes[res_id].append(fractional_uptake)
    
    return residue_uptakes


def average_overlapping_peptides(
    residue_uptakes: Dict[int, List[float]]
) -> Dict[int, float]:
    """
    Average fractional uptake for residues covered by multiple peptides.
    
    Args:
        residue_uptakes: Dictionary mapping residue_id -> list of uptake values
    
    Returns:
        Dictionary mapping residue_id -> averaged fractional uptake
    
    Validates: Requirements 9.4
    """
    averaged_uptakes = {}
    
    for res_id, uptakes in residue_uptakes.items():
        if uptakes:
            averaged_uptakes[res_id] = np.mean(uptakes)
    
    return averaged_uptakes


def map_wrapping_counts_to_residues(
    dehydrons: List[Dehydron]
) -> Dict[int, List[int]]:
    """
    Map dehydron wrapping counts to residues.
    
    Each residue can participate in multiple hydrogen bonds (as donor or acceptor),
    so we store all wrapping counts for later averaging.
    
    Args:
        dehydrons: List of detected dehydrons
    
    Returns:
        Dictionary mapping residue_id -> list of wrapping counts
    """
    residue_wrapping: Dict[int, List[int]] = {}
    
    for dehydron in dehydrons:
        # Map to donor residue
        if dehydron.donor_res_id not in residue_wrapping:
            residue_wrapping[dehydron.donor_res_id] = []
        residue_wrapping[dehydron.donor_res_id].append(dehydron.wrapping_count)
        
        # Map to acceptor residue
        if dehydron.acceptor_res_id not in residue_wrapping:
            residue_wrapping[dehydron.acceptor_res_id] = []
        residue_wrapping[dehydron.acceptor_res_id].append(dehydron.wrapping_count)
    
    return residue_wrapping


def compute_hdx_correlation(
    csv_content: str,
    sequence: str,
    dehydrons: List[Dehydron]
) -> HDXCorrelationResult:
    """
    Compute Spearman correlation between HDX-MS data and wrapping counts.
    
    Algorithm:
    1. Parse DynamX CSV format
    2. Map peptide ranges to residue IDs
    3. Calculate fractional deuterium uptake per residue
    4. Average overlapping peptide segments
    5. Map wrapping counts to residues
    6. Compute Spearman correlation
    7. Calculate R² and p-value
    8. Determine pass/fail (R² ≥ 0.50 and p < 0.05)
    
    Args:
        csv_content: HDX-MS data in DynamX CSV format
        sequence: Full protein amino acid sequence
        dehydrons: List of detected dehydrons with wrapping counts
    
    Returns:
        HDXCorrelationResult with correlation statistics
    
    Raises:
        HDXMSError: If correlation computation fails
    
    Validates: Requirements 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7
    """
    start_time = time.time()
    
    try:
        # Parse HDX-MS data
        peptides = parse_dynamx_csv(csv_content, sequence)
        
        # Map peptides to residues
        residue_uptakes = map_peptides_to_residues(peptides, len(sequence))
        
        # Average overlapping peptides
        averaged_uptakes = average_overlapping_peptides(residue_uptakes)
        
        # Map wrapping counts to residues
        residue_wrapping = map_wrapping_counts_to_residues(dehydrons)
        averaged_wrapping = {
            res_id: np.mean(counts)
            for res_id, counts in residue_wrapping.items()
        }
        
        # Find residues with both HDX and wrapping data
        common_residues = set(averaged_uptakes.keys()) & set(averaged_wrapping.keys())
        
        if len(common_residues) < 3:
            raise HDXMSError(
                f"Insufficient overlapping data: only {len(common_residues)} residues "
                "have both HDX and wrapping data (minimum 3 required)"
            )
        
        # Prepare data for correlation
        residue_data = []
        hdx_values = []
        wrapping_values = []
        
        for res_id in sorted(common_residues):
            uptake = averaged_uptakes[res_id]
            wrapping = averaged_wrapping[res_id]
            num_peptides = len(residue_uptakes[res_id])
            
            residue_data.append(ResidueHDXData(
                residue_id=res_id,
                fractional_uptake=uptake,
                wrapping_count=int(wrapping),
                num_peptides=num_peptides
            ))
            
            hdx_values.append(uptake)
            wrapping_values.append(wrapping)
        
        # Compute Spearman correlation
        # HDX uptake should be inversely correlated with wrapping
        # (more wrapping = less solvent exposure = less deuterium uptake)
        spearman_rho, p_value = spearmanr(wrapping_values, hdx_values)
        
        # Handle edge case: constant values result in NaN correlation
        # When all values are constant, there is no correlation to measure
        if np.isnan(spearman_rho) or np.isnan(p_value):
            raise HDXMSError(
                "Cannot compute correlation: input values are constant. "
                "All HDX uptake values or all wrapping counts are identical."
            )
        
        # Calculate R² (coefficient of determination)
        # For Spearman, R² = rho²
        r_squared = spearman_rho ** 2
        
        # Determine pass/fail
        passed = (r_squared >= 0.50) and (p_value < 0.05)
        
        computation_time = time.time() - start_time
        
        return HDXCorrelationResult(
            r_squared=float(r_squared),
            p_value=float(p_value),
            spearman_rho=float(spearman_rho),
            num_residues=len(common_residues),
            passed=passed,
            residue_data=residue_data,
            computation_time_sec=computation_time
        )
    
    except HDXMSError:
        raise
    except Exception as e:
        raise HDXMSError(f"HDX-MS correlation failed: {str(e)}")
