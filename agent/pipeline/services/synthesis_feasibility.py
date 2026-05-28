"""
Synthesis Feasibility Service.

This service handles sequence extraction, reverse translation, and integration
with Twist Bioscience API for manufacturability assessment.

Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.5, 10.6
"""

from typing import Dict, List, Optional
import logging
from gosp.models.data_models import (
    StructureData,
    SynthesisFeasibilityResult,
)

logger = logging.getLogger(__name__)


class SynthesisFeasibilityError(Exception):
    """Base exception for synthesis feasibility errors."""
    pass


class SequenceExtractionError(SynthesisFeasibilityError):
    """Raised when sequence extraction fails."""
    pass


class ReverseTranslationError(SynthesisFeasibilityError):
    """Raised when reverse translation fails."""
    pass


class TwistAPIError(SynthesisFeasibilityError):
    """Raised when Twist Bioscience API call fails."""
    pass


# Codon optimization table for E. coli (most frequent codons)
CODON_TABLE = {
    'A': ['GCG', 'GCC', 'GCA', 'GCT'],  # Alanine
    'C': ['TGC', 'TGT'],                 # Cysteine
    'D': ['GAT', 'GAC'],                 # Aspartic acid
    'E': ['GAA', 'GAG'],                 # Glutamic acid
    'F': ['TTT', 'TTC'],                 # Phenylalanine
    'G': ['GGC', 'GGT', 'GGA', 'GGG'],  # Glycine
    'H': ['CAT', 'CAC'],                 # Histidine
    'I': ['ATT', 'ATC', 'ATA'],         # Isoleucine
    'K': ['AAA', 'AAG'],                 # Lysine
    'L': ['CTG', 'TTG', 'CTA', 'CTT', 'CTC', 'TTA'],  # Leucine
    'M': ['ATG'],                        # Methionine (start)
    'N': ['AAT', 'AAC'],                 # Asparagine
    'P': ['CCG', 'CCA', 'CCT', 'CCC'],  # Proline
    'Q': ['CAG', 'CAA'],                 # Glutamine
    'R': ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],  # Arginine
    'S': ['AGC', 'TCT', 'TCC', 'TCA', 'TCG', 'AGT'],  # Serine
    'T': ['ACC', 'ACA', 'ACG', 'ACT'],  # Threonine
    'V': ['GTG', 'GTT', 'GTC', 'GTA'],  # Valine
    'W': ['TGG'],                        # Tryptophan
    'Y': ['TAT', 'TAC'],                 # Tyrosine
    '*': ['TAA', 'TAG', 'TGA'],         # Stop codons
}

# Reverse codon table for translation
DNA_TO_AA = {
    'GCG': 'A', 'GCC': 'A', 'GCA': 'A', 'GCT': 'A',
    'TGC': 'C', 'TGT': 'C',
    'GAT': 'D', 'GAC': 'D',
    'GAA': 'E', 'GAG': 'E',
    'TTT': 'F', 'TTC': 'F',
    'GGC': 'G', 'GGT': 'G', 'GGA': 'G', 'GGG': 'G',
    'CAT': 'H', 'CAC': 'H',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I',
    'AAA': 'K', 'AAG': 'K',
    'CTG': 'L', 'TTG': 'L', 'CTA': 'L', 'CTT': 'L', 'CTC': 'L', 'TTA': 'L',
    'ATG': 'M',
    'AAT': 'N', 'AAC': 'N',
    'CCG': 'P', 'CCA': 'P', 'CCT': 'P', 'CCC': 'P',
    'CAG': 'Q', 'CAA': 'Q',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R', 'AGA': 'R', 'AGG': 'R',
    'AGC': 'S', 'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S', 'AGT': 'S',
    'ACC': 'T', 'ACA': 'T', 'ACG': 'T', 'ACT': 'T',
    'GTG': 'V', 'GTT': 'V', 'GTC': 'V', 'GTA': 'V',
    'TGG': 'W',
    'TAT': 'Y', 'TAC': 'Y',
    'TAA': '*', 'TAG': '*', 'TGA': '*',
}


def extract_sequence_from_structure(structure: StructureData) -> str:
    """
    Extract amino acid sequence from structure data.
    
    Validates: Requirements 10.1
    
    Args:
        structure: StructureData object containing sequence
        
    Returns:
        Amino acid sequence string
        
    Raises:
        SequenceExtractionError: If sequence extraction fails
    """
    try:
        if not structure.sequence:
            raise SequenceExtractionError("Structure has no sequence data")
        
        # Validate sequence contains only valid amino acid codes
        valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
        if not all(aa in valid_aa for aa in structure.sequence):
            raise SequenceExtractionError(
                f"Sequence contains invalid amino acid codes: {structure.sequence}"
            )
        
        return structure.sequence
    except Exception as e:
        logger.error(f"Sequence extraction failed: {e}")
        raise SequenceExtractionError(f"Failed to extract sequence: {e}")


def reverse_translate_sequence(
    aa_sequence: str, 
    optimize_for: str = "ecoli"
) -> str:
    """
    Reverse-translate amino acid sequence to DNA with codon optimization.
    
    Validates: Requirements 10.2
    
    Args:
        aa_sequence: Amino acid sequence (single-letter codes)
        optimize_for: Organism for codon optimization (default: "ecoli")
        
    Returns:
        DNA sequence (optimized codons)
        
    Raises:
        ReverseTranslationError: If translation fails
    """
    try:
        if not aa_sequence:
            raise ReverseTranslationError("Empty amino acid sequence")
        
        # Validate amino acid sequence
        valid_aa = set('ACDEFGHIKLMNPQRSTVWY*')
        if not all(aa in valid_aa for aa in aa_sequence):
            invalid = set(aa_sequence) - valid_aa
            raise ReverseTranslationError(
                f"Invalid amino acid codes: {invalid}"
            )
        
        # Reverse translate using most frequent codons (first in list)
        dna_sequence = ""
        for aa in aa_sequence:
            if aa not in CODON_TABLE:
                raise ReverseTranslationError(f"No codon found for amino acid: {aa}")
            # Use first (most frequent) codon for E. coli optimization
            dna_sequence += CODON_TABLE[aa][0]
        
        return dna_sequence
    except Exception as e:
        logger.error(f"Reverse translation failed: {e}")
        raise ReverseTranslationError(f"Failed to reverse translate: {e}")


def translate_dna_to_protein(dna_sequence: str) -> str:
    """
    Translate DNA sequence to amino acid sequence.
    
    Used for round-trip validation.
    
    Args:
        dna_sequence: DNA sequence (codons)
        
    Returns:
        Amino acid sequence
        
    Raises:
        ReverseTranslationError: If translation fails
    """
    try:
        if len(dna_sequence) % 3 != 0:
            raise ReverseTranslationError(
                f"DNA sequence length must be multiple of 3, got {len(dna_sequence)}"
            )
        
        aa_sequence = ""
        for i in range(0, len(dna_sequence), 3):
            codon = dna_sequence[i:i+3]
            if codon not in DNA_TO_AA:
                raise ReverseTranslationError(f"Invalid codon: {codon}")
            aa_sequence += DNA_TO_AA[codon]
        
        return aa_sequence
    except Exception as e:
        logger.error(f"DNA translation failed: {e}")
        raise ReverseTranslationError(f"Failed to translate DNA: {e}")


def check_synthesis_feasibility(
    sequences: List[str],
    api_key: Optional[str] = None
) -> Dict[str, SynthesisFeasibilityResult]:
    """
    Check synthesis feasibility for amino acid sequences.
    
    Validates: Requirements 10.3, 10.4, 10.5, 10.6
    
    Args:
        sequences: List of amino acid sequences
        api_key: Optional Twist Bioscience API key
        
    Returns:
        Dictionary mapping sequence IDs to feasibility results
        
    Raises:
        TwistAPIError: If API call fails
    """
    results = {}
    
    for idx, aa_sequence in enumerate(sequences):
        seq_id = f"seq_{idx + 1}"
        
        try:
            # Reverse translate to DNA
            dna_sequence = reverse_translate_sequence(aa_sequence)
            
            # If API key provided, call Twist Bioscience API
            if api_key:
                # Mock API call for now - in production, this would call actual API
                # import requests
                # response = requests.post(
                #     "https://api.twistbioscience.com/v1/feasibility",
                #     headers={"Authorization": f"Bearer {api_key}"},
                #     json={"sequence": dna_sequence}
                # )
                # result_data = response.json()
                
                # For now, perform basic checks
                result = _perform_basic_feasibility_check(dna_sequence, aa_sequence)
            else:
                # Without API key, perform basic checks only
                result = _perform_basic_feasibility_check(dna_sequence, aa_sequence)
            
            results[seq_id] = result
            
        except (ReverseTranslationError, SequenceExtractionError) as e:
            logger.warning(f"Feasibility check failed for {seq_id}: {e}")
            results[seq_id] = SynthesisFeasibilityResult(
                manufacturable=False,
                complexity_score=None,
                issues=[str(e)],
                details={}
            )
    
    return results


def _perform_basic_feasibility_check(
    dna_sequence: str,
    aa_sequence: str
) -> SynthesisFeasibilityResult:
    """
    Perform basic feasibility checks without external API.
    
    Checks for:
    - Sequence length limits
    - GC content
    - Repeat regions
    - Homopolymer runs
    
    Args:
        dna_sequence: DNA sequence
        aa_sequence: Amino acid sequence
        
    Returns:
        SynthesisFeasibilityResult
    """
    issues = []
    
    # Check sequence length (typical synthesis limits)
    if len(dna_sequence) > 3000:
        issues.append(f"Sequence too long for standard synthesis: {len(dna_sequence)} bp")
    
    # Check GC content (optimal range: 40-60%)
    gc_count = dna_sequence.count('G') + dna_sequence.count('C')
    gc_content = gc_count / len(dna_sequence) if len(dna_sequence) > 0 else 0
    if gc_content < 0.3 or gc_content > 0.7:
        issues.append(f"GC content outside optimal range: {gc_content:.2%}")
    
    # Check for long homopolymer runs (>6 bases)
    for base in ['A', 'T', 'G', 'C']:
        if base * 7 in dna_sequence:
            issues.append(f"Long homopolymer run detected: {base}7+")
    
    # Check for repeat regions (simple check for 10bp repeats)
    for i in range(len(dna_sequence) - 20):
        motif = dna_sequence[i:i+10]
        if dna_sequence.count(motif) > 3:
            issues.append(f"Repeat region detected: {motif}")
            break
    
    # Calculate complexity score (simple metric based on issues)
    complexity_score = len(issues) * 0.5 + (abs(gc_content - 0.5) * 10)
    
    manufacturable = len(issues) == 0
    
    return SynthesisFeasibilityResult(
        manufacturable=manufacturable,
        complexity_score=complexity_score,
        issues=issues,
        details={
            "gc_content": gc_content,
            "sequence_length": len(dna_sequence),
            "aa_length": len(aa_sequence)
        }
    )
