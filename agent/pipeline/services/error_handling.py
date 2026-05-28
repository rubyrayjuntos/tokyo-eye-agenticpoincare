"""
Centralized error handling utilities for GOSP backend services.

This module provides:
- Retry logic with exponential backoff
- Timeout management
- Graceful degradation for external API failures
- Structured error response format

Validates: Requirements 15.1, 15.2, 15.3, 15.4, 15.5, 15.6, 15.7
"""

import time
import logging
from typing import Any, Callable, Optional, TypeVar, Dict
from functools import wraps
from datetime import datetime

logger = logging.getLogger(__name__)

T = TypeVar('T')


class GOSPError(Exception):
    """Base exception for all GOSP errors."""
    
    def __init__(
        self,
        message: str,
        error_type: str,
        details: Optional[Dict[str, Any]] = None,
        suggested_action: Optional[str] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.details = details or {}
        self.suggested_action = suggested_action
        self.timestamp = datetime.utcnow().isoformat() + "Z"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to structured response format."""
        return {
            "error": True,
            "error_type": self.error_type,
            "message": self.message,
            "details": self.details,
            "suggested_action": self.suggested_action,
            "timestamp": self.timestamp
        }


class StructuralError(GOSPError):
    """Raised for structural validation errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, suggested_action: Optional[str] = None):
        super().__init__(message, "StructuralError", details, suggested_action)


class ComputationError(GOSPError):
    """Raised for computation failures."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, suggested_action: Optional[str] = None):
        super().__init__(message, "ComputationError", details, suggested_action)


class ExternalServiceError(GOSPError):
    """Raised for external service failures."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, suggested_action: Optional[str] = None):
        super().__init__(message, "ExternalServiceError", details, suggested_action)


class TimeoutError(GOSPError):
    """Raised when operations exceed timeout."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, suggested_action: Optional[str] = None):
        super().__init__(message, "TimeoutError", details, suggested_action)


def retry_with_exponential_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        initial_delay: Initial delay in seconds (default: 1.0)
        max_delay: Maximum delay in seconds (default: 60.0)
        exponential_base: Base for exponential backoff (default: 2.0)
        exceptions: Tuple of exceptions to catch and retry (default: all exceptions)
    
    Returns:
        Decorated function with retry logic
    
    Validates: Requirements 15.5
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(
                            f"Function {func.__name__} failed after {max_retries} retries: {str(e)}"
                        )
                        raise
                    
                    logger.warning(
                        f"Function {func.__name__} failed (attempt {attempt + 1}/{max_retries}): {str(e)}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    
                    time.sleep(delay)
                    delay = min(delay * exponential_base, max_delay)
            
            # This should never be reached, but just in case
            if last_exception:
                raise last_exception
            
        return wrapper
    return decorator


def with_timeout(timeout_seconds: float):
    """
    Decorator for enforcing timeout on functions.
    
    Note: This is a simple timeout check that should be used in conjunction
    with internal timeout logic in long-running functions.
    
    Args:
        timeout_seconds: Maximum execution time in seconds
    
    Returns:
        Decorated function with timeout enforcement
    
    Validates: Requirements 15.6
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            start_time = time.time()
            
            # Store start time in kwargs for function to check
            kwargs['_timeout_start'] = start_time
            kwargs['_timeout_seconds'] = timeout_seconds
            
            result = func(*args, **kwargs)
            
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                logger.warning(
                    f"Function {func.__name__} exceeded timeout "
                    f"({elapsed:.2f}s > {timeout_seconds}s)"
                )
            
            return result
        
        return wrapper
    return decorator


def graceful_external_api_call(
    func: Callable[..., T],
    fallback_value: Optional[T] = None,
    log_warning: bool = True
) -> Callable[..., Optional[T]]:
    """
    Wrapper for external API calls with graceful degradation.
    
    If the API call fails, logs a warning and returns the fallback value
    instead of crashing the entire system.
    
    Args:
        func: Function making the external API call
        fallback_value: Value to return on failure (default: None)
        log_warning: Whether to log warnings on failure (default: True)
    
    Returns:
        Wrapped function with graceful error handling
    
    Validates: Requirements 15.3
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Optional[T]:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if log_warning:
                logger.warning(
                    f"External API call {func.__name__} failed: {str(e)}. "
                    f"Continuing with degraded functionality."
                )
            return fallback_value
    
    return wrapper


def validate_structure_size(num_residues: int, max_residues: int = 1000) -> None:
    """
    Validate that structure size is within acceptable limits.
    
    Args:
        num_residues: Number of residues in the structure
        max_residues: Maximum allowed residues (default: 1000)
    
    Raises:
        StructuralError: If structure exceeds size limit
    
    Validates: Requirements 15.1
    """
    if num_residues > max_residues:
        raise StructuralError(
            message="Structure exceeds single-domain scope",
            details={
                "residue_count": num_residues,
                "max_allowed": max_residues
            },
            suggested_action="Consider analyzing individual domains separately"
        )


def validate_secondary_structure(
    secondary_structure_fraction: float,
    min_fraction: float = 0.20
) -> Optional[str]:
    """
    Validate that structure has sufficient secondary structure.
    
    Args:
        secondary_structure_fraction: Fraction of residues in secondary structure
        min_fraction: Minimum required fraction (default: 0.20)
    
    Returns:
        Warning message if below threshold, None otherwise
    
    Validates: Requirements 15.2
    """
    if secondary_structure_fraction < min_fraction:
        warning = (
            f"Insufficient secondary structure - dehydron detection unreliable "
            f"(found {secondary_structure_fraction:.1%}, expected ≥{min_fraction:.1%})"
        )
        logger.warning(warning)
        return warning
    return None


def detect_out_of_scope_structure(
    sequence: str,
    secondary_structure: Optional[list] = None
) -> Optional[str]:
    """
    Detect if structure is out of scope (DNA/RNA or disordered protein).
    
    Args:
        sequence: Amino acid or nucleotide sequence
        secondary_structure: Optional list of secondary structure codes
    
    Returns:
        Error message if out of scope, None otherwise
    
    Validates: Requirements 14.7, 15.1
    """
    # Check for DNA/RNA nucleotides
    # Common nucleotide codes in PDB files
    dna_nucleotides = {'DA', 'DT', 'DC', 'DG'}  # DNA
    rna_nucleotides = {'A', 'U', 'C', 'G'}  # RNA
    nucleotides = dna_nucleotides | rna_nucleotides
    
    # Check if sequence contains nucleotide codes
    # For single-letter codes, check if it's mostly ATCGU (not protein)
    if len(sequence) > 0:
        nucleotide_chars = set('ATCGU')
        protein_only_chars = set('DEFHIKLMNPQRSVWY')  # Amino acids not in nucleotides
        
        # Count characters
        nucleotide_count = sum(1 for c in sequence.upper() if c in nucleotide_chars)
        protein_only_count = sum(1 for c in sequence.upper() if c in protein_only_chars)
        
        # If >80% nucleotide characters and no protein-only characters, likely DNA/RNA
        if nucleotide_count > 0.8 * len(sequence) and protein_only_count == 0:
            return "DNA/RNA structure detected - out of scope"
    
    # Check for disordered protein (low secondary structure)
    if secondary_structure is not None and len(secondary_structure) > 0:
        # Count structured residues (H=helix, E=strand, B=beta-bridge)
        structured_codes = {'H', 'E', 'B'}
        structured_count = sum(1 for code in secondary_structure if code in structured_codes)
        structure_fraction = structured_count / len(secondary_structure)
        
        # If <20% structured, likely disordered
        if structure_fraction < 0.20:
            return f"Intrinsically disordered protein detected ({structure_fraction:.1%} structured) - out of scope"
    
    return None


def create_error_response(
    error: Exception,
    include_traceback: bool = False
) -> Dict[str, Any]:
    """
    Create a structured error response from an exception.
    
    Args:
        error: The exception to convert
        include_traceback: Whether to include traceback (default: False)
    
    Returns:
        Structured error response dictionary
    
    Validates: Requirements 15.4
    """
    if isinstance(error, GOSPError):
        return error.to_dict()
    
    # For non-GOSP errors, create a generic response
    response = {
        "error": True,
        "error_type": type(error).__name__,
        "message": str(error),
        "details": {},
        "suggested_action": "Check logs for more details",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    
    if include_traceback:
        import traceback
        response["traceback"] = traceback.format_exc()
    
    return response


def check_timeout(start_time: float, timeout_seconds: float, operation_name: str) -> None:
    """
    Check if operation has exceeded timeout.
    
    Args:
        start_time: Operation start time (from time.time())
        timeout_seconds: Maximum allowed time in seconds
        operation_name: Name of the operation for error message
    
    Raises:
        TimeoutError: If operation has exceeded timeout
    
    Validates: Requirements 15.6
    """
    elapsed = time.time() - start_time
    if elapsed > timeout_seconds:
        raise TimeoutError(
            message=f"{operation_name} exceeded timeout of {timeout_seconds}s",
            details={
                "elapsed_seconds": elapsed,
                "timeout_seconds": timeout_seconds,
                "operation": operation_name
            },
            suggested_action="Increase timeout or optimize the operation"
        )
