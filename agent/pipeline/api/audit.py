"""
Audit Trail API Endpoints

Provides REST API for cryptographic audit trail logging and verification.

Validates: Requirements 12.1-12.7
"""

from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field, ConfigDict

from gosp.models.data_models import AuditEvent
from gosp.services.audit_trail import (
    log_event,
    verify_integrity,
    export_dataset_json,
    get_history,
    get_audit_service,
    AuditTrailError
)


router = APIRouter(prefix="/api/audit", tags=["audit"])


class LogEventRequest(BaseModel):
    """Request model for logging an audit event."""
    state_from: str = Field(..., description="Source state name")
    state_to: str = Field(..., description="Destination state name")
    event: str = Field(..., description="Event description")
    calc_values: Dict[str, Any] = Field(
        ...,
        description="Dictionary of calculated values (ΔG, wrapping counts, etc.)"
    )
    user_id: str = Field(..., description="User identifier")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "state_from": "ALIGN",
                "state_to": "LOCK",
                "event": "energy_validation",
                "calc_values": {
                    "delta_g": -18.2,
                    "sasa": 2500.0,
                    "potential_energy": -1850.3
                },
                "user_id": "user@example.com"
            }
        }
    )


class LogEventResponse(BaseModel):
    """Response model for logging an audit event."""
    event_id: str = Field(..., description="Unique event identifier")
    hash: str = Field(..., description="SHA-256 hash of this event")
    prev_hash: str = Field(..., description="SHA-256 hash of previous event")
    timestamp: str = Field(..., description="Event timestamp (ISO 8601)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "event_id": "evt_000001",
                "hash": "a3f2b8c9d1e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0",
                "prev_hash": "b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3",
                "timestamp": "2026-02-13T14:23:01Z"
            }
        }
    )


@router.post("/log-event", response_model=LogEventResponse)
async def log_event_endpoint(
    request: LogEventRequest = Body(...)
) -> LogEventResponse:
    """
    Log a state transition event with cryptographic hash chaining.
    
    This endpoint:
    - Creates a new audit event with timestamp and user ID
    - Computes SHA-256 hash chained to previous event
    - Stores event in audit trail history
    - Returns event ID and hash for verification
    
    The hash chain ensures:
    - Non-repudiation: Events cannot be modified without detection
    - Integrity: Any tampering breaks the hash chain
    - Compliance: Meets 21 CFR Part 11 requirements
    
    Args:
        request: LogEventRequest with state transition and calculated values
    
    Returns:
        LogEventResponse with event ID and cryptographic hashes
    
    Raises:
        HTTPException: If event logging fails
    
    Validates: Requirements 12.1, 12.2, 12.3
    """
    try:
        # Log the event
        audit_event = log_event(
            state_from=request.state_from,
            state_to=request.state_to,
            event=request.event,
            calc_values=request.calc_values,
            user_id=request.user_id
        )
        
        return LogEventResponse(
            event_id=audit_event.event_id,
            hash=audit_event.hash,
            prev_hash=audit_event.prev_hash,
            timestamp=audit_event.timestamp
        )
    
    except AuditTrailError as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": True,
                "error_type": "AuditTrailError",
                "message": str(e),
                "details": {},
                "suggested_action": "Check event data and try again"
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


class VerifyIntegrityResponse(BaseModel):
    """Response model for integrity verification."""
    valid: bool = Field(..., description="True if hash chain is valid")
    event_count: int = Field(..., ge=0, description="Number of events in audit trail")
    message: str = Field(..., description="Verification result message")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "valid": True,
                "event_count": 42,
                "message": "Audit trail integrity verified successfully"
            }
        }
    )


@router.get("/verify-integrity", response_model=VerifyIntegrityResponse)
async def verify_integrity_endpoint() -> VerifyIntegrityResponse:
    """
    Verify cryptographic integrity of the audit trail.
    
    This endpoint:
    - Recomputes SHA-256 hashes for all events
    - Verifies hash chain continuity
    - Detects any tampering or corruption
    
    Returns:
        VerifyIntegrityResponse with validation result
    
    Validates: Requirements 12.3, 12.6
    """
    try:
        is_valid = verify_integrity()
        service = get_audit_service()
        event_count = len(service.history)
        
        if is_valid:
            message = "Audit trail integrity verified successfully"
        else:
            message = "Audit trail integrity verification FAILED - hash chain broken"
        
        return VerifyIntegrityResponse(
            valid=is_valid,
            event_count=event_count,
            message=message
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


class ExportDatasetJsonRequest(BaseModel):
    """Request model for Dataset-JSON export."""
    define_xml_ref: str = Field(
        "define.xml",
        description="Reference to define.xml schema file"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "define_xml_ref": "define.xml"
            }
        }
    )


class ExportDatasetJsonResponse(BaseModel):
    """Response model for Dataset-JSON export."""
    dataset_json: Dict[str, Any] = Field(..., description="CDISC Dataset-JSON format")
    event_count: int = Field(..., ge=0, description="Number of events exported")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "dataset_json": {
                    "clinicalData": {
                        "studyOID": "GOSP.AUDIT.001",
                        "metaDataVersionOID": "MDV.GOSP.001",
                        "itemGroupData": {}
                    },
                    "define_xml_ref": "define.xml"
                },
                "event_count": 42
            }
        }
    )


@router.post("/export-dataset-json", response_model=ExportDatasetJsonResponse)
async def export_dataset_json_endpoint(
    request: ExportDatasetJsonRequest = Body(...)
) -> ExportDatasetJsonResponse:
    """
    Export audit trail to CDISC Dataset-JSON format.
    
    This endpoint:
    - Serializes audit trail to CDISC Dataset-JSON format
    - References define.xml schema for regulatory compliance
    - Includes all events with cryptographic hashes
    - Suitable for FDA 21 CFR Part 11 submissions
    
    Args:
        request: ExportDatasetJsonRequest with define.xml reference
    
    Returns:
        ExportDatasetJsonResponse with Dataset-JSON structure
    
    Validates: Requirements 12.4, 12.5, 12.7
    """
    try:
        dataset_json = export_dataset_json(define_xml_ref=request.define_xml_ref)
        service = get_audit_service()
        event_count = len(service.history)
        
        return ExportDatasetJsonResponse(
            dataset_json=dataset_json,
            event_count=event_count
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


class GetHistoryResponse(BaseModel):
    """Response model for audit trail history."""
    history: List[AuditEvent] = Field(..., description="Complete audit trail history")
    event_count: int = Field(..., ge=0, description="Number of events in history")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "history": [
                    {
                        "event_id": "evt_000001",
                        "timestamp": "2026-02-13T14:23:01Z",
                        "state_from": "ALIGN",
                        "state_to": "LOCK",
                        "event": "energy_validation",
                        "calc_values": {"delta_g": -18.2},
                        "user_id": "user@example.com",
                        "hash": "a3f2b8c9...",
                        "prev_hash": ""
                    }
                ],
                "event_count": 1
            }
        }
    )


@router.get("/history", response_model=GetHistoryResponse)
async def get_history_endpoint() -> GetHistoryResponse:
    """
    Get complete audit trail history.
    
    Returns:
        GetHistoryResponse with all audit events
    
    Validates: Requirements 12.1, 12.2
    """
    try:
        history = get_history()
        
        return GetHistoryResponse(
            history=history,
            event_count=len(history)
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
