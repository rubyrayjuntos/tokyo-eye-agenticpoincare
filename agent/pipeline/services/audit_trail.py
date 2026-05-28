"""
Audit Trail Service

Provides cryptographic audit trail logging with SHA-256 hash chaining
and CDISC Dataset-JSON serialization for regulatory compliance.

Validates: Requirements 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7
"""

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from gosp.models.data_models import AuditEvent


class AuditTrailError(Exception):
    """Base exception for audit trail errors."""
    pass


class AuditTrailService:
    """
    Service for managing cryptographic audit trails.
    
    Features:
    - SHA-256 hash chaining for integrity verification
    - CDISC Dataset-JSON format serialization
    - Cryptographic proof of data integrity
    - Regulatory compliance (21 CFR Part 11)
    """
    
    def __init__(self):
        """Initialize audit trail service with empty history."""
        self.history: List[AuditEvent] = []
        self._last_hash: Optional[str] = None
    
    def log_event(
        self,
        state_from: str,
        state_to: str,
        event: str,
        calc_values: Dict[str, Any],
        user_id: str
    ) -> AuditEvent:
        """
        Log a state transition event with cryptographic hash chaining.
        
        Args:
            state_from: Source state name
            state_to: Destination state name
            event: Event description
            calc_values: Dictionary of calculated values (ΔG, wrapping counts, etc.)
            user_id: User identifier
        
        Returns:
            AuditEvent with hash chain
        
        Validates: Requirements 12.1, 12.2, 12.3
        """
        # Generate event ID
        event_id = f"evt_{len(self.history) + 1:06d}"
        
        # Get current timestamp in ISO 8601 format
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Get previous hash (or empty string for first event)
        prev_hash = self._last_hash if self._last_hash else ""
        
        # Create event data for hashing
        event_data = {
            "event_id": event_id,
            "timestamp": timestamp,
            "state_from": state_from,
            "state_to": state_to,
            "event": event,
            "calc_values": calc_values,
            "user_id": user_id,
            "prev_hash": prev_hash
        }
        
        # Compute SHA-256 hash
        hash_input = json.dumps(event_data, sort_keys=True).encode('utf-8')
        current_hash = hashlib.sha256(hash_input).hexdigest()
        
        # Create audit event
        audit_event = AuditEvent(
            event_id=event_id,
            timestamp=timestamp,
            state_from=state_from,
            state_to=state_to,
            event=event,
            calc_values=calc_values,
            user_id=user_id,
            hash=current_hash,
            prev_hash=prev_hash
        )
        
        # Add to history
        self.history.append(audit_event)
        self._last_hash = current_hash
        
        return audit_event
    
    def verify_integrity(self) -> bool:
        """
        Verify cryptographic integrity of the audit trail.
        
        Returns:
            True if hash chain is valid, False otherwise
        
        Validates: Requirements 12.3, 12.6
        """
        if not self.history:
            return True
        
        # Verify first event has empty prev_hash
        if self.history[0].prev_hash != "":
            return False
        
        # Verify hash chain
        for i, event in enumerate(self.history):
            # Reconstruct event data
            event_data = {
                "event_id": event.event_id,
                "timestamp": event.timestamp,
                "state_from": event.state_from,
                "state_to": event.state_to,
                "event": event.event,
                "calc_values": event.calc_values,
                "user_id": event.user_id,
                "prev_hash": event.prev_hash
            }
            
            # Recompute hash
            hash_input = json.dumps(event_data, sort_keys=True).encode('utf-8')
            computed_hash = hashlib.sha256(hash_input).hexdigest()
            
            # Verify hash matches
            if computed_hash != event.hash:
                return False
            
            # Verify prev_hash chain (except for first event)
            if i > 0:
                if event.prev_hash != self.history[i - 1].hash:
                    return False
        
        return True
    
    def export_dataset_json(self, define_xml_ref: str = "define.xml") -> Dict[str, Any]:
        """
        Export audit trail to CDISC Dataset-JSON format.
        
        Args:
            define_xml_ref: Reference to define.xml schema file
        
        Returns:
            Dictionary in CDISC Dataset-JSON format
        
        Validates: Requirements 12.4, 12.5
        """
        # Build Dataset-JSON structure
        dataset_json = {
            "clinicalData": {
                "studyOID": "GOSP.AUDIT.001",
                "metaDataVersionOID": "MDV.GOSP.001",
                "itemGroupData": {
                    "IG.AUDIT": {
                        "records": len(self.history),
                        "name": "Audit Trail",
                        "label": "GOSP Molecular Imager Audit Trail",
                        "items": [
                            {
                                "OID": "IT.EVENT_ID",
                                "name": "EVENT_ID",
                                "label": "Event Identifier",
                                "type": "string"
                            },
                            {
                                "OID": "IT.TIMESTAMP",
                                "name": "TIMESTAMP",
                                "label": "Event Timestamp",
                                "type": "datetime"
                            },
                            {
                                "OID": "IT.STATE_FROM",
                                "name": "STATE_FROM",
                                "label": "Source State",
                                "type": "string"
                            },
                            {
                                "OID": "IT.STATE_TO",
                                "name": "STATE_TO",
                                "label": "Destination State",
                                "type": "string"
                            },
                            {
                                "OID": "IT.EVENT",
                                "name": "EVENT",
                                "label": "Event Description",
                                "type": "string"
                            },
                            {
                                "OID": "IT.CALC_VALUES",
                                "name": "CALC_VALUES",
                                "label": "Calculated Values",
                                "type": "string"
                            },
                            {
                                "OID": "IT.USER_ID",
                                "name": "USER_ID",
                                "label": "User Identifier",
                                "type": "string"
                            },
                            {
                                "OID": "IT.HASH",
                                "name": "HASH",
                                "label": "SHA-256 Hash",
                                "type": "string"
                            },
                            {
                                "OID": "IT.PREV_HASH",
                                "name": "PREV_HASH",
                                "label": "Previous Hash",
                                "type": "string"
                            }
                        ],
                        "itemData": []
                    }
                }
            },
            "define_xml_ref": define_xml_ref
        }
        
        # Add event data
        for event in self.history:
            item_data = [
                {"itemOID": "IT.EVENT_ID", "value": event.event_id},
                {"itemOID": "IT.TIMESTAMP", "value": event.timestamp},
                {"itemOID": "IT.STATE_FROM", "value": event.state_from},
                {"itemOID": "IT.STATE_TO", "value": event.state_to},
                {"itemOID": "IT.EVENT", "value": event.event},
                {"itemOID": "IT.CALC_VALUES", "value": json.dumps(event.calc_values)},
                {"itemOID": "IT.USER_ID", "value": event.user_id},
                {"itemOID": "IT.HASH", "value": event.hash},
                {"itemOID": "IT.PREV_HASH", "value": event.prev_hash}
            ]
            dataset_json["clinicalData"]["itemGroupData"]["IG.AUDIT"]["itemData"].append(item_data)
        
        return dataset_json
    
    def get_history(self) -> List[AuditEvent]:
        """
        Get complete audit trail history.
        
        Returns:
            List of all audit events
        """
        return self.history.copy()
    
    def clear_history(self):
        """Clear audit trail history (for testing only)."""
        self.history.clear()
        self._last_hash = None


# Global audit trail service instance
_audit_service = AuditTrailService()


def get_audit_service() -> AuditTrailService:
    """Get the global audit trail service instance."""
    return _audit_service


def log_event(
    state_from: str,
    state_to: str,
    event: str,
    calc_values: Dict[str, Any],
    user_id: str
) -> AuditEvent:
    """
    Convenience function to log an audit event.
    
    Args:
        state_from: Source state name
        state_to: Destination state name
        event: Event description
        calc_values: Dictionary of calculated values
        user_id: User identifier
    
    Returns:
        AuditEvent with hash chain
    """
    service = get_audit_service()
    return service.log_event(state_from, state_to, event, calc_values, user_id)


def verify_integrity() -> bool:
    """
    Verify cryptographic integrity of the audit trail.
    
    Returns:
        True if hash chain is valid, False otherwise
    """
    service = get_audit_service()
    return service.verify_integrity()


def export_dataset_json(define_xml_ref: str = "define.xml") -> Dict[str, Any]:
    """
    Export audit trail to CDISC Dataset-JSON format.
    
    Args:
        define_xml_ref: Reference to define.xml schema file
    
    Returns:
        Dictionary in CDISC Dataset-JSON format
    """
    service = get_audit_service()
    return service.export_dataset_json(define_xml_ref)


def get_history() -> List[AuditEvent]:
    """
    Get complete audit trail history.
    
    Returns:
        List of all audit events
    """
    service = get_audit_service()
    return service.get_history()
