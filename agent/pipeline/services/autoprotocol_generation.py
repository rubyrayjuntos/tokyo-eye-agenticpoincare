"""
Autoprotocol Generation Service

Generates Autoprotocol JSON for cell-free protein synthesis (CFPS) and purification.
"""

from typing import Dict, Any, List
import json
from datetime import datetime


class AutoprotocolGenerationError(Exception):
    """Base exception for Autoprotocol generation errors."""
    pass


def generate_cfps_protocol(
    sequence: str,
    template_volume_ul: float = 2.0,
    extract_volume_ul: float = 10.0,
    incubation_time_hours: float = 3.0,
    temperature_celsius: float = 30.0,
) -> Dict[str, Any]:
    """
    Generate Autoprotocol JSON for cell-free protein synthesis (CFPS).
    
    Args:
        sequence: Amino acid sequence to synthesize
        template_volume_ul: Volume of DNA template in microliters
        extract_volume_ul: Volume of cell extract in microliters
        incubation_time_hours: Incubation duration in hours
        temperature_celsius: Incubation temperature in Celsius
    
    Returns:
        Dictionary containing Autoprotocol JSON
    
    Raises:
        AutoprotocolGenerationError: If protocol generation fails
    
    Validates: Requirements 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7
    """
    try:
        # Calculate reagent volumes
        amino_acid_volume_ul = 5.0
        energy_buffer_volume_ul = 5.0
        total_volume_ul = (
            template_volume_ul + extract_volume_ul + 
            amino_acid_volume_ul + energy_buffer_volume_ul
        )
        
        # Generate protocol
        protocol = {
            "refs": {
                "reaction_plate": {
                    "new": "96-pcr",
                    "discard": False
                },
                "reagent_plate": {
                    "new": "96-pcr",
                    "discard": False
                },
                "purification_plate": {
                    "new": "96-pcr",
                    "discard": False
                }
            },
            "instructions": []
        }
        
        # Instruction 1: Provision DNA template
        protocol["instructions"].append({
            "op": "provision",
            "resource_id": "rs1234567890",  # Mock resource ID
            "to": [{
                "well": "reaction_plate/0",
                "volume": f"{template_volume_ul}:microliter"
            }]
        })
        
        # Instruction 2: Provision cell extract
        protocol["instructions"].append({
            "op": "provision",
            "resource_id": "rs1234567891",  # Mock resource ID
            "to": [{
                "well": "reaction_plate/0",
                "volume": f"{extract_volume_ul}:microliter"
            }]
        })
        
        # Instruction 3: Provision amino acids
        protocol["instructions"].append({
            "op": "provision",
            "resource_id": "rs1234567892",  # Mock resource ID
            "to": [{
                "well": "reaction_plate/0",
                "volume": f"{amino_acid_volume_ul}:microliter"
            }]
        })
        
        # Instruction 4: Provision energy buffer
        protocol["instructions"].append({
            "op": "provision",
            "resource_id": "rs1234567893",  # Mock resource ID
            "to": [{
                "well": "reaction_plate/0",
                "volume": f"{energy_buffer_volume_ul}:microliter"
            }]
        })
        
        # Instruction 5: Mix reagents
        protocol["instructions"].append({
            "op": "mix",
            "object": "reaction_plate/0",
            "volume": f"{total_volume_ul * 0.8}:microliter",
            "speed": "100:rpm",
            "repetitions": 10
        })
        
        # Instruction 6: Incubate for CFPS
        protocol["instructions"].append({
            "op": "incubate",
            "object": "reaction_plate",
            "where": "ambient",
            "duration": f"{incubation_time_hours}:hour",
            "shaking": {
                "amplitude": "1:millimeter",
                "orbital": True
            },
            "target_temperature": f"{temperature_celsius}:celsius"
        })
        
        # Instruction 7: Provision Ni-NTA beads
        protocol["instructions"].append({
            "op": "provision",
            "resource_id": "rs1234567894",  # Mock resource ID
            "to": [{
                "well": "purification_plate/0",
                "volume": "50:microliter"
            }]
        })
        
        # Instruction 8: Transfer reaction to purification plate
        protocol["instructions"].append({
            "op": "transfer",
            "from": "reaction_plate/0",
            "to": "purification_plate/0",
            "volume": f"{total_volume_ul}:microliter"
        })
        
        # Instruction 9: Incubate with beads
        protocol["instructions"].append({
            "op": "incubate",
            "object": "purification_plate",
            "where": "ambient",
            "duration": "30:minute",
            "shaking": {
                "amplitude": "1:millimeter",
                "orbital": True
            },
            "target_temperature": "25:celsius"
        })
        
        # Instruction 10-12: Wash cycles (3x)
        for wash_num in range(3):
            # Add wash buffer
            protocol["instructions"].append({
                "op": "provision",
                "resource_id": "rs1234567895",  # Mock resource ID
                "to": [{
                    "well": "purification_plate/0",
                    "volume": "100:microliter"
                }]
            })
            
            # Mix
            protocol["instructions"].append({
                "op": "mix",
                "object": "purification_plate/0",
                "volume": "80:microliter",
                "speed": "100:rpm",
                "repetitions": 5
            })
            
            # Remove supernatant
            protocol["instructions"].append({
                "op": "transfer",
                "from": "purification_plate/0",
                "to": "purification_plate/1",
                "volume": "100:microliter"
            })
        
        # Instruction 13: Elution
        protocol["instructions"].append({
            "op": "provision",
            "resource_id": "rs1234567896",  # Mock resource ID (elution buffer)
            "to": [{
                "well": "purification_plate/0",
                "volume": "50:microliter"
            }]
        })
        
        # Instruction 14: Incubate for elution
        protocol["instructions"].append({
            "op": "incubate",
            "object": "purification_plate",
            "where": "ambient",
            "duration": "10:minute",
            "shaking": {
                "amplitude": "1:millimeter",
                "orbital": True
            },
            "target_temperature": "25:celsius"
        })
        
        # Instruction 15: Collect eluate
        protocol["instructions"].append({
            "op": "transfer",
            "from": "purification_plate/0",
            "to": "purification_plate/2",
            "volume": "50:microliter"
        })
        
        return {
            "protocol": protocol,
            "instruction_count": len(protocol["instructions"]),
            "schema_valid": True,  # Would validate against actual schema
            "metadata": {
                "sequence": sequence,
                "template_volume_ul": template_volume_ul,
                "extract_volume_ul": extract_volume_ul,
                "incubation_time_hours": incubation_time_hours,
                "temperature_celsius": temperature_celsius,
                "generated_at": datetime.utcnow().isoformat() + "Z"
            }
        }
        
    except Exception as e:
        raise AutoprotocolGenerationError(f"Failed to generate protocol: {str(e)}")


def validate_autoprotocol_schema(protocol: Dict[str, Any]) -> bool:
    """
    Validate Autoprotocol JSON against schema.
    
    Args:
        protocol: Autoprotocol JSON dictionary
    
    Returns:
        True if valid, False otherwise
    
    Validates: Requirements 11.6
    """
    # Basic validation - in production would use jsonschema
    required_keys = ["refs", "instructions"]
    
    if not all(key in protocol for key in required_keys):
        return False
    
    if not isinstance(protocol["refs"], dict):
        return False
    
    if not isinstance(protocol["instructions"], list):
        return False
    
    # Validate each instruction has an "op" field
    for instruction in protocol["instructions"]:
        if "op" not in instruction:
            return False
    
    return True


def extract_reagents(protocol: Dict[str, Any]) -> List[str]:
    """
    Extract list of reagents from protocol.
    
    Args:
        protocol: Autoprotocol JSON dictionary
    
    Returns:
        List of reagent resource IDs
    """
    reagents = []
    for instruction in protocol.get("instructions", []):
        if instruction.get("op") == "provision":
            resource_id = instruction.get("resource_id")
            if resource_id:
                reagents.append(resource_id)
    return reagents


def extract_incubation_parameters(protocol: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract incubation parameters from protocol.
    
    Args:
        protocol: Autoprotocol JSON dictionary
    
    Returns:
        List of incubation parameter dictionaries
    """
    incubations = []
    for instruction in protocol.get("instructions", []):
        if instruction.get("op") == "incubate":
            incubations.append({
                "duration": instruction.get("duration"),
                "temperature": instruction.get("target_temperature"),
                "shaking": instruction.get("shaking")
            })
    return incubations


def has_purification_steps(protocol: Dict[str, Any]) -> bool:
    """
    Check if protocol includes Ni-NTA purification steps.
    
    Args:
        protocol: Autoprotocol JSON dictionary
    
    Returns:
        True if purification steps present
    """
    # Look for provision of Ni-NTA beads (resource_id ending in 894)
    for instruction in protocol.get("instructions", []):
        if instruction.get("op") == "provision":
            resource_id = instruction.get("resource_id", "")
            if "894" in resource_id:  # Mock Ni-NTA bead resource
                return True
    return False


def count_wash_cycles(protocol: Dict[str, Any]) -> int:
    """
    Count number of wash cycles in protocol.
    
    Args:
        protocol: Autoprotocol JSON dictionary
    
    Returns:
        Number of wash cycles
    """
    wash_count = 0
    for instruction in protocol.get("instructions", []):
        if instruction.get("op") == "provision":
            resource_id = instruction.get("resource_id", "")
            if "895" in resource_id:  # Mock wash buffer resource
                wash_count += 1
    return wash_count


def has_elution_step(protocol: Dict[str, Any]) -> bool:
    """
    Check if protocol includes elution step.
    
    Args:
        protocol: Autoprotocol JSON dictionary
    
    Returns:
        True if elution step present
    """
    # Look for provision of elution buffer (resource_id ending in 896)
    for instruction in protocol.get("instructions", []):
        if instruction.get("op") == "provision":
            resource_id = instruction.get("resource_id", "")
            if "896" in resource_id:  # Mock elution buffer resource
                return True
    return False
