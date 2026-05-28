"""
Core data models for GOSP Molecular Imager.

These Pydantic models provide type validation and serialization/deserialization
for all data structures used in the system.
"""

from typing import List, Optional, Dict, Any, Literal, Tuple
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict, field_validator
from datetime import datetime


class Atom(BaseModel):
    """Represents a single atom in a protein structure."""
    atom_id: int = Field(..., description="Atom serial number")
    atom_name: str = Field(..., description="Atom name (e.g., 'CA', 'N', 'O')")
    residue_name: str = Field(..., description="Name of the residue the atom belongs to (e.g., 'ALA', 'LYS')")
    chain_id: str = Field(..., description="Chain identifier for the atom")
    residue_id: int = Field(..., description="Residue sequence number")
    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")
    z: float = Field(..., description="Z coordinate")
    occupancy: float = Field(..., description="Occupancy value")
    b_factor: float = Field(..., description="B-factor (temperature factor)")
    element: str = Field(..., description="Element symbol (e.g., 'C', 'N', 'O', 'S')")

class Residue(BaseModel):
    """Represents a single amino acid residue, containing multiple atoms."""
    residue_id: int = Field(..., description="Residue sequence number")
    residue_name: str = Field(..., description="Three-letter code for the residue (e.g., 'ALA', 'LYS')")
    atoms: List[Atom] = Field(..., description="List of atoms in this residue")

class Chain(BaseModel):
    """Represents a single polypeptide chain, containing multiple residues."""
    chain_id: str = Field(..., description="Chain identifier (e.g., 'A', 'B')")
    residues: List[Residue] = Field(..., description="List of residues in this chain")
    sequence: str = Field(..., description="Full amino acid sequence for this chain")

class StructureData(BaseModel):
    """
    Represents a complete protein structure with full atomistic detail and metadata.
    This model is a faithful, non-lossy representation of the source structure file.
    
    Validates: Requirements 3.3, 3.5
    """
    pdb_id: str = Field(..., description="PDB identifier")
    num_models: int = Field(..., ge=1, description="Number of models in source file (Note: only model 1 is stored)")
    resolution: Optional[float] = Field(None, ge=0, description="Resolution in Angstroms")
    
    chains: List[Chain] = Field(..., description="List of chains in the structure")
    
    # Metadata derived from the detailed structure
    total_residues: int = Field(..., ge=1, description="Total number of residues across all chains")
    total_atoms: int = Field(..., ge=1, description="Total number of atoms across all chains")
    secondary_structure: Dict[str, List[str]] = Field(
        ...,
        description="Secondary structure annotations per chain [chain_id][residue_index]"
    )
    missing_residues: Optional[Dict[str, List[int]]] = Field(
        None,
        description="Dictionary of missing residue IDs per chain"
    )


    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "pdb_id": "1L2Y",
                "num_models": 1,
                "resolution": 1.2,
                "chains": [
                    {
                        "chain_id": "A",
                        "sequence": "NLYIQWLKDGGPSSGRPPPS",
                        "residues": [
                            {
                                "residue_id": 1,
                                "residue_name": "ASN",
                                "atoms": [
                                    {
                                        "atom_id": 1, "atom_name": "N", "residue_name": "ASN", "chain_id": "A", 
                                        "residue_id": 1, "x": 10.0, "y": 10.0, "z": 10.0, 
                                        "occupancy": 1.0, "b_factor": 20.5, "element": "N"
                                    },
                                    {
                                        "atom_id": 2, "atom_name": "CA", "residue_name": "ASN", "chain_id": "A", 
                                        "residue_id": 1, "x": 11.0, "y": 10.0, "z": 10.0, 
                                        "occupancy": 1.0, "b_factor": 21.5, "element": "C"
                                    }
                                ]
                            }
                        ]
                    }
                ],
                "total_residues": 20,
                "total_atoms": 304,
                "secondary_structure": {"A": ["C", "C", "C", "H", "H", "H", "H", "C"]},
                "missing_residues": {"A": [21, 22]}
            }
        }
    )


class ResidueRange(BaseModel):
    """Represents a contiguous residue range (inclusive)."""
    start: int = Field(..., ge=1, description="Start residue index")
    end: int = Field(..., ge=1, description="End residue index")


class CddAccession(BaseModel):
    """CDD accession metadata."""
    id: str = Field(..., description="CDD accession identifier")
    url: str = Field(..., description="CDD accession URL")


class CddHierarchy(BaseModel):
    """CDD hierarchy metadata."""
    root: str = Field(..., description="CDD root cluster")
    superfamily: Optional[str] = Field(None, description="CDD superfamily cluster")
    family: Optional[str] = Field(None, description="CDD family identifier")
    lineage: List[str] = Field(default_factory=list, description="CDD lineage path")


class CddResidueConfidence(BaseModel):
    """Residue-level confidence score for a CDD domain."""
    pos: int = Field(..., ge=1, description="Residue position")
    score: float = Field(..., ge=0.0, description="Confidence score")


class CddDomainUi(BaseModel):
    """UI hints for CDD domain rendering."""
    color: str = Field(..., description="Hex color for the domain")
    label: Optional[str] = Field(None, description="Short domain label")


class CddDomain(BaseModel):
    """CDD domain definition for a chain."""
    cdd_id: str = Field(..., description="CDD domain identifier")
    name: str = Field(..., description="CDD domain name")
    short_name: Optional[str] = Field(None, description="Short domain name")
    range: ResidueRange = Field(..., description="Residue range for the domain")
    sequence: Optional[str] = Field(None, description="Domain sequence segment")
    evalue: Optional[float] = Field(None, description="CDD domain E-value")
    bit_score: Optional[float] = Field(None, description="CDD domain bit score")
    confidence: Optional[str] = Field(None, description="Confidence category")
    accession: Optional[CddAccession] = Field(None, description="CDD accession metadata")
    hierarchy: Optional[CddHierarchy] = Field(None, description="CDD hierarchy metadata")
    residue_confidence: List[CddResidueConfidence] = Field(
        default_factory=list,
        description="Residue-level confidence scores"
    )
    ui: Optional[CddDomainUi] = Field(None, description="UI hints for rendering")


class CddHingePrior(BaseModel):
    """Hinge prior parameters for linkers."""
    axis: List[float] = Field(..., description="Preferred hinge axis vector")
    stiffness: float = Field(..., ge=0.0, description="Hinge stiffness")
    max_extension_factor: Optional[float] = Field(
        None,
        ge=0.0,
        description="Maximum extension factor for linker"
    )


class LinkerType(str, Enum):
    """Valid linker types for domain connections.
    
    - PEG: Polyethylene glycol linker (unit length 3.5Å)
    - FLEXIBLE_ALKYL: Flexible alkyl chain (unit length 3.8Å)
    - RIGID_PROLINE: Rigid proline-based linker (unit length 3.0Å)
    """
    PEG = "peg"
    FLEXIBLE_ALKYL = "flexible_alkyl"
    RIGID_PROLINE = "rigid_proline"


class CddLinker(BaseModel):
    """Linker between two CDD domains."""
    id: str = Field(..., description="Linker identifier")
    from_domain: Optional[str] = Field(None, description="Source domain CDD ID")
    to_domain: Optional[str] = Field(None, description="Target domain CDD ID")
    range: ResidueRange = Field(..., description="Residue range for the linker")
    sequence: Optional[str] = Field(None, description="Linker sequence segment")
    type: Optional[LinkerType] = Field(None, description="Linker type (peg, flexible_alkyl, or rigid_proline)")
    hinge_prior: Optional[CddHingePrior] = Field(
        None,
        description="Linker hinge prior parameters"
    )


class CddInterfaceUi(BaseModel):
    """UI hints for interfaces."""
    highlight_color: Optional[str] = Field(None, description="Highlight color")


class CddInterface(BaseModel):
    """CDD interface definition."""
    id: str = Field(..., description="Interface identifier")
    domains: List[str] = Field(..., description="CDD domain pair")
    residue_pairs: List[List[int]] = Field(
        default_factory=list,
        description="Residue pairs at the interface"
    )
    distance_average_A: Optional[float] = Field(
        None,
        ge=0.0,
        description="Average interface distance in Angstroms"
    )
    interaction_types: List[str] = Field(
        default_factory=list,
        description="Interaction types at the interface"
    )
    dehydron_count: Optional[int] = Field(
        None,
        ge=0,
        description="Number of dehydrons at the interface"
    )
    glue_priority: Optional[str] = Field(None, description="Glue priority")
    ui: Optional[CddInterfaceUi] = Field(None, description="UI hints")


class RigidBodySphere(BaseModel):
    """Sphere approximation for a rigid body domain."""
    center: List[float] = Field(..., description="Sphere center coordinates")
    radius: float = Field(..., ge=0.0, description="Sphere radius")


class RigidBodyConstraint(BaseModel):
    """Constraint settings for rigid bodies."""
    type: str = Field(..., description="Constraint type")
    strength: float = Field(..., ge=0.0, description="Constraint strength")


class CddRigidBody(BaseModel):
    """Rigid body definition for a CDD domain."""
    id: str = Field(..., description="Rigid body identifier")
    domains: List[str] = Field(..., description="CDD domains in this rigid body")
    residue_range: List[int] = Field(..., description="Residue range for the rigid body")
    sphere: Optional[RigidBodySphere] = Field(
        None,
        description="Sphere approximation for rigid body"
    )
    constraint: Optional[RigidBodyConstraint] = Field(
        None,
        description="Rigid body constraint settings"
    )
    ui: Optional[CddDomainUi] = Field(None, description="UI hints")


class CddArchitecture(BaseModel):
    """CDD architecture metadata."""
    arch_id: str = Field(..., description="CDD architecture identifier")
    order: List[str] = Field(..., description="Ordered list of CDD domains")


class CddAnnotation(BaseModel):
    """CDD annotation response for a PDB chain."""
    pdb_id: str = Field(..., description="PDB identifier")
    chain_id: str = Field(..., description="Chain identifier")
    source: str = Field(..., description="Annotation source")
    fetch_date: str = Field(..., description="Fetch date for CDD data")
    cdd_version: str = Field(..., description="CDD version")
    rpsblast_version: Optional[str] = Field(None, description="RPS-BLAST version")
    full_sequence: Optional[str] = Field(None, description="Full chain sequence")
    architecture: Optional[CddArchitecture] = Field(
        None,
        description="CDD architecture metadata"
    )
    domains: List[CddDomain] = Field(default_factory=list, description="CDD domains")
    linkers: List[CddLinker] = Field(default_factory=list, description="CDD linkers")
    interfaces: List[CddInterface] = Field(
        default_factory=list,
        description="CDD interfaces"
    )
    rigid_bodies: List[CddRigidBody] = Field(
        default_factory=list,
        description="CDD rigid bodies"
    )


class GatingAuditRecord(BaseModel):
    """Audit record for deterministic gating decisions."""
    state_id: str = Field(..., description="State machine identifier")
    gate: str = Field(..., description="Gate name")
    status: str = Field(..., description="Gate status")
    reason: str = Field(..., description="Gate failure reason")
    details: Dict[str, Any] = Field(default_factory=dict, description="Gate details")
    provenance: Dict[str, Any] = Field(
        default_factory=dict,
        description="CDD provenance metadata"
    )


class Dehydron(BaseModel):
    """
    Represents an under-wrapped backbone hydrogen bond.
    
    Validates: Requirements 4.9
    """
    donor_res_id: int = Field(..., ge=1, description="Donor residue ID")
    acceptor_res_id: int = Field(..., ge=1, description="Acceptor residue ID")
    wrapping_count: int = Field(..., ge=0, description="Number of wrapping non-polar carbons")
    midpoint: tuple[float, float, float] = Field(..., description="3D midpoint coordinates")
    distance: float = Field(..., gt=0, le=3.5, description="N-O distance in Angstroms")
    is_dehydron: bool = Field(..., description="True if wrapping_count < threshold")
    donor_chain_id: str = Field("", description="Chain of donor residue")
    acceptor_chain_id: str = Field("", description="Chain of acceptor residue")
    is_interchain: bool = Field(False, description="True if H-bond crosses chain boundary")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "donor_res_id": 5,
                "acceptor_res_id": 12,
                "wrapping_count": 14,
                "midpoint": [10.2, 15.3, 8.7],
                "distance": 2.9,
                "is_dehydron": True,
                "donor_chain_id": "A",
                "acceptor_chain_id": "A",
                "is_interchain": False
            }
        }
    )


class Void(BaseModel):
    """
    Represents a volumetric cavity in protein structure.
    
    Validates: Requirements 5.6
    """
    void_id: int = Field(..., ge=1, description="Unique void identifier")
    center: tuple[float, float, float] = Field(..., description="Void center coordinates")
    volume: float = Field(..., ge=10.0, description="Void volume in cubic Angstroms")
    point_count: int = Field(..., ge=1, description="Number of grid points in void")
    nearby_dehydrons: List[int] = Field(
        default_factory=list, 
        description="IDs of dehydrons within 6.0Å"
    )
    points: Optional[List[List[float]]] = Field(
        None, 
        description="3D point cloud of void"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "void_id": 1,
                "center": [12.5, 18.3, 10.2],
                "volume": 45.3,
                "point_count": 362,
                "nearby_dehydrons": [5, 12],
                "points": [[12.0, 18.0, 10.0], [12.5, 18.5, 10.5]]
            }
        }
    )


class RedZoneFlagDetails(BaseModel):
    """Details for Red Zone flags (immunogenicity or metabolism)."""
    # Immunogenicity fields
    sequence: Optional[str] = None
    start_res: Optional[int] = None
    end_res: Optional[int] = None
    allele: Optional[str] = None
    ic50: Optional[float] = None
    
    # Metabolism fields
    vulnerable_atoms: Optional[List[int]] = None
    isoform: Optional[str] = None
    som_count: Optional[int] = None


class RedZoneFlag(BaseModel):
    """
    Represents a safety violation (immunogenicity or metabolic liability).
    
    Validates: Requirements 7.6, 8.7
    """
    flag_type: Literal['immunogenicity', 'metabolism'] = Field(
        ..., 
        description="Type of Red Zone violation"
    )
    severity: Literal['high', 'medium', 'low'] = Field(
        ..., 
        description="Severity level"
    )
    details: RedZoneFlagDetails = Field(..., description="Type-specific details")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
        }
    )


class Stabilizer(BaseModel):
    """
    Represents a molecular stabilizer placed near dehydrons/voids.
    
    Validates: Requirements 6.1, 6.2
    """
    stabilizer_id: str = Field(..., description="Unique stabilizer identifier")
    type: Literal['trehalose', 'glycerol', 'proline', 'custom'] = Field(
        ..., 
        description="Stabilizer molecule type"
    )
    position: tuple[float, float, float] = Field(..., description="3D placement coordinates")
    target_dehydron_id: int = Field(..., ge=1, description="Target dehydron ID")
    wrapping_bonus: int = Field(..., ge=0, description="Calculated wrapping increase")
    effect_radius: float = Field(..., gt=0, description="Effect radius in Angstroms")
    atom_count: int = Field(..., ge=1, description="Number of atoms in stabilizer")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "stabilizer_id": "stab_001",
                "type": "trehalose",
                "position": [10.5, 15.2, 8.9],
                "target_dehydron_id": 5,
                "wrapping_bonus": 8,
                "effect_radius": 6.5,
                "atom_count": 45
            }
        }
    )


class BondOutlier(BaseModel):
    """Represents a bond length or angle outlier from a validation report."""
    mol_id: int = Field(..., description="Molecule ID from validation report")
    chain: str = Field(..., description="Chain identifier")
    residue_id: int = Field(..., description="Residue ID")
    residue_type: str = Field(..., description="Residue name")
    atoms: tuple[str, str] = Field(..., description="Atom names involved in outlier")
    z_score: float = Field(..., description="Z-score of the outlier")
    observed: float = Field(..., description="Observed value (Angstroms or degrees)")
    ideal: float = Field(..., description="Ideal value (Angstroms or degrees)")
    deviation: float = Field(..., description="Observed - ideal")
    outlier_type: Literal["length", "angle"] = Field(..., description="Outlier type")

    model_config = ConfigDict(frozen=True)


class Clash(BaseModel):
    """Represents an atomic clash from a validation report."""
    atom1: str = Field(..., description="Atom identifier for clash partner 1")
    atom2: str = Field(..., description="Atom identifier for clash partner 2")
    distance: float = Field(..., ge=0, description="Clash distance in Angstroms")
    clash_magnitude: float = Field(..., ge=0, description="Clash overlap magnitude")


class OutlierDehydronCorrelation(BaseModel):
    """Represents correlation between an outlier and nearby dehydron."""
    outlier: BondOutlier
    dehydron_id: int = Field(..., ge=0, description="Index of dehydron in list")
    distance: float = Field(..., ge=0, description="Distance between outlier and dehydron")
    correlation_score: float = Field(..., description="Correlation score")


class GlueableSite(BaseModel):
    """Represents a cluster of dehydrons that forms a glueable site."""
    site_id: int = Field(..., ge=1, description="Unique site identifier")
    dehydron_ids: List[int] = Field(..., description="Indices of dehydrons in site")
    avg_rho: float = Field(..., ge=0, description="Average wrapping count")
    void_volume: float = Field(..., ge=0, description="Estimated void volume")
    score: float = Field(..., description="Glueability score")
    representative_dehydron: Optional[str] = Field(
        None, description="Human-readable representative dehydron"
    )


class WrapperSuggestion(BaseModel):
    """Represents wrapper/stabilizer suggestions for a glueable site."""
    site_id: int = Field(..., ge=1, description="Glueable site identifier")
    wrappers: List[Dict[str, Any]] = Field(..., description="Wrapper list with type/gain")
    predicted_ddg: float = Field(..., description="Predicted delta delta G")


class ValidationMiningConclusion(BaseModel):
    """Summary conclusion from the validation mining pipeline."""
    summary: str = Field(..., description="Human-readable summary")
    supported: bool = Field(..., description="Whether hypothesis is supported")
    confidence: Literal["high", "moderate", "low"] = Field(..., description="Confidence")
    warnings: List[str] = Field(default_factory=list, description="Warnings and caveats")


class ValidationMiningTiming(BaseModel):
    """Timing breakdown for pipeline steps."""
    total: float = Field(..., ge=0, description="Total runtime (seconds)")
    fetch: float = Field(..., ge=0, description="Fetch time (seconds)")
    dehydrons: float = Field(..., ge=0, description="Dehydron computation time")
    correlation: float = Field(..., ge=0, description="Correlation computation time")
    glue_detection: Optional[float] = Field(None, ge=0, description="Glue site detection time")


class ValidationMiningResponse(BaseModel):
    """Response model for validation mining pipeline."""
    status: Literal["success", "partial", "error"]
    run_id: str = Field(..., description="Unique run identifier")
    timestamp: datetime = Field(..., description="Run timestamp")
    input: Dict[str, Any] = Field(..., description="Original request input")
    structure_summary: Dict[str, Any] = Field(..., description="Structure metadata")
    validation_outliers: Dict[str, Any] = Field(..., description="Outlier summary")
    dehydrons: Dict[str, Any] = Field(..., description="Dehydron summary")
    voids: Dict[str, Any] = Field(..., description="Void summary")
    correlations: List[OutlierDehydronCorrelation] = Field(default_factory=list)
    glueable_sites: List[GlueableSite] = Field(default_factory=list)
    wrapper_suggestions: List[WrapperSuggestion] = Field(default_factory=list)
    gating_audit: List[GatingAuditRecord] = Field(default_factory=list)
    conclusion: ValidationMiningConclusion
    timings: ValidationMiningTiming


class AuditEvent(BaseModel):
    """
    Represents a cryptographically chained audit trail event.
    
    Validates: Requirements 12.1, 12.2, 12.3
    """
    event_id: str = Field(..., description="Unique event identifier")
    timestamp: str = Field(..., description="Event timestamp (ISO 8601)")
    state_from: str = Field(..., description="Source state")
    state_to: str = Field(..., description="Destination state")
    event: str = Field(..., description="Event type/name")
    calc_values: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Calculated parameters"
    )
    user_id: str = Field(..., description="User identifier")
    hash: str = Field(..., min_length=64, max_length=64, description="SHA-256 hash")
    prev_hash: str = Field(..., description="Previous event hash (empty for first event)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "event_id": "evt_123",
                "timestamp": "2026-02-13T14:23:01Z",
                "state_from": "ALIGN",
                "state_to": "LOCK",
                "event": "energy_validation",
                "calc_values": {"delta_g": -18.2, "sasa": 2500.0},
                "user_id": "user@example.com",
                "hash": "a3f2b8c9d1e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0",
                "prev_hash": "b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3"
            }
        }
    )


class EnergyCalculationResult(BaseModel):
    """
    Represents the result of a physics energy calculation.
    
    Validates: Requirements 2.1, 2.2, 2.3, 2.4
    """
    delta_g: float = Field(..., description="Gibbs free energy in kcal/mol")
    potential_energy: float = Field(..., description="Potential energy from force field in kcal/mol")
    solvation_term: float = Field(..., description="Solvation penalty in kcal/mol")
    sasa: float = Field(..., ge=0, description="Solvent-accessible surface area in Ų")
    computation_time_sec: float = Field(..., ge=0, description="Computation time in seconds")
    force_field: str = Field(..., description="Force field used (MMFF94 or UFF)")
    converged: bool = Field(..., description="Whether minimization converged")
    provenance: Optional["ComputationProvenance"] = Field(
        None, description="Full computation provenance chain"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "delta_g": -18.2,
                "potential_energy": -1850.3,
                "solvation_term": 12.5,
                "sasa": 2500.0,
                "computation_time_sec": 2.3,
                "force_field": "MMFF94",
                "converged": True
            }
        }
    )


class ProvenanceStep(BaseModel):
    """
    A single step in a computation pipeline.

    Records what method was attempted, whether it succeeded or failed,
    its inputs/outputs, and any approximations made.
    """
    step_id: int = Field(..., description="Sequential step number within this computation")
    method: str = Field(..., description="Method name, e.g. 'mmff94_parameterization'")
    status: Literal["success", "failed", "skipped"] = Field(
        ..., description="Outcome of this step"
    )
    duration_sec: float = Field(..., ge=0, description="Wall-clock duration in seconds")
    inputs: Dict[str, Any] = Field(
        default_factory=dict, description="What went into this step"
    )
    outputs: Dict[str, Any] = Field(
        default_factory=dict, description="What came out of this step"
    )
    reason: Optional[str] = Field(
        None, description="Why this step ran or why it failed"
    )
    approximations: List[str] = Field(
        default_factory=list,
        description="Explicit flags for any approximations made"
    )


class ComputationProvenance(BaseModel):
    """
    Full provenance record for a computation.

    Tracks every step in the pipeline, what method was requested vs.
    what actually produced the result, and a confidence assessment.
    """
    computation_id: str = Field(..., description="UUID for this computation")
    computation_type: str = Field(
        ..., description="Type: 'energy', 'dehydron_analysis', etc."
    )
    requested_method: str = Field(..., description="What method was requested")
    actual_method: str = Field(..., description="What method produced the result")
    steps: List[ProvenanceStep] = Field(
        default_factory=list, description="Ordered pipeline steps"
    )
    warnings: List[str] = Field(
        default_factory=list, description="Operator-facing warning messages"
    )
    total_duration_sec: float = Field(..., ge=0, description="Total wall-clock time")
    confidence: Literal["high", "medium", "low"] = Field(
        ..., description="Confidence in the result"
    )


# Rebuild EnergyCalculationResult to resolve the forward reference to ComputationProvenance
EnergyCalculationResult.model_rebuild()


class SynthesisFeasibilityResult(BaseModel):
    """
    Represents the result of a synthesis feasibility check.
    
    Validates: Requirements 10.5, 10.6
    """
    manufacturable: bool = Field(..., description="Whether sequence is manufacturable")
    complexity_score: Optional[float] = Field(None, ge=0, description="Synthesis complexity score")
    issues: List[str] = Field(default_factory=list, description="Specific synthesis issues")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details from API")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "manufacturable": True,
                "complexity_score": 2.3,
                "issues": [],
                "details": {"gc_content": 0.52, "repeat_regions": 0}
            }
        }
    )


class SynthesisFeasibilityRequest(BaseModel):
    """
    Request for synthesis feasibility check.
    
    Validates: Requirements 10.1, 10.2, 10.3
    """
    sequences: List[str] = Field(..., min_length=1, description="Amino acid sequences to check")
    api_key: Optional[str] = Field(None, description="Twist Bioscience API key (optional)")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "sequences": ["MKFLKFSLLTAVLLSVVFAFSSCGDDDDKGAEDLGKDGKIGKEFKRIVQRIKDFLRNLVPRTES"],
                "api_key": "twist_api_key_here"
            }
        }
    )


class SynthesisFeasibilityResponse(BaseModel):
    """
    Response from synthesis feasibility check.
    
    Validates: Requirements 10.4, 10.5, 10.6
    """
    results: Dict[str, SynthesisFeasibilityResult] = Field(
        ..., 
        description="Results keyed by sequence identifier"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "results": {
                    "seq_1": {
                        "manufacturable": True,
                        "complexity_score": 2.3,
                        "issues": [],
                        "details": {}
                    }
                }
            }
        }
    )


class AutoprotocolRequest(BaseModel):
    """
    Request for Autoprotocol generation.
    
    Validates: Requirements 11.1, 11.2, 11.3
    """
    sequence: str = Field(..., min_length=1, description="Amino acid sequence to synthesize")
    template_volume_ul: float = Field(2.0, gt=0, description="DNA template volume in microliters")
    extract_volume_ul: float = Field(10.0, gt=0, description="Cell extract volume in microliters")
    incubation_time_hours: float = Field(3.0, gt=0, description="Incubation time in hours")
    temperature_celsius: float = Field(30.0, gt=0, le=100, description="Incubation temperature in Celsius")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "sequence": "MKFLKFSLLTAVLLSVVFAFSSCGDDDDKGAEDLGKDGKIGKEFKRIVQRIKDFLRNLVPRTES",
                "template_volume_ul": 2.0,
                "extract_volume_ul": 10.0,
                "incubation_time_hours": 3.0,
                "temperature_celsius": 30.0
            }
        }
    )


class AutoprotocolResponse(BaseModel):
    """
    Response from Autoprotocol generation.
    
    Validates: Requirements 11.4, 11.5, 11.6, 11.7
    """
    protocol: Dict[str, Any] = Field(..., description="Autoprotocol JSON")
    instruction_count: int = Field(..., ge=0, description="Number of instructions in protocol")
    schema_valid: bool = Field(..., description="Whether protocol passes schema validation")
    metadata: Dict[str, Any] = Field(..., description="Protocol metadata")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "protocol": {
                    "refs": {
                        "reaction_plate": {"new": "96-pcr", "discard": False}
                    },
                    "instructions": [
                        {"op": "provision", "resource_id": "rs123", "to": [{"well": "reaction_plate/0", "volume": "2:microliter"}]}
                    ]
                },
                "instruction_count": 15,
                "schema_valid": True,
                "metadata": {
                    "sequence": "MKFLKFSLLTAVLLSVVFAFSSCGDDDDKGAEDLGKDGKIGKEFKRIVQRIKDFLRNLVPRTES",
                    "template_volume_ul": 2.0,
                    "extract_volume_ul": 10.0,
                    "incubation_time_hours": 3.0,
                    "temperature_celsius": 30.0,
                    "generated_at": "2026-02-13T14:23:01Z"
                }
            }
        }
    )

class DehydronBonusItem(BaseModel):
    """
    Individual dehydron bonus calculation result.
    
    Spec Reference: Section 6.6 "Dehydron Bonus"
    """
    dehydron_id: str = Field(..., description="Unique identifier for dehydron (e.g., 'HB_12_15')")
    residue_pair: Tuple[int, int] = Field(..., description="Donor and acceptor residue IDs")
    wrapping_unbound: int = Field(..., ge=0, description="Wrapping count in unbound state")
    wrapping_bound: int = Field(..., ge=0, description="Wrapping count in bound state")
    bonus: int = Field(..., ge=0, description="Stability bonus (wrapping_bound - wrapping_unbound)")
    is_dehydron: bool = Field(..., description="Whether this is an active dehydron (wrapping_unbound < 19)")
    stability_weight: float = Field(..., ge=0.0, le=1.0, description="Normalized stability weight [0.0, 1.0]")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "dehydron_id": "HB_12_15",
                "residue_pair": [12, 15],
                "wrapping_unbound": 14,
                "wrapping_bound": 20,
                "bonus": 6,
                "is_dehydron": True,
                "stability_weight": 0.45
            }
        }
    )


class DehydronBonusRequest(BaseModel):
    """
    Request for dehydron bonus calculation.
    
    Spec Reference: Section 6.6 "Dehydron Bonus"
    """
    dehydrons: List[Dehydron] = Field(..., min_items=1, description="List of detected dehydrons")
    baseline_wrapping: Optional[Dict[str, int]] = Field(
        None,
        description="Optional baseline wrapping map for unbound state. Keys: 'res_a-res_b' format"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "dehydrons": [
                    {
                        "id": "HB_12_15",
                        "donor_res_id": 12,
                        "acceptor_res_id": 15,
                        "donor_atom": "O",
                        "acceptor_atom": "N",
                        "distance": 2.8,
                        "secondary_structure": "H",
                        "environment": "core",
                        "wrapping_count": 20,
                        "midpoint": [10.5, 8.3, -2.9]
                    }
                ],
                "baseline_wrapping": None
            }
        }
    )


class DehydronBonusResponse(BaseModel):
    """
    Response from dehydron bonus calculation service.
    
    Spec Reference: Section 6.6 "Dehydron Bonus"
    """
    status: Literal["success", "partial", "error"] = Field(..., description="Response status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    bonuses: List[DehydronBonusItem] = Field(..., description="Calculated bonuses for each dehydron")
    summary: Dict[str, Any] = Field(..., description="Summary metrics")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal warnings during calculation")
    error: Optional[str] = Field(None, description="Error message if status is 'error'")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "success",
                "timestamp": "2026-02-16T14:23:01Z",
                "bonuses": [
                    {
                        "dehydron_id": "HB_12_15",
                        "residue_pair": [12, 15],
                        "wrapping_unbound": 14,
                        "wrapping_bound": 20,
                        "bonus": 6,
                        "is_dehydron": True,
                        "stability_weight": 0.45
                    }
                ],
                "summary": {
                    "total_bonus": 18,
                    "avg_stability_weight": 0.38,
                    "dehydron_count": 3,
                    "stabilized_count": 3
                },
                "warnings": [],
                "error": None
            }
        }
    )