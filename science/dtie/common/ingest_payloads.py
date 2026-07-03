"""Pydantic payload models for structure ingestion and alignment.

These models define the contract between the ingestion pipeline (science layer)
and the governed data layer (Normalizer). Two main payload paths:

Path A: IngestDimensionPayload — dimensions + covalent bonds from BinaryCIF parsing
Path B: AlignmentPayload — UniProt residue alignment + structural superposition

Requirements: 1.3, 1.4, 1.5, 1.6, 6.2, 7.3, 8.1
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from science.dtie.common.keys import validate_residue_id
from science.dtie.common.normalizer_payloads import ProvenanceContext


# ---------------------------------------------------------------------------
# Sub-models: Dimensions
# ---------------------------------------------------------------------------


class StructureDimension(BaseModel):
    """Dimension record for a PDB structure (Requirement 1.3)."""

    structure_id: str
    pdb_id: str
    source: str = "rcsb"
    method: str
    resolution: float | None = None
    r_factor: float | None = None
    r_free: float | None = None
    title: str
    organism: str | None = None
    release_date: str | None = None
    polymer_composition: str
    model_count: int = 1
    assembly_id: str | None = None


class ChainDimension(BaseModel):
    """Dimension record for a polymer chain (Requirement 1.4)."""

    chain_id: str
    structure_id: str
    auth_asym_id: str
    label_asym_id: str
    entity_id: str
    entity_type: str
    sequence_length: int
    uniprot_accession: str | None = None
    uniprot_start: int | None = None
    uniprot_end: int | None = None
    is_entity_duplicate: bool = False
    is_representative: bool = True


class ResidueDimension(BaseModel):
    """Dimension record for a residue (Requirement 1.5)."""

    residue_id: str
    chain_id: str
    residue_index: int  # auth_seq_id
    label_seq_id: int | None = None
    insertion_code: str | None = None
    residue_name: str
    residue_name_3: str
    comp_id: str
    parent_comp_id: str | None = None
    sse_code: str | None = None
    is_resolved: bool = True
    is_modified: bool = False
    max_b_factor: float | None = None
    low_confidence_coords: bool = False
    partial_backbone: bool = False

    @field_validator("residue_id")
    @classmethod
    def validate_residue_id_format(cls, v: str) -> str:
        if not validate_residue_id(v):
            raise ValueError(
                f"residue_id '{v}' does not match canonical format "
                "(expected: <structure>:<chain>:<index>[:<insertion>])"
            )
        return v


class AtomDimension(BaseModel):
    """Dimension record for an atom (Requirement 1.6)."""

    atom_id: str
    residue_id: str
    atom_name: str
    element: str
    x: float
    y: float
    z: float
    occupancy: float
    b_factor: float
    altloc: str | None = None
    is_hetero: bool = False
    model_id: int = 1

    @field_validator("residue_id")
    @classmethod
    def validate_residue_id_format(cls, v: str) -> str:
        if not validate_residue_id(v):
            raise ValueError(
                f"residue_id '{v}' does not match canonical format "
                "(expected: <structure>:<chain>:<index>[:<insertion>])"
            )
        return v


# ---------------------------------------------------------------------------
# Sub-models: Facts
# ---------------------------------------------------------------------------


class CovalentBondFact(BaseModel):
    """Fact record for a covalent bond (Requirement 8.1)."""

    residue_id_1: str
    residue_id_2: str
    atom_name_1: str
    atom_name_2: str
    bond_type: str  # disulf, covale, metalc, etc.

    @field_validator("residue_id_1", "residue_id_2")
    @classmethod
    def validate_residue_id_format(cls, v: str) -> str:
        if not validate_residue_id(v):
            raise ValueError(
                f"residue_id '{v}' does not match canonical format "
                "(expected: <structure>:<chain>:<index>[:<insertion>])"
            )
        return v


# ---------------------------------------------------------------------------
# Main payload: Ingest Dimensions
# ---------------------------------------------------------------------------


class IngestDimensionPayload(BaseModel):
    """Complete payload for persisting structure dimensions via Normalizer.

    This is the contract between the ingestion pipeline and the governed
    data layer. All dimension writes go through this payload.
    """

    provenance: ProvenanceContext
    structure: StructureDimension
    chains: list[ChainDimension]
    residues: list[ResidueDimension]
    atoms: list[AtomDimension]
    covalent_bonds: list[CovalentBondFact] = Field(default_factory=list)
    file_hash: str = Field(..., description="SHA-256 hash of the BinaryCIF file")
    biotite_version: str = Field(..., description="biotite library version used for parsing")
    rcsbapi_version: str = Field(..., description="rcsbapi library version used for metadata")


# ---------------------------------------------------------------------------
# Sub-models: Alignment
# ---------------------------------------------------------------------------


class ResidueAlignmentRecord(BaseModel):
    """Record mapping a residue to its UniProt canonical position (Requirement 6.2)."""

    residue_id: str
    uniprot_accession: str
    uniprot_position: int | None = None  # NULL = unmapped
    isoform_id: str | None = None
    mapping_source: str = "sifts"
    mapping_confidence: float
    reason_code: str | None = None  # NULL = success; else: tag, engineered, unmapped

    @field_validator("residue_id")
    @classmethod
    def validate_residue_id_format(cls, v: str) -> str:
        if not validate_residue_id(v):
            raise ValueError(
                f"residue_id '{v}' does not match canonical format "
                "(expected: <structure>:<chain>:<index>[:<insertion>])"
            )
        return v


class StructuralAlignmentRecord(BaseModel):
    """Record for a Kabsch superposition between two structures (Requirement 7.3)."""

    query_structure_id: str
    reference_structure_id: str
    uniprot_accession: str
    rotation_matrix: list[list[float]] = Field(
        ..., description="3x3 rotation matrix as nested list"
    )
    translation: list[float] = Field(..., description="3-element translation vector")
    rmsd: float
    aligned_residue_count: int
    comparable_core: dict = Field(
        ..., description="JSON: criteria + positions defining the comparable core"
    )
    protocol_version: str

    @field_validator("rotation_matrix")
    @classmethod
    def validate_rotation_shape(cls, v: list[list[float]]) -> list[list[float]]:
        if len(v) != 3 or any(len(row) != 3 for row in v):
            raise ValueError("rotation_matrix must be 3x3")
        return v

    @field_validator("translation")
    @classmethod
    def validate_translation_shape(cls, v: list[float]) -> list[float]:
        if len(v) != 3:
            raise ValueError("translation must have exactly 3 elements")
        return v


# ---------------------------------------------------------------------------
# Main payload: Alignment
# ---------------------------------------------------------------------------


class AlignmentPayload(BaseModel):
    """Complete payload for persisting alignment data via Normalizer.

    Covers both residue-level UniProt mapping and structural superposition.
    """

    provenance: ProvenanceContext
    structure_id: str
    residue_alignments: list[ResidueAlignmentRecord]
    structural_alignments: list[StructuralAlignmentRecord] | None = None


# ---------------------------------------------------------------------------
# Computation scope (assign_computation_scope foundation job)
# ---------------------------------------------------------------------------


class QualityFiltersPayload(BaseModel):
    """Quality filter thresholds stored on structure_computation_scope."""

    max_b_factor_threshold: float = 100.0
    min_resolution: float | None = None


class ComputationScopeRecord(BaseModel):
    """Row payload for structure_computation_scope."""

    structure_id: str
    primary_chain_ids: list[str]
    reference_chain: str
    exclude_chain_ids: list[str] = Field(default_factory=list)
    scope_source: str
    selection_reason: str
    normalization_protocol: str = "graph_default"
    quality_filters: QualityFiltersPayload | None = None
    model_index: int = 1


class ComputationScopePayload(BaseModel):
    """Governed write contract for assign_computation_scope."""

    provenance: ProvenanceContext
    scope: ComputationScopeRecord
