# Migrated from: new (Phase 1 implementation) on 2026-05-27
"""Normalizer payload schemas for the two highest-volume write paths.

These Pydantic models define the EXACT contract between science code and
the governed data layer. The Normalizer will only accept payloads that
validate against these schemas.

Path 1: GNN Node Output (v4 hyperbolic + v3 Euclidean)
Path 2: Phase 3 Persistence Output (v3 + v4 variants)

See: data/NORMALIZER_DESIGN.md for the overall write-path architecture.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SourceType(str, Enum):
    """Classification of how a data point was produced."""

    DETERMINISTIC = "deterministic"
    PROBABILISTIC = "probabilistic"
    EXTERNAL = "external"
    DERIVED = "derived"


class RunType(str, Enum):
    """Classification of the producing run."""

    INFERENCE = "inference"
    TRAINING = "training"
    ANALYSIS = "analysis"
    HISTORICAL_BACKFILL = "historical_backfill"


class SpaceType(str, Enum):
    """Embedding space geometry type."""

    EUCLIDEAN = "euclidean"
    HYPERBOLIC = "hyperbolic"


# ---------------------------------------------------------------------------
# Provenance context (required for all payloads)
# ---------------------------------------------------------------------------


class ProvenanceContext(BaseModel):
    """Provenance metadata that MUST accompany every Normalizer call.

    This is not optional. The Normalizer will reject any payload without
    a valid provenance context.
    """

    run_id: str = Field(..., description="Unique identifier for this computation run")
    structure_id: str = Field(..., description="Canonical structure_id")
    model_version: str = Field(..., description="Model identifier (e.g., 'GOSPConeMapper-v4')")
    pipeline_name: str = Field(..., description="Pipeline that produced this (e.g., 'dtie_v4')")
    run_type: RunType = Field(default=RunType.INFERENCE)
    source_type: SourceType = Field(default=SourceType.PROBABILISTIC)
    checkpoint_uri: str | None = Field(default=None)
    checkpoint_sha256: str | None = Field(default=None)
    code_version: str | None = Field(default=None, description="Git commit hash")
    parameters: dict[str, Any] | None = Field(default=None)
    parent_run_id: str | None = Field(default=None)


# ---------------------------------------------------------------------------
# Path 1: GNN Node Output Payload
# ---------------------------------------------------------------------------


class GNNNodeResult(BaseModel):
    """Per-residue GNN output for a single node.

    This represents the output of one forward pass through the GNN for
    one residue in one structure.
    """

    residue_id: str = Field(..., description="Canonical residue_id")
    residue_index: int = Field(..., description="Author residue number")
    chain_label: str = Field(..., description="Chain identifier")

    # Input features (stored for auditability)
    input_rho: float = Field(..., description="Dehydron density")
    input_tau_flag: float = Field(..., description="Tau torsion flag")
    input_ss_type: float = Field(..., description="Secondary structure encoding")
    input_sasa: float = Field(..., description="Solvent accessible surface area")

    # Core outputs
    embedding: list[float] = Field(..., description="Primary embedding vector")
    cone_depth: float | None = Field(default=None)
    cone_width: float | None = Field(default=None)

    # Uncertainty (v4 produces both; v3 may only have epistemic)
    epistemic_uncertainty: float | None = Field(default=None)
    aleatoric_uncertainty: float | None = Field(default=None)
    total_uncertainty: float | None = Field(default=None)

    # v4-specific: native hyperbolic outputs
    x_hyp: list[float] | None = Field(
        default=None, description="Full hyperbolic embedding (Poincaré ball)"
    )
    hyp_projections: list[float] | None = Field(
        default=None, description="Native 2D Poincaré disc projection"
    )
    x_routed_hyp: list[float] | None = Field(
        default=None, description="Post-MoE hyperbolic embedding"
    )

    # Expert routing info
    expert_weights: list[float] | None = Field(default=None)

    @field_validator("embedding")
    @classmethod
    def embedding_not_empty(cls, v: list[float]) -> list[float]:
        if len(v) == 0:
            raise ValueError("embedding must not be empty")
        return v


class GNNOutputPayload(BaseModel):
    """Complete GNN output payload for one structure.

    This is what the science code passes to the Normalizer after running
    GNN inference on a structure.
    """

    provenance: ProvenanceContext
    space_type: SpaceType = Field(
        ..., description="Primary embedding space geometry"
    )
    space_name: str = Field(
        ..., description="Registered embedding space name (must exist in registry)"
    )
    dimensionality: int = Field(..., description="Embedding vector dimensionality")
    curvature: float | None = Field(
        default=None, description="Curvature parameter (hyperbolic only)"
    )
    nodes: list[GNNNodeResult] = Field(
        ..., description="Per-residue results", min_length=1
    )
    computed_at: datetime = Field(default_factory=_utcnow)

    @field_validator("curvature")
    @classmethod
    def curvature_required_for_hyperbolic(cls, v: float | None, info: Any) -> float | None:
        if info.data.get("space_type") == SpaceType.HYPERBOLIC and v is None:
            raise ValueError("curvature is required for hyperbolic spaces")
        return v


# ---------------------------------------------------------------------------
# Path 2: Phase 3 Persistence Output Payload
# ---------------------------------------------------------------------------


class PersistenceBarcode(BaseModel):
    """A single persistence barcode entry."""

    birth: float
    death: float
    dimension: int = Field(default=0, description="Homological dimension (0, 1, 2)")
    generator_residues: list[str] | None = Field(
        default=None, description="Residue IDs that generate this feature"
    )


class Phase3ResidueContribution(BaseModel):
    """Per-residue contribution to the persistence landscape."""

    residue_id: str
    persistence_score: float = Field(..., description="Aggregated persistence contribution")
    max_barcode_length: float | None = Field(default=None)
    topological_significance: float | None = Field(default=None)


class Phase3PersistencePayload(BaseModel):
    """Complete Phase 3 persistence output for one structure.

    Covers both v3 (standard witness persistence) and v4 (enhanced
    witness persistence with hyperbolic distance integration).
    """

    provenance: ProvenanceContext

    # Global persistence results
    barcodes: list[PersistenceBarcode] = Field(
        ..., description="Persistence diagram as list of barcodes"
    )
    max_alpha: float = Field(..., description="Maximum filtration value used")
    n_witnesses: int
    n_landmarks: int

    # Per-residue contributions (optional but strongly encouraged)
    residue_contributions: list[Phase3ResidueContribution] | None = Field(default=None)

    # v4-specific: hyperbolic distance integration
    hyperbolic_distances_used: bool = Field(
        default=False,
        description="Whether hyperbolic (Poincaré) distances were used in filtration",
    )
    curvature_c: float | None = Field(
        default=None, description="Curvature used for hyperbolic distance computation"
    )

    # Witness-to-residue mapping (for provenance and visualization)
    landmark_to_residue: dict[str, str] | None = Field(
        default=None, description="Mapping from landmark index to residue_id"
    )

    computed_at: datetime = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# Path 3: Graph Topology Payload
# ---------------------------------------------------------------------------

VALID_EDGE_TYPES = frozenset({"h_bond", "contact", "covalent", "disulfide", "salt_bridge"})


class GraphEdge(BaseModel):
    """A single edge in the molecular contact graph."""

    source_residue_id: str = Field(..., description="Canonical residue_id of source node")
    target_residue_id: str = Field(..., description="Canonical residue_id of target node")
    edge_type: str = Field(..., description="One of: h_bond, contact, covalent, disulfide, salt_bridge")
    distance_angstrom: float | None = Field(default=None, description="Euclidean Cα distance")
    hyperbolic_distance: float | None = Field(default=None, description="Poincaré distance")
    weight: float = Field(default=1.0, description="Edge weight")
    metadata: dict[str, Any] | None = Field(default=None, description="Extensible metadata")

    @field_validator("edge_type")
    @classmethod
    def edge_type_in_allowlist(cls, v: str) -> str:
        if v not in VALID_EDGE_TYPES:
            raise ValueError(
                f"edge_type must be one of {sorted(VALID_EDGE_TYPES)}, got '{v}'"
            )
        return v

    @field_validator("source_residue_id", "target_residue_id")
    @classmethod
    def residue_id_format(cls, v: str) -> str:
        from science.dtie.common.keys import validate_residue_id

        if not validate_residue_id(v):
            raise ValueError(
                f"residue_id '{v}' does not match canonical format "
                "(expected: <structure>:<chain>:<index>[:<insertion>])"
            )
        return v


class GraphTopologyPayload(BaseModel):
    """Complete graph topology payload for one structure.

    This is what the science code passes to the Normalizer after building
    the contact graph for a structure.
    """

    provenance: ProvenanceContext
    structure_id: str = Field(..., description="Canonical structure_id")
    edges: list[GraphEdge] = Field(
        ..., description="All edges in the contact graph", min_length=1
    )
    computed_at: datetime = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# Path 4: Hypothesis Engine Payloads
# ---------------------------------------------------------------------------


class HypothesisPredictionPayload(BaseModel):
    """A single testable prediction within a hypothesis."""

    prediction_id: str = Field(..., description="Unique prediction identifier")
    statement: str = Field(..., description="What this prediction claims")
    test_tool: str | None = Field(default=None, description="Tool to call for testing")
    test_params: dict[str, Any] | None = Field(default=None, description="Parameters for the test tool")
    threshold: str | None = Field(default=None, description="Threshold expression (e.g., 'value > 0.15')")


class HypothesisPayload(BaseModel):
    """Payload for creating or updating a hypothesis via the Normalizer.

    Requirements: 1.3, 3.1
    """

    provenance: ProvenanceContext
    hypothesis_id: str = Field(..., description="Unique hypothesis identifier")
    structure_id: str = Field(..., description="Structure this hypothesis is about")
    statement: str = Field(..., description="The scientific claim")
    mechanism: str | None = Field(default=None, description="Proposed mechanism")
    predictions: list[HypothesisPredictionPayload] = Field(
        ..., description="Testable predictions (at least one required)", min_length=1
    )
    status: str = Field(default="proposed", description="Hypothesis lifecycle status")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    created_by: str = Field(default="agent")


class EvidencePayload(BaseModel):
    """Payload for adding evidence to an existing hypothesis.

    Requirements: 1.3, 3.1
    """

    provenance: ProvenanceContext
    hypothesis_id: str = Field(..., description="Hypothesis this evidence applies to")
    evidence_id: str = Field(..., description="Unique evidence identifier")
    source_tool: str = Field(..., description="Tool that produced this evidence")
    source_run_id: str | None = Field(default=None, description="Run that produced this evidence")
    supports: bool = Field(..., description="True if evidence supports the hypothesis")
    strength: float = Field(default=0.5, ge=0.0, le=1.0, description="Evidence strength weight")
    description: str = Field(..., description="Human-readable description of the evidence")


# ---------------------------------------------------------------------------
# Normalizer Response
# ---------------------------------------------------------------------------


class NormalizerResult(BaseModel):
    """Response from the Normalizer after a successful write."""

    success: bool
    run_id: str
    assets_created: int
    asset_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
