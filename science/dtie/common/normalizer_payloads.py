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
    RESISTANCE_BASELINE = "resistance_baseline"
    IN_SILICO_MUTATION = "in_silico_mutation"


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

    # High-precision float64 embedding for hyperbolic/Lorentz math (dual-stored with embedding)
    embedding_double: list[float] | None = Field(
        default=None,
        description="High-precision (float64) version of the primary embedding for non-Euclidean spaces. "
                    "Used for exact Lorentz inner product / arcosh calculations.",
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

    # V6 MoE routing metadata (per-run aggregates)
    expert_load: list[float] | None = Field(
        default=None,
        description="V6: Per-expert mean routing probability for this run (e.g. [0.3, 0.25, 0.2, 0.25])",
    )
    routing_entropy: float | None = Field(
        default=None,
        description="V6: Shannon entropy of the expert routing distribution for this run",
    )

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

    @field_validator("residue_id")
    @classmethod
    def residue_id_format(cls, v: str) -> str:
        from science.dtie.common.keys import validate_residue_id

        if not validate_residue_id(v):
            raise ValueError(
                f"residue_id '{v}' does not match canonical format "
                "(expected: <structure>:<chain>:<index>[:<insertion>])"
            )
        return v


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
# Path 2b: Source-Leak Detection Payload
# ---------------------------------------------------------------------------


class SourceLeakResidue(BaseModel):
    """A single residue identified as a potential source leak."""

    residue_id: str = Field(..., description="Canonical residue_id")
    epistemic_uncertainty: float = Field(..., description="Epistemic uncertainty at this residue")
    cone_depth: float = Field(..., description="Cone depth value")
    leak_score: float = Field(..., description="Composite leak score")
    is_confirmed: bool = Field(default=False, description="Whether leak is experimentally confirmed")

    @field_validator("residue_id")
    @classmethod
    def residue_id_format(cls, v: str) -> str:
        from science.dtie.common.keys import validate_residue_id

        if not validate_residue_id(v):
            raise ValueError(
                f"residue_id '{v}' does not match canonical format "
                "(expected: <structure>:<chain>:<index>[:<insertion>])"
            )
        return v


class SourceLeakPayload(BaseModel):
    """Complete source-leak detection output for one structure.

    Contains all residues flagged as potential source leaks along with
    their scoring metrics and provenance.
    """

    provenance: ProvenanceContext
    leak_residues: list[SourceLeakResidue] = Field(
        ..., description="Per-residue source-leak candidates"
    )
    total_leaks: int = Field(..., description="Total number of leaks detected")
    threshold_used: float = Field(..., description="Score threshold used for detection")
    computed_at: datetime = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# Path 2c: Allosteric Site Payload
# ---------------------------------------------------------------------------


class AllostericSiteRecord(BaseModel):
    """A single predicted allosteric site."""

    site_id: str = Field(..., description="Canonical site identifier")
    residue_ids: list[str] = Field(
        ..., description="Constituent residue_ids for this site", min_length=1
    )
    centroid_x: float = Field(..., description="Centroid X coordinate (Angstroms)")
    centroid_y: float = Field(..., description="Centroid Y coordinate (Angstroms)")
    centroid_z: float = Field(..., description="Centroid Z coordinate (Angstroms)")
    confidence_score: float = Field(..., description="Prediction confidence [0, 1]")
    cluster_method: str = Field(default="dbscan", description="Clustering method used")
    n_residues: int = Field(..., description="Number of residues in this site")

    @field_validator("residue_ids")
    @classmethod
    def all_residue_ids_valid(cls, v: list[str]) -> list[str]:
        from science.dtie.common.keys import validate_residue_id

        for rid in v:
            if not validate_residue_id(rid):
                raise ValueError(
                    f"residue_id '{rid}' does not match canonical format "
                    "(expected: <structure>:<chain>:<index>[:<insertion>])"
                )
        return v


class AllostericSitePayload(BaseModel):
    """Complete allosteric site prediction output for one structure.

    Contains all predicted allosteric pockets with their constituent
    residues, centroids, and confidence scores.
    """

    provenance: ProvenanceContext
    sites: list[AllostericSiteRecord] = Field(
        ..., description="Predicted allosteric sites"
    )
    total_sites: int = Field(..., description="Total number of sites identified")
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
# Path 5: Phase 2 Vulnerability Payload
# ---------------------------------------------------------------------------


class VulnerabilityDoorway(BaseModel):
    """A single residue identified as a vulnerability doorway."""

    residue_id: str = Field(..., description="Canonical residue_id (structure:chain:index)")
    cone_depth: float = Field(..., description="Cone depth at this residue")
    epistemic_uncertainty: float = Field(..., description="Epistemic uncertainty value")
    aleatoric_uncertainty: float = Field(default=0.0, description="Aleatoric uncertainty value")


class Phase2VulnerabilityPayload(BaseModel):
    """Complete Phase 2 vulnerability scan output for one structure.

    Contains doorway residues with epistemic uncertainty and depth thresholds.
    """

    provenance: ProvenanceContext
    doorways: list[VulnerabilityDoorway] = Field(
        ..., description="Residues identified as vulnerability doorways"
    )
    epistemic_median: float = Field(..., description="Median epistemic uncertainty across all residues")
    depth_threshold: float = Field(..., description="Depth threshold used for doorway detection")
    total_residues: int = Field(..., description="Total residues evaluated")
    computed_at: datetime = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# Path 6: Phase 3.5 Topological Lift Payload
# ---------------------------------------------------------------------------


class LiftedSite(BaseModel):
    """A single allosteric site lifted from 2D disc to 3D ball coordinates."""

    site_index: int = Field(..., description="Index of the lifted site")
    lifted_x: float = Field(..., description="Lifted X coordinate")
    lifted_y: float = Field(..., description="Lifted Y coordinate")
    lifted_z: float = Field(..., description="Lifted Z coordinate")
    vertex_count: int = Field(..., description="Number of vertices in this site")
    source_method: str = Field(default="unknown", description="Method used for lifting")


class TopologicalLiftPayload(BaseModel):
    """Complete Phase 3.5 topological lift output for one structure.

    Contains lifted allosteric site coordinates in 3D ball space.
    """

    provenance: ProvenanceContext
    lifted_sites: list[LiftedSite] = Field(
        ..., description="Allosteric sites lifted to 3D coordinates"
    )
    method: str = Field(..., description="Lift method identifier (e.g., 'v5_native_ca_lift')")
    computed_at: datetime = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# Path 7: Phase 4 Resistance Pathway Payload
# ---------------------------------------------------------------------------


class ResistancePathway(BaseModel):
    """A single resistance pathway between two graph nodes."""

    source_node: int = Field(..., description="Source node index in the contact graph")
    target_node: int = Field(..., description="Target node index in the contact graph")
    source_residue: int = Field(..., description="Source residue index")
    target_residue: int = Field(..., description="Target residue index")
    effective_resistance: float = Field(..., description="Effective resistance between nodes")
    coupling_strength: float = Field(..., description="Coupling strength of the pathway")


class ResistancePathwayPayload(BaseModel):
    """Complete Phase 4 resistance pathway output for one structure.

    Contains resistance pathways and spectral analysis results.
    """

    provenance: ProvenanceContext
    pathways: list[ResistancePathway] = Field(
        ..., description="Resistance pathways through the contact graph"
    )
    lambda_2: float = Field(..., description="Second eigenvalue (algebraic connectivity)")
    hinge_residues: list[int] = Field(
        default_factory=list, description="Residue indices identified as hinges"
    )
    graph_nodes: int = Field(..., description="Total nodes in the contact graph")
    graph_edges: int = Field(..., description="Total edges in the contact graph")
    computed_at: datetime = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# Path 8: Phase 5 Pharmacophore Payload
# ---------------------------------------------------------------------------


class PharmacophoreRecord(BaseModel):
    """A single pharmacophore feature identified from cone geometry."""

    pocket_index: int = Field(..., description="Index of the pocket")
    center_x: float = Field(..., description="Pocket center X coordinate")
    center_y: float = Field(..., description="Pocket center Y coordinate")
    center_z: float = Field(..., description="Pocket center Z coordinate")
    druggability_score: float = Field(..., description="Druggability score [0, 1]")
    residue_count: int = Field(..., description="Number of residues in this pocket")
    residue_indices: list[int] = Field(..., description="Residue indices forming the pocket")
    allosteric_coupling: float = Field(default=0.0, description="Allosteric coupling strength")
    volume_estimate: float = Field(default=0.0, description="Estimated pocket volume (Å³)")


class PharmacophorePayload(BaseModel):
    """Complete Phase 5 pharmacophore output for one structure.

    Contains druggable pharmacophore features identified from cone geometry.
    """

    provenance: ProvenanceContext
    pharmacophores: list[PharmacophoreRecord] = Field(
        ..., description="Identified pharmacophore features"
    )
    druggability_threshold: float = Field(
        ..., description="Threshold used for druggability classification"
    )
    computed_at: datetime = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# Path 9: Phase 6 Drug Candidate Payload
# ---------------------------------------------------------------------------


class DrugCandidate(BaseModel):
    """A single scored drug candidate pocket."""

    pocket_index: int = Field(..., description="Index of the pocket")
    center_x: float = Field(..., description="Pocket center X coordinate")
    center_y: float = Field(..., description="Pocket center Y coordinate")
    center_z: float = Field(..., description="Pocket center Z coordinate")
    accessibility_score: float = Field(..., description="Solvent accessibility score")
    binding_potential: float = Field(..., description="Predicted binding potential")
    admet_pass: bool = Field(..., description="Whether pocket passes ADMET filtering")
    selectivity_ratio: float = Field(..., description="State selectivity ratio")
    is_state_selective: bool = Field(..., description="Whether candidate is state-selective")
    combined_druggability: float = Field(..., description="Combined druggability score")


class DrugCandidatePayload(BaseModel):
    """Complete Phase 6 drug discovery output for one structure.

    Contains scored pockets, ADMET results, and state-selective candidates.
    """

    provenance: ProvenanceContext
    candidates: list[DrugCandidate] = Field(
        ..., description="Scored drug candidate pockets"
    )
    admet_passed_count: int = Field(..., description="Number of candidates passing ADMET")
    state_selective_count: int = Field(..., description="Number of state-selective candidates")
    computed_at: datetime = Field(default_factory=_utcnow)


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
    asset_metadata: dict[str, Any] | None = Field(default=None)
