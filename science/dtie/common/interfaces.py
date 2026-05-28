# Migrated from: new (Phase 1/3 implementation) on 2026-05-27
"""Versioned protocols for DTIE science code.

These protocols define the contracts that v3 and v4 GNN implementations
must satisfy. They enable:
1. Type-safe interaction between science code and the data layer
2. Clear documentation of what each version produces
3. Adapter pattern for translating internal outputs to Normalizer payloads

Per the Deep Audit (Phase 0.1): v3 and v4 have the same class name
(GOSPConeMapper) but very different internals. These protocols make
the differences explicit and prevent accidental mixing.

See: docs/audit/DEEP_AUDIT_PHASE_0.1.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np


# ---------------------------------------------------------------------------
# GNN Output Contracts (what the models produce)
# ---------------------------------------------------------------------------


@dataclass
class GNNNodeOutput:
    """Per-node output from any GNN version.

    This is the internal representation BEFORE normalization.
    The adapter layer converts this to NormalizerPayload format.
    """

    residue_index: int
    chain_label: str

    # Input features (always 4-dim: rho, tau_flag, ss_type, sasa)
    input_features: np.ndarray  # shape: (4,)

    # Core outputs (present in both v3 and v4)
    projections: np.ndarray  # Euclidean projection (v3: 64-dim, v4: 64-dim)
    cone_depth: float
    cone_width: float
    epistemic_uncertainty: float

    # v4-only outputs (None for v3)
    x_hyp: np.ndarray | None = None  # Full hyperbolic embedding
    x_routed_hyp: np.ndarray | None = None  # Post-MoE hyperbolic
    hyp_projections: np.ndarray | None = None  # Native 2D disc projection
    aleatoric_uncertainty: float | None = None
    total_uncertainty: float | None = None

    # Expert routing (both versions, but different semantics)
    expert_weights: np.ndarray | None = None


@dataclass
class GNNInferenceResult:
    """Complete result from running GNN inference on a structure.

    This is what the GNN runner produces — a collection of per-node
    outputs plus metadata about the run.
    """

    structure_id: str
    model_version: str
    checkpoint_path: str | None
    nodes: list[GNNNodeOutput]
    curvature: float | None = None  # Only for v4 (learned curvature)
    embedding_dim: int = 64
    space_type: str = "euclidean"  # "euclidean" or "hyperbolic"
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# GNN Runner Protocol (what the orchestrator calls)
# ---------------------------------------------------------------------------


@runtime_checkable
class GNNRunner(Protocol):
    """Protocol for running GNN inference.

    Both v3 and v4 runners must implement this interface.
    The orchestrator doesn't need to know which version it's using.
    """

    @property
    def model_version(self) -> str:
        """Return the model version identifier (e.g., 'GOSPConeMapper-v3')."""
        ...

    @property
    def space_type(self) -> str:
        """Return the primary embedding space type ('euclidean' or 'hyperbolic')."""
        ...

    async def run_inference(
        self,
        structure_id: str,
        graph_data: Any,  # torch_geometric.data.Data
        checkpoint_path: str | None = None,
    ) -> GNNInferenceResult:
        """Run GNN inference on a prepared graph.

        Args:
            structure_id: Canonical structure_id.
            graph_data: Prepared PyG Data object with node features and edges.
            checkpoint_path: Path to model checkpoint (optional, uses default).

        Returns:
            GNNInferenceResult with per-node outputs.
        """
        ...


# ---------------------------------------------------------------------------
# Phase Runner Protocol
# ---------------------------------------------------------------------------


@dataclass
class PhaseResult:
    """Generic result from a DTIE phase execution."""

    phase_name: str
    structure_id: str
    model_version: str
    success: bool
    outputs: dict[str, Any]  # Phase-specific outputs
    residue_contributions: dict[str, float] | None = None  # residue_id -> score
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@runtime_checkable
class PhaseRunner(Protocol):
    """Protocol for running a DTIE phase.

    Each phase (1-6d) should implement this interface.
    """

    @property
    def phase_name(self) -> str:
        """Return the phase identifier (e.g., 'phase3_persistence')."""
        ...

    @property
    def model_version(self) -> str:
        """Return the version (e.g., 'DTIE-v4-phase3')."""
        ...

    async def run(
        self,
        structure_id: str,
        gnn_result: GNNInferenceResult | None = None,
        upstream_results: dict[str, PhaseResult] | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> PhaseResult:
        """Execute this phase.

        Args:
            structure_id: Canonical structure_id.
            gnn_result: GNN inference output (if this phase needs it).
            upstream_results: Results from earlier phases (keyed by phase_name).
            parameters: Phase-specific parameters.

        Returns:
            PhaseResult with outputs and metadata.
        """
        ...


# ---------------------------------------------------------------------------
# Orchestrator Protocol
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Configuration for a DTIE pipeline run."""

    structure_id: str
    phases: list[str]  # Which phases to run (e.g., ["phase1", "phase3"])
    model_version: str  # "v3" or "v4"
    checkpoint_path: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    parent_run_id: str | None = None


@dataclass
class PipelineResult:
    """Complete result from a DTIE pipeline execution."""

    run_id: str
    structure_id: str
    model_version: str
    success: bool
    phase_results: dict[str, PhaseResult]
    gnn_result: GNNInferenceResult | None = None
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Orchestrator(Protocol):
    """Protocol for DTIE pipeline orchestration.

    Both v3 (broad) and v4 (narrow) orchestrators implement this.
    """

    @property
    def model_version(self) -> str:
        """Return the orchestrator version."""
        ...

    async def run_pipeline(self, config: PipelineConfig) -> PipelineResult:
        """Execute a configured pipeline.

        Args:
            config: Pipeline configuration specifying what to run.

        Returns:
            PipelineResult with all phase outputs.
        """
        ...
