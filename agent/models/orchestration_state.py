"""Typed orchestration state models for cross-agent and temporal validation.

These models provide a high-assurance state contract for multi-agent scientific
reasoning. They are designed to make cross-agent contradictions and temporal
drift explicit validation failures instead of silent orchestration mistakes.
"""

from __future__ import annotations

import math
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

_DEEP_CONE_DEPTH_THRESHOLD = 8.0
_LOW_STRAIN_UNCERTAINTY_THRESHOLD = 1.0
_FLOAT_TOLERANCE = 1e-9


class DepthClassification(str, Enum):
    BURIED = "buried"
    SURFACE = "surface"
    SHALLOW = "shallow"


class StrainState(str, Enum):
    HIGH_STRAIN = "high_strain"
    RELAXED = "relaxed"


class StepEventType(str, Enum):
    INITIAL = "initial"
    WRITER_REVISION = "writer_revision"
    RESEARCHER_REVISION = "researcher_revision"
    RECLUSTER = "recluster"
    REINFERENCE = "reinference"
    GOVERNANCE_REVIEW = "governance_review"


class RunContext(BaseModel):
    """Immutable world-model context for an orchestration run."""

    run_context_id: str = Field(..., min_length=1)
    space_id: str = Field(..., min_length=1)
    curvature: float = Field(..., gt=0)
    atlas_version: str = Field(..., min_length=1)
    motif_id: str | None = Field(None, min_length=1)

    model_config = ConfigDict(frozen=True)


class MotifMetrics(BaseModel):
    """Structured motif state emitted by the motif interpreter."""

    medoid_residue_id: str = Field(..., min_length=1)
    cluster_residue_ids: frozenset[str] = Field(..., min_length=1)
    max_cone_depth: float = Field(..., ge=0)
    mean_epistemic_uncertainty: float = Field(..., ge=0)

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def enforce_medoid_validity(self) -> "MotifMetrics":
        if self.medoid_residue_id not in self.cluster_residue_ids:
            raise ValueError(
                "Motif validity error: medoid_residue_id must be present in cluster_residue_ids."
            )
        return self


class WriterNarrative(BaseModel):
    """Structured claims extracted from the writer agent output."""

    narrative_text: str = Field(..., min_length=1)
    claimed_depth_classification: DepthClassification
    claimed_strain_state: StrainState

    model_config = ConfigDict(frozen=True)


class ResearcherOutput(BaseModel):
    """Structured claims extracted from the researcher agent output."""

    cited_literature_dois: tuple[str, ...] = Field(default_factory=tuple)
    mapped_residue_ids: frozenset[str] = Field(default_factory=frozenset)

    model_config = ConfigDict(frozen=True)


class CrossAgentConsensus(BaseModel):
    """Cross-agent contradiction compiler for a single orchestrator step."""

    run_context: RunContext
    motif_interpreter_state: MotifMetrics
    writer_state: WriterNarrative
    researcher_state: ResearcherOutput

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def enforce_geometry_narrative_alignment(self) -> "CrossAgentConsensus":
        motif = self.motif_interpreter_state
        writer = self.writer_state

        if (
            motif.max_cone_depth > _DEEP_CONE_DEPTH_THRESHOLD
            and writer.claimed_depth_classification in {DepthClassification.SURFACE, DepthClassification.SHALLOW}
        ):
            raise ValueError(
                "Narrative-Geometry Contradiction: the motif is deeply buried by max_cone_depth, "
                "but the writer classified it as surface/shallow. Revise the narrative to match the geometry."
            )

        if (
            motif.mean_epistemic_uncertainty <= _LOW_STRAIN_UNCERTAINTY_THRESHOLD
            and writer.claimed_strain_state == StrainState.HIGH_STRAIN
        ):
            raise ValueError(
                "Narrative-Geometry Contradiction: the motif has low mean_epistemic_uncertainty, "
                "but the writer claims high structural strain. Remove unsupported strain claims."
            )

        return self

    @model_validator(mode="after")
    def enforce_interpretation_coherence(self) -> "CrossAgentConsensus":
        motif = self.motif_interpreter_state
        researcher = self.researcher_state

        invalid_residues = researcher.mapped_residue_ids - motif.cluster_residue_ids
        if invalid_residues:
            raise ValueError(
                "Cross-Agent Coherence Error: researcher residue mappings extend outside the active motif "
                f"cluster: {sorted(invalid_residues)}."
            )

        return self


class TemporalStepState(BaseModel):
    """Validated snapshot of one orchestrator step in a discovery run."""

    step_index: int = Field(..., ge=0)
    event_type: StepEventType = StepEventType.INITIAL
    consensus: CrossAgentConsensus
    active_contradiction_ids: frozenset[str] = Field(default_factory=frozenset)
    resolved_contradiction_ids: frozenset[str] = Field(default_factory=frozenset)

    model_config = ConfigDict(frozen=True)


class TemporalConsensusChain(BaseModel):
    """Temporal state machine enforcing coherence across orchestrator steps."""

    steps: tuple[TemporalStepState, ...] = Field(..., min_length=1)

    @model_validator(mode="after")
    def enforce_temporal_invariants(self) -> "TemporalConsensusChain":
        previous = self.steps[0]

        for step in self.steps[1:]:
            if step.step_index <= previous.step_index:
                raise ValueError("Temporal integrity error: step_index values must increase strictly.")

            self._enforce_run_context(previous, step)
            self._enforce_motif_stability(previous, step)
            self._enforce_writer_stability(previous, step)
            self._enforce_researcher_stability(previous, step)
            self._enforce_epistemic_stability(previous, step)
            self._enforce_negative_memory(previous, step)
            previous = step

        return self

    @staticmethod
    def _enforce_run_context(previous: TemporalStepState, step: TemporalStepState) -> None:
        prev_ctx = previous.consensus.run_context
        cur_ctx = step.consensus.run_context
        if prev_ctx.run_context_id != cur_ctx.run_context_id:
            raise ValueError("Temporal run-context integrity error: run_context_id changed across steps.")
        if prev_ctx.space_id != cur_ctx.space_id:
            raise ValueError("Temporal provenance error: space_id changed across steps.")
        if prev_ctx.atlas_version != cur_ctx.atlas_version:
            raise ValueError("Temporal provenance error: atlas_version changed across steps.")
        if not math.isclose(prev_ctx.curvature, cur_ctx.curvature, abs_tol=_FLOAT_TOLERANCE):
            raise ValueError("Temporal provenance error: curvature changed across steps.")

    @staticmethod
    def _enforce_motif_stability(previous: TemporalStepState, step: TemporalStepState) -> None:
        prev_motif = previous.consensus.motif_interpreter_state
        cur_motif = step.consensus.motif_interpreter_state
        if step.event_type != StepEventType.RECLUSTER:
            if prev_motif.medoid_residue_id != cur_motif.medoid_residue_id:
                raise ValueError(
                    "Temporal geometry error: medoid changed without a recluster event."
                )
            if prev_motif.cluster_residue_ids != cur_motif.cluster_residue_ids:
                raise ValueError(
                    "Temporal residue-set stability error: cluster_residue_ids changed without a recluster event."
                )
            if not math.isclose(prev_motif.max_cone_depth, cur_motif.max_cone_depth, abs_tol=_FLOAT_TOLERANCE):
                raise ValueError(
                    "Temporal geometry error: max_cone_depth changed without a recluster event."
                )

    @staticmethod
    def _enforce_writer_stability(previous: TemporalStepState, step: TemporalStepState) -> None:
        prev_writer = previous.consensus.writer_state
        cur_writer = step.consensus.writer_state
        if step.event_type not in {StepEventType.WRITER_REVISION, StepEventType.RECLUSTER}:
            if prev_writer.claimed_depth_classification != cur_writer.claimed_depth_classification:
                raise ValueError(
                    "Temporal narrative alignment error: writer depth classification changed without a writer revision event."
                )
            if prev_writer.claimed_strain_state != cur_writer.claimed_strain_state:
                raise ValueError(
                    "Temporal narrative alignment error: writer strain classification changed without a writer revision event."
                )

    @staticmethod
    def _enforce_researcher_stability(previous: TemporalStepState, step: TemporalStepState) -> None:
        prev_researcher = previous.consensus.researcher_state
        cur_researcher = step.consensus.researcher_state
        if step.event_type not in {StepEventType.RESEARCHER_REVISION, StepEventType.RECLUSTER}:
            if prev_researcher.mapped_residue_ids != cur_researcher.mapped_residue_ids:
                raise ValueError(
                    "Temporal citation coherence error: mapped_residue_ids changed without a researcher revision event."
                )
            if prev_researcher.cited_literature_dois != cur_researcher.cited_literature_dois:
                raise ValueError(
                    "Temporal citation coherence error: cited_literature_dois changed without a researcher revision event."
                )

    @staticmethod
    def _enforce_epistemic_stability(previous: TemporalStepState, step: TemporalStepState) -> None:
        prev_motif = previous.consensus.motif_interpreter_state
        cur_motif = step.consensus.motif_interpreter_state
        if (
            step.event_type != StepEventType.REINFERENCE
            and not math.isclose(
                prev_motif.mean_epistemic_uncertainty,
                cur_motif.mean_epistemic_uncertainty,
                abs_tol=_FLOAT_TOLERANCE,
            )
        ):
            raise ValueError(
                "Temporal epistemic stability error: mean_epistemic_uncertainty changed without a reinference event."
            )

    @staticmethod
    def _enforce_negative_memory(previous: TemporalStepState, step: TemporalStepState) -> None:
        reintroduced = previous.resolved_contradiction_ids.intersection(step.active_contradiction_ids)
        if reintroduced:
            raise ValueError(
                "Temporal contradiction memory error: previously resolved contradictions reappeared: "
                f"{sorted(reintroduced)}."
            )
