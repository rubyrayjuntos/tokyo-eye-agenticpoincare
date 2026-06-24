"""Compilation helpers for typed cross-agent orchestration state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from agent.models.orchestration_state import (
    CrossAgentConsensus,
    ResearcherOutput,
    RunContext,
    StepEventType,
    TemporalConsensusChain,
    TemporalStepState,
    WriterNarrative,
    MotifMetrics,
)


class OrchestrationCompileError(ValueError):
    """Raised when a cross-agent or temporal state fails typed validation."""

    def __init__(self, stage: str, error: ValidationError):
        self.stage = stage
        self.error = error
        super().__init__(f"{stage} compilation failed: {error}")


def _wrap_validation(stage: str, fn):
    try:
        return fn()
    except ValidationError as exc:
        raise OrchestrationCompileError(stage, exc) from exc


def compile_run_context(payload: RunContext | dict[str, Any]) -> RunContext:
    """Compile run-context payload into an immutable typed model."""
    return _wrap_validation(
        "run_context",
        lambda: payload if isinstance(payload, RunContext) else RunContext.model_validate(payload),
    )


def compile_consensus(
    *,
    run_context: RunContext | dict[str, Any],
    motif_interpreter_state: MotifMetrics | dict[str, Any],
    writer_state: WriterNarrative | dict[str, Any],
    researcher_state: ResearcherOutput | dict[str, Any],
) -> CrossAgentConsensus:
    """Compile one orchestrator-step consensus payload."""
    return _wrap_validation(
        "cross_agent_consensus",
        lambda: CrossAgentConsensus.model_validate(
            {
                "run_context": run_context,
                "motif_interpreter_state": motif_interpreter_state,
                "writer_state": writer_state,
                "researcher_state": researcher_state,
            }
        ),
    )


def compile_temporal_step(
    *,
    step_index: int,
    consensus: CrossAgentConsensus | dict[str, Any],
    event_type: StepEventType = StepEventType.INITIAL,
    active_contradiction_ids: set[str] | frozenset[str] | tuple[str, ...] = (),
    resolved_contradiction_ids: set[str] | frozenset[str] | tuple[str, ...] = (),
) -> TemporalStepState:
    """Compile one temporal step snapshot."""
    return _wrap_validation(
        "temporal_step",
        lambda: TemporalStepState.model_validate(
            {
                "step_index": step_index,
                "event_type": event_type,
                "consensus": consensus,
                "active_contradiction_ids": frozenset(active_contradiction_ids),
                "resolved_contradiction_ids": frozenset(resolved_contradiction_ids),
            }
        ),
    )


def compile_temporal_chain(
    steps: list[TemporalStepState | dict[str, Any]] | tuple[TemporalStepState | dict[str, Any], ...]
) -> TemporalConsensusChain:
    """Compile and validate a temporal chain across multiple orchestrator steps."""
    return _wrap_validation(
        "temporal_chain",
        lambda: TemporalConsensusChain.model_validate({"steps": steps}),
    )


def export_orchestration_state(chain: TemporalConsensusChain) -> dict[str, Any]:
    """Serialize the validated temporal chain into a carry-forward context payload."""
    latest = chain.steps[-1]
    consensus = latest.consensus
    return {
        "run_context": consensus.run_context.model_dump(mode="json"),
        "motif_interpreter_state": consensus.motif_interpreter_state.model_dump(mode="json"),
        "writer_state": consensus.writer_state.model_dump(mode="json"),
        "researcher_state": consensus.researcher_state.model_dump(mode="json"),
        "event_type": latest.event_type.value,
        "active_contradiction_ids": sorted(latest.active_contradiction_ids),
        "resolved_contradiction_ids": sorted(latest.resolved_contradiction_ids),
        "temporal_steps": [step.model_dump(mode="json") for step in chain.steps],
    }


@dataclass
class TemporalConsensusBuilder:
    """Mutable builder for orchestrator step compilation with temporal validation."""

    run_context: RunContext | dict[str, Any]
    _steps: list[TemporalStepState] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.run_context = compile_run_context(self.run_context)

    @property
    def steps(self) -> tuple[TemporalStepState, ...]:
        return tuple(self._steps)

    @property
    def current_step(self) -> TemporalStepState | None:
        return self._steps[-1] if self._steps else None

    def current_chain(self) -> TemporalConsensusChain | None:
        if not self._steps:
            return None
        return compile_temporal_chain(self._steps)

    def export_state(self) -> dict[str, Any] | None:
        chain = self.current_chain()
        if chain is None:
            return None
        return export_orchestration_state(chain)

    def append(
        self,
        *,
        motif_interpreter_state: MotifMetrics | dict[str, Any],
        writer_state: WriterNarrative | dict[str, Any],
        researcher_state: ResearcherOutput | dict[str, Any],
        event_type: StepEventType | None = None,
        active_contradiction_ids: set[str] | frozenset[str] | tuple[str, ...] = (),
        resolved_contradiction_ids: set[str] | frozenset[str] | tuple[str, ...] = (),
    ) -> TemporalStepState:
        """Append a validated step and recompile the full temporal chain."""
        consensus = compile_consensus(
            run_context=self.run_context,
            motif_interpreter_state=motif_interpreter_state,
            writer_state=writer_state,
            researcher_state=researcher_state,
        )
        step = compile_temporal_step(
            step_index=len(self._steps),
            consensus=consensus,
            event_type=event_type or (StepEventType.INITIAL if not self._steps else StepEventType.GOVERNANCE_REVIEW),
            active_contradiction_ids=active_contradiction_ids,
            resolved_contradiction_ids=resolved_contradiction_ids,
        )
        candidate_steps = [*self._steps, step]
        compile_temporal_chain(candidate_steps)
        self._steps = candidate_steps
        return step

    def append_writer_revision(
        self,
        *,
        motif_interpreter_state: MotifMetrics | dict[str, Any],
        writer_state: WriterNarrative | dict[str, Any],
        researcher_state: ResearcherOutput | dict[str, Any],
        resolved_contradiction_ids: set[str] | frozenset[str] | tuple[str, ...] = (),
    ) -> TemporalStepState:
        return self.append(
            motif_interpreter_state=motif_interpreter_state,
            writer_state=writer_state,
            researcher_state=researcher_state,
            event_type=StepEventType.WRITER_REVISION,
            resolved_contradiction_ids=resolved_contradiction_ids,
        )

    def append_researcher_revision(
        self,
        *,
        motif_interpreter_state: MotifMetrics | dict[str, Any],
        writer_state: WriterNarrative | dict[str, Any],
        researcher_state: ResearcherOutput | dict[str, Any],
        resolved_contradiction_ids: set[str] | frozenset[str] | tuple[str, ...] = (),
    ) -> TemporalStepState:
        return self.append(
            motif_interpreter_state=motif_interpreter_state,
            writer_state=writer_state,
            researcher_state=researcher_state,
            event_type=StepEventType.RESEARCHER_REVISION,
            resolved_contradiction_ids=resolved_contradiction_ids,
        )

    def append_recluster(
        self,
        *,
        motif_interpreter_state: MotifMetrics | dict[str, Any],
        writer_state: WriterNarrative | dict[str, Any],
        researcher_state: ResearcherOutput | dict[str, Any],
        resolved_contradiction_ids: set[str] | frozenset[str] | tuple[str, ...] = (),
    ) -> TemporalStepState:
        return self.append(
            motif_interpreter_state=motif_interpreter_state,
            writer_state=writer_state,
            researcher_state=researcher_state,
            event_type=StepEventType.RECLUSTER,
            resolved_contradiction_ids=resolved_contradiction_ids,
        )

    def append_reinference(
        self,
        *,
        motif_interpreter_state: MotifMetrics | dict[str, Any],
        writer_state: WriterNarrative | dict[str, Any],
        researcher_state: ResearcherOutput | dict[str, Any],
        resolved_contradiction_ids: set[str] | frozenset[str] | tuple[str, ...] = (),
    ) -> TemporalStepState:
        return self.append(
            motif_interpreter_state=motif_interpreter_state,
            writer_state=writer_state,
            researcher_state=researcher_state,
            event_type=StepEventType.REINFERENCE,
            resolved_contradiction_ids=resolved_contradiction_ids,
        )
