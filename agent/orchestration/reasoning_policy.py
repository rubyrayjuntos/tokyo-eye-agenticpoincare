# Backend Orchestration Layer — Reasoning Policy
"""Maps hypothesis lifecycle state to language guidance for the LLM.

Each lifecycle state determines:
- reasoning_mode: how the LLM should structure its response
- allowed_speech_acts: language patterns the LLM may use
- blocked_speech_acts: language patterns the LLM must avoid
- speech_style: overall tone descriptor
- language_guidance: explicit natural-language instruction for the system prompt
"""

from __future__ import annotations

from dataclasses import dataclass

from agent.orchestration.models import HypothesisLifecycleState


@dataclass(frozen=True)
class ReasoningPolicy:
    """Language policy derived from hypothesis lifecycle state."""

    reasoning_mode: str
    speech_style: str
    allowed_speech_acts: tuple[str, ...]
    blocked_speech_acts: tuple[str, ...]
    language_guidance: str


# ---------------------------------------------------------------------------
# Policy table: one entry per lifecycle state
# ---------------------------------------------------------------------------

_REASONING_POLICIES: dict[HypothesisLifecycleState, ReasoningPolicy] = {
    HypothesisLifecycleState.EMERGENT: ReasoningPolicy(
        reasoning_mode="explore",
        speech_style="exploratory",
        allowed_speech_acts=(
            "suggest",
            "hypothesize",
            "question",
            "observe",
        ),
        blocked_speech_acts=(
            "conclude",
            "recommend_treatment",
            "assert_mechanism",
        ),
        language_guidance=(
            "Use exploratory language: may, could, possible signal, worth testing. "
            "Do not assert mechanisms or make definitive claims."
        ),
    ),
    HypothesisLifecycleState.FRAMED: ReasoningPolicy(
        reasoning_mode="frame",
        speech_style="structured",
        allowed_speech_acts=(
            "hypothesize",
            "define_test",
            "identify_evidence",
            "propose_experiment",
        ),
        blocked_speech_acts=(
            "conclude",
            "recommend_treatment",
        ),
        language_guidance=(
            "Frame the hypothesis clearly. Identify what evidence would support or "
            "contradict it. Propose specific tests."
        ),
    ),
    HypothesisLifecycleState.TESTING: ReasoningPolicy(
        reasoning_mode="evaluate",
        speech_style="analytical",
        allowed_speech_acts=(
            "compare",
            "measure",
            "evaluate_evidence",
            "report_finding",
        ),
        blocked_speech_acts=(
            "conclude",
            "recommend_treatment",
            "assert_mechanism",
        ),
        language_guidance=(
            "Evaluate evidence objectively. Report findings without drawing "
            "premature conclusions. Compare against predictions."
        ),
    ),
    HypothesisLifecycleState.SUPPORTED: ReasoningPolicy(
        reasoning_mode="synthesize",
        speech_style="bounded_confident",
        allowed_speech_acts=(
            "conclude_bounded",
            "summarize_evidence",
            "identify_limitations",
            "suggest_next_steps",
        ),
        blocked_speech_acts=(
            "assert_absolute",
            "recommend_treatment",
        ),
        language_guidance=(
            "Use bounded confident language: consistent with, supported by current "
            "evidence, the data suggest. Always note limitations and remaining unknowns."
        ),
    ),
    HypothesisLifecycleState.CONTRADICTED: ReasoningPolicy(
        reasoning_mode="diagnose",
        speech_style="diagnostic",
        allowed_speech_acts=(
            "name_contradiction",
            "compare_expected_vs_actual",
            "recommend_regression",
            "suggest_revision",
        ),
        blocked_speech_acts=(
            "conclude",
            "assert_mechanism",
            "recommend_treatment",
        ),
        language_guidance=(
            "Explicitly name the contradiction. Explain what was expected vs observed. "
            "Recommend regression to an earlier phase or hypothesis revision."
        ),
    ),
    HypothesisLifecycleState.REVISED: ReasoningPolicy(
        reasoning_mode="refine",
        speech_style="iterative",
        allowed_speech_acts=(
            "revise_hypothesis",
            "identify_new_evidence",
            "propose_experiment",
            "compare_versions",
        ),
        blocked_speech_acts=(
            "conclude",
            "recommend_treatment",
        ),
        language_guidance=(
            "Present the revised hypothesis clearly. Explain what changed and why. "
            "Identify new tests needed to validate the revision."
        ),
    ),
    HypothesisLifecycleState.SYNTHESIZED: ReasoningPolicy(
        reasoning_mode="report",
        speech_style="authoritative",
        allowed_speech_acts=(
            "conclude",
            "summarize_evidence",
            "recommend_next_steps",
            "generate_report",
        ),
        blocked_speech_acts=(
            "recommend_treatment",
        ),
        language_guidance=(
            "Synthesize findings authoritatively. Present conclusions with full "
            "evidence chain. Recommend concrete next steps."
        ),
    ),
}


def reasoning_policy_for_lifecycle_state(
    state: HypothesisLifecycleState,
) -> ReasoningPolicy:
    """Return the reasoning/language policy for a given hypothesis lifecycle state.

    Every valid HypothesisLifecycleState has a corresponding policy entry.
    """
    return _REASONING_POLICIES[state]
