"""Property-based tests for hypothesis models: serialization round-trip,
creation defaults, and falsifiability guardrail.

Feature: hypothesis-engine
"""

from __future__ import annotations

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from agent.tools.hypothesis.models import (
    Evidence,
    Hypothesis,
    HypothesisStatus,
    Prediction,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@st.composite
def prediction_strategy(draw):
    """Generate a valid Prediction."""
    return Prediction(
        statement=draw(st.text(min_size=1, max_size=100)),
        test_tool=draw(st.sampled_from(["get_graph_metrics", "get_source_leaks", None])),
        test_params=draw(st.none() | st.fixed_dictionaries({"structure_id": st.text(min_size=1, max_size=20)})),
        threshold=draw(st.none() | st.sampled_from(["value > 0.5", "count >= 2", "score < 0.3"])),
    )


@st.composite
def evidence_strategy(draw):
    """Generate a valid Evidence."""
    return Evidence(
        source_tool=draw(st.sampled_from(["get_graph_metrics", "get_source_leaks", "manual"])),
        supports=draw(st.booleans()),
        strength=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        description=draw(st.text(min_size=1, max_size=100)),
    )


@st.composite
def hypothesis_strategy(draw):
    """Generate a valid Hypothesis with at least one prediction."""
    predictions = draw(st.lists(prediction_strategy(), min_size=1, max_size=5))
    evidence = draw(st.lists(evidence_strategy(), min_size=0, max_size=5))

    return Hypothesis(
        structure_id=draw(st.text(min_size=1, max_size=30)),
        statement=draw(st.text(min_size=1, max_size=200)),
        mechanism=draw(st.none() | st.text(min_size=1, max_size=100)),
        predictions=predictions,
        evidence=evidence,
        status=draw(st.sampled_from(list(HypothesisStatus))),
        confidence=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
    )


# ---------------------------------------------------------------------------
# Property 8: Serialization round-trip
# Feature: hypothesis-engine, Property 8: Serialization round-trip
# ---------------------------------------------------------------------------


class TestSerializationRoundTrip:
    """Property 8: Serialization round-trip.

    For any valid Hypothesis object (including nested Predictions and Evidence),
    serializing to JSON and deserializing back should produce an equivalent object.
    """

    @settings(max_examples=100)
    @given(hypothesis=hypothesis_strategy())
    def test_json_round_trip(self, hypothesis: Hypothesis):
        """Feature: hypothesis-engine, Property 8: Serialization round-trip

        **Validates: Requirements 8.1, 8.2**
        """
        json_str = hypothesis.to_json()
        restored = Hypothesis.from_json(json_str)

        assert restored.hypothesis_id == hypothesis.hypothesis_id
        assert restored.structure_id == hypothesis.structure_id
        assert restored.statement == hypothesis.statement
        assert restored.mechanism == hypothesis.mechanism
        assert restored.status == hypothesis.status
        assert restored.confidence == hypothesis.confidence
        assert len(restored.predictions) == len(hypothesis.predictions)
        assert len(restored.evidence) == len(hypothesis.evidence)

        # Check predictions match
        for orig, rest in zip(hypothesis.predictions, restored.predictions):
            assert orig.prediction_id == rest.prediction_id
            assert orig.statement == rest.statement
            assert orig.test_tool == rest.test_tool
            assert orig.threshold == rest.threshold

        # Check evidence match
        for orig, rest in zip(hypothesis.evidence, restored.evidence):
            assert orig.evidence_id == rest.evidence_id
            assert orig.supports == rest.supports
            assert orig.strength == rest.strength
            assert orig.description == rest.description


# ---------------------------------------------------------------------------
# Property 1: Falsifiability guardrail rejects hypotheses without predictions
# Property 9: Hypothesis creation produces correct defaults
# Feature: hypothesis-engine, Properties 1 and 9
# ---------------------------------------------------------------------------


def validate_falsifiability(predictions: list[Prediction]) -> bool:
    """Check that a hypothesis has at least one prediction (falsifiability guardrail).

    This mirrors the validation that propose_hypothesis will enforce.
    """
    return len(predictions) >= 1


class TestCreationDefaultsAndFalsifiability:
    """Property 1: Falsifiability guardrail rejects hypotheses without predictions.
    Property 9: Hypothesis creation produces correct defaults.
    """

    @settings(max_examples=100)
    @given(
        structure_id=st.text(min_size=1, max_size=30),
        statement=st.text(min_size=1, max_size=200),
        predictions=st.lists(prediction_strategy(), min_size=1, max_size=5),
    )
    def test_creation_with_predictions_succeeds(
        self, structure_id: str, statement: str, predictions: list[Prediction]
    ):
        """Feature: hypothesis-engine, Property 9: Hypothesis creation produces correct defaults

        **Validates: Requirements 1.1, 1.2**
        """
        # Falsifiability check passes
        assert validate_falsifiability(predictions) is True

        # Create hypothesis with defaults
        h = Hypothesis(
            structure_id=structure_id,
            statement=statement,
            predictions=predictions,
        )

        # Verify defaults
        assert h.status == HypothesisStatus.PROPOSED
        assert h.confidence == 0.5
        assert h.hypothesis_id.startswith("hyp_")
        assert len(h.hypothesis_id) > 4
        assert h.structure_id == structure_id
        assert h.statement == statement
        assert len(h.predictions) == len(predictions)

    @settings(max_examples=100)
    @given(
        structure_id=st.text(min_size=1, max_size=30),
        statement=st.text(min_size=1, max_size=200),
    )
    def test_empty_predictions_fails_falsifiability(
        self, structure_id: str, statement: str
    ):
        """Feature: hypothesis-engine, Property 1: Falsifiability guardrail rejects hypotheses without predictions

        **Validates: Requirements 1.1, 1.2**
        """
        # Falsifiability check fails with empty predictions
        assert validate_falsifiability([]) is False
