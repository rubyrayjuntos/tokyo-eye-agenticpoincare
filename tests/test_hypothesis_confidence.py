"""Property-based tests for hypothesis confidence calculation, decay, and status transitions.

Feature: hypothesis-engine
"""

from __future__ import annotations

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from agent.tools.hypothesis.confidence import (
    apply_decay,
    calculate_confidence,
    determine_status,
)
from agent.tools.hypothesis.models import Evidence, HypothesisStatus


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@st.composite
def evidence_strategy(draw):
    """Generate a single Evidence object."""
    return Evidence(
        source_tool=draw(st.sampled_from(["get_graph_metrics", "get_source_leaks", "manual"])),
        supports=draw(st.booleans()),
        strength=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        description=draw(st.text(min_size=1, max_size=50)),
    )


@st.composite
def evidence_list_strategy(draw, min_size=0, max_size=10):
    """Generate a list of Evidence objects."""
    return draw(st.lists(evidence_strategy(), min_size=min_size, max_size=max_size))


# ---------------------------------------------------------------------------
# Property 2: Confidence calculation formula
# Feature: hypothesis-engine, Property 2: Confidence calculation formula
# ---------------------------------------------------------------------------


class TestConfidenceCalculation:
    """Property 2: Confidence calculation formula.

    For any list of Evidence objects, the confidence should equal
    sum(e.strength for supporting) / (sum(e.strength for supporting) + sum(e.strength for contradicting)).
    When the list is empty or total strength is zero, confidence should be 0.5.
    """

    @settings(max_examples=100)
    @given(evidence=evidence_list_strategy())
    def test_confidence_matches_formula(self, evidence: list[Evidence]):
        """Feature: hypothesis-engine, Property 2: Confidence calculation formula

        **Validates: Requirements 3.3, 7.1, 7.2, 7.3**
        """
        result = calculate_confidence(evidence)

        # Independently compute expected value
        supporting = sum(e.strength for e in evidence if e.supports)
        contradicting = sum(e.strength for e in evidence if not e.supports)
        total = supporting + contradicting

        if not evidence or total == 0.0:
            assert result == 0.5
        else:
            expected = supporting / total
            assert abs(result - expected) < 1e-10

    @settings(max_examples=100)
    @given(evidence=evidence_list_strategy())
    def test_confidence_in_bounds(self, evidence: list[Evidence]):
        """Confidence is always in [0.0, 1.0]."""
        result = calculate_confidence(evidence)
        assert 0.0 <= result <= 1.0

    def test_empty_evidence_returns_half(self):
        """Empty evidence list returns 0.5."""
        assert calculate_confidence([]) == 0.5


# ---------------------------------------------------------------------------
# Property 3: Status transitions based on confidence and evidence count
# Feature: hypothesis-engine, Property 3: Status transitions
# ---------------------------------------------------------------------------


class TestStatusTransitions:
    """Property 3: Status transitions based on confidence and evidence count.

    For any hypothesis with confidence > 0.7 and at least 2 evidence records,
    the status should be `supported`. For confidence < 0.3 and at least 2 evidence,
    the status should be `contradicted`. For fewer than 2 evidence records,
    the status should not transition regardless of confidence.
    """

    @settings(max_examples=100)
    @given(
        confidence=st.floats(min_value=0.701, max_value=1.0, allow_nan=False, allow_infinity=False),
        evidence_count=st.integers(min_value=2, max_value=100),
        current_status=st.sampled_from(list(HypothesisStatus)),
    )
    def test_high_confidence_becomes_supported(
        self, confidence: float, evidence_count: int, current_status: HypothesisStatus
    ):
        """Feature: hypothesis-engine, Property 3: Status transitions

        **Validates: Requirements 4.1, 4.2**
        """
        result = determine_status(confidence, evidence_count, current_status)
        assert result == HypothesisStatus.SUPPORTED

    @settings(max_examples=100)
    @given(
        confidence=st.floats(min_value=0.0, max_value=0.299, allow_nan=False, allow_infinity=False),
        evidence_count=st.integers(min_value=2, max_value=100),
        current_status=st.sampled_from(list(HypothesisStatus)),
    )
    def test_low_confidence_becomes_contradicted(
        self, confidence: float, evidence_count: int, current_status: HypothesisStatus
    ):
        """Feature: hypothesis-engine, Property 3: Status transitions

        **Validates: Requirements 4.1, 4.2**
        """
        result = determine_status(confidence, evidence_count, current_status)
        assert result == HypothesisStatus.CONTRADICTED

    @settings(max_examples=100)
    @given(
        confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        evidence_count=st.integers(min_value=0, max_value=1),
        current_status=st.sampled_from(list(HypothesisStatus)),
    )
    def test_insufficient_evidence_no_transition(
        self, confidence: float, evidence_count: int, current_status: HypothesisStatus
    ):
        """Feature: hypothesis-engine, Property 3: Status transitions

        **Validates: Requirements 4.1, 4.2**
        """
        result = determine_status(confidence, evidence_count, current_status)
        assert result == current_status


# ---------------------------------------------------------------------------
# Property 4: Confidence decay toward 0.5 over time
# Feature: hypothesis-engine, Property 4: Confidence decay
# ---------------------------------------------------------------------------


class TestConfidenceDecay:
    """Property 4: Confidence decay toward 0.5 over time.

    For any hypothesis confidence value and number of stale days > 0,
    the decayed confidence should be closer to 0.5 than the original.
    The decay formula should be monotonically approaching 0.5.
    """

    @settings(max_examples=100)
    @given(
        confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        days_stale=st.integers(min_value=1, max_value=365),
    )
    def test_decay_moves_toward_half(self, confidence: float, days_stale: int):
        """Feature: hypothesis-engine, Property 4: Confidence decay toward 0.5 over time

        **Validates: Requirements 4.3**
        """
        result = apply_decay(confidence, days_stale)

        # Result should be closer to 0.5 than original (or equal if already 0.5)
        original_distance = abs(confidence - 0.5)
        result_distance = abs(result - 0.5)
        assert result_distance <= original_distance + 1e-10

    @settings(max_examples=100)
    @given(
        confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        days_a=st.integers(min_value=1, max_value=100),
        days_b=st.integers(min_value=1, max_value=100),
    )
    def test_decay_monotonic(self, confidence: float, days_a: int, days_b: int):
        """More days → closer to 0.5 (monotonic approach)."""
        assume(days_a < days_b)
        result_a = apply_decay(confidence, days_a)
        result_b = apply_decay(confidence, days_b)

        dist_a = abs(result_a - 0.5)
        dist_b = abs(result_b - 0.5)
        assert dist_b <= dist_a + 1e-10

    @settings(max_examples=100)
    @given(
        confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    def test_zero_days_no_change(self, confidence: float):
        """Zero stale days should not change confidence."""
        assert apply_decay(confidence, 0) == confidence
