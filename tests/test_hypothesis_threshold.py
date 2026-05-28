"""Property-based tests for hypothesis threshold evaluation.

Feature: hypothesis-engine
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agent.tools.hypothesis.threshold import evaluate_threshold


# ---------------------------------------------------------------------------
# Property 5: Threshold evaluation correctness
# Feature: hypothesis-engine, Property 5: Threshold evaluation correctness
# ---------------------------------------------------------------------------


class TestThresholdEvaluation:
    """Property 5: Threshold evaluation correctness.

    For any numeric threshold expression and numeric result, the evaluation
    should return True if and only if the comparison holds. The evaluator
    should be consistent with Python's numeric comparison semantics.
    """

    @settings(max_examples=100)
    @given(
        metric_name=st.sampled_from(["value", "score", "betweenness", "count"]),
        value=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        threshold_val=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    )
    def test_greater_than(self, metric_name: str, value: float, threshold_val: float):
        """Feature: hypothesis-engine, Property 5: Threshold evaluation correctness

        **Validates: Requirements 2.2**
        """
        expr = f"{metric_name} > {threshold_val}"
        result = evaluate_threshold(expr, value)
        assert result == (value > threshold_val)

    @settings(max_examples=100)
    @given(
        metric_name=st.sampled_from(["value", "score", "betweenness", "count"]),
        value=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        threshold_val=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    )
    def test_less_than(self, metric_name: str, value: float, threshold_val: float):
        """Feature: hypothesis-engine, Property 5: Threshold evaluation correctness

        **Validates: Requirements 2.2**
        """
        expr = f"{metric_name} < {threshold_val}"
        result = evaluate_threshold(expr, value)
        assert result == (value < threshold_val)

    @settings(max_examples=100)
    @given(
        metric_name=st.sampled_from(["value", "score", "betweenness", "count"]),
        value=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        threshold_val=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    )
    def test_greater_equal(self, metric_name: str, value: float, threshold_val: float):
        """Feature: hypothesis-engine, Property 5: Threshold evaluation correctness

        **Validates: Requirements 2.2**
        """
        expr = f"{metric_name} >= {threshold_val}"
        result = evaluate_threshold(expr, value)
        assert result == (value >= threshold_val)

    @settings(max_examples=100)
    @given(
        metric_name=st.sampled_from(["value", "score", "betweenness", "count"]),
        value=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        threshold_val=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    )
    def test_less_equal(self, metric_name: str, value: float, threshold_val: float):
        """Feature: hypothesis-engine, Property 5: Threshold evaluation correctness

        **Validates: Requirements 2.2**
        """
        expr = f"{metric_name} <= {threshold_val}"
        result = evaluate_threshold(expr, value)
        assert result == (value <= threshold_val)

    @settings(max_examples=100)
    @given(
        metric_name=st.sampled_from(["value", "score", "betweenness", "count"]),
        value=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        threshold_val=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    )
    def test_not_equal(self, metric_name: str, value: float, threshold_val: float):
        """Feature: hypothesis-engine, Property 5: Threshold evaluation correctness

        **Validates: Requirements 2.2**
        """
        expr = f"{metric_name} != {threshold_val}"
        result = evaluate_threshold(expr, value)
        assert result == (value != threshold_val)

    def test_invalid_format_raises(self):
        """Invalid threshold format raises ValueError."""
        with pytest.raises(ValueError):
            evaluate_threshold("not a threshold", 5.0)

    def test_non_numeric_result_raises(self):
        """Non-numeric result raises ValueError."""
        with pytest.raises(ValueError):
            evaluate_threshold("value > 0.5", "not_a_number")
