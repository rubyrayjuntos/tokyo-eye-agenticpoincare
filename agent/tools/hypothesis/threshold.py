"""Threshold evaluation for hypothesis predictions.

Parses and evaluates comparison expressions like "value > 0.15"
against numeric result values.
"""

from __future__ import annotations

import operator
import re
from typing import Any

# Supported comparison operators
_OPERATORS: dict[str, Any] = {
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}

# Pattern: optional_metric_name operator numeric_value (supports scientific notation)
_THRESHOLD_PATTERN = re.compile(
    r"^\s*\w*\s*(>=|<=|!=|==|>|<)\s*(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$"
)


def evaluate_threshold(threshold: str, result_value: Any) -> bool:
    """Evaluate a threshold expression against a result value.

    Supported formats:
    - "metric > 0.15"
    - "value < 0.5"
    - "count >= 10"
    - "score == 3"
    - "delta != 0"
    - "> 0.15" (no metric name)

    Returns True if the result passes the threshold.
    Raises ValueError if the threshold format is invalid or result is not numeric.
    """
    match = _THRESHOLD_PATTERN.match(threshold)
    if not match:
        raise ValueError(f"Invalid threshold format: '{threshold}'")

    op_str = match.group(1)
    threshold_value = float(match.group(2))

    try:
        numeric_result = float(result_value)
    except (TypeError, ValueError):
        raise ValueError(
            f"Result value '{result_value}' cannot be converted to a number"
        )

    op_func = _OPERATORS[op_str]
    return op_func(numeric_result, threshold_value)
