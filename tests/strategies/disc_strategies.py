"""Hypothesis strategies for generating valid Poincare disc point sets.

Feature: poincare-visual-context
"""

from __future__ import annotations

import math
from typing import Any

from hypothesis import strategies as st
from hypothesis.strategies import SearchStrategy


@st.composite
def valid_disc_point(
    draw: st.DrawFn, curvature: float = 1.0
) -> tuple[float, float]:
    """Generate a single point strictly within the Poincare disc boundary."""
    max_radius = 1.0 / math.sqrt(curvature) - 0.01
    angle = draw(st.floats(min_value=0.0, max_value=2 * math.pi))
    radius = draw(st.floats(min_value=0.0, max_value=max_radius))
    x = radius * math.cos(angle)
    y = radius * math.sin(angle)
    return (x, y)


@st.composite
def valid_disc_points(
    draw: st.DrawFn,
    min_points: int = 10,
    max_points: int = 80,
    curvature: float = 1.0,
) -> tuple[list[tuple[str, float, float]], float]:
    """Generate a set of labeled points strictly within the Poincare disc."""
    n = draw(st.integers(min_value=min_points, max_value=max_points))
    max_radius = 1.0 / math.sqrt(curvature) - 0.01
    points: list[tuple[str, float, float]] = []
    for i in range(n):
        angle = draw(st.floats(min_value=0.0, max_value=2 * math.pi))
        radius = draw(st.floats(min_value=0.0, max_value=max_radius))
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        points.append((f"R{i}", x, y))
    return points, curvature


@st.composite
def valid_disc_point_pair(
    draw: st.DrawFn, curvature: float = 1.0
) -> tuple[tuple[float, float], tuple[float, float], float]:
    """Generate a pair of points on the disc plus curvature."""
    p = draw(valid_disc_point(curvature=curvature))
    q = draw(valid_disc_point(curvature=curvature))
    return p, q, curvature


@st.composite
def valid_disc_point_triple(
    draw: st.DrawFn, curvature: float = 1.0
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], float]:
    """Generate three points on the disc for triangle inequality testing."""
    p = draw(valid_disc_point(curvature=curvature))
    q = draw(valid_disc_point(curvature=curvature))
    r = draw(valid_disc_point(curvature=curvature))
    return p, q, r, curvature


def positive_curvature() -> SearchStrategy[float]:
    """Strategy for valid curvature values."""
    return st.floats(min_value=0.1, max_value=4.0)


@st.composite
def malformed_embedding_row(
    draw: st.DrawFn,
    curvature: float = 1.0,
) -> dict[str, Any]:
    """Governed embedding row that must not parse as a valid disc point."""
    kind = draw(
        st.sampled_from(
            (
                "missing",
                "nan",
                "inf",
                "out_of_ball",
                "bad_string",
                "single_coord",
                "empty_projections",
            )
        )
    )
    row: dict[str, Any] = {"residue_id": "test:A:1", "curvature": curvature}
    if kind == "missing":
        return row
    if kind == "nan":
        row["hyp_projection_2d"] = [float("nan"), 0.1]
        return row
    if kind == "inf":
        row["hyp_projection_2d"] = [float("inf"), 0.1]
        return row
    if kind == "out_of_ball":
        r_ball = 1.0 / math.sqrt(curvature)
        radius = draw(st.floats(min_value=r_ball + 0.05, max_value=r_ball + 2.0))
        angle = draw(st.floats(min_value=0.0, max_value=2 * math.pi))
        row["hyp_projection_2d"] = [radius * math.cos(angle), radius * math.sin(angle)]
        return row
    if kind == "bad_string":
        row["hyp_projection_2d"] = draw(
            st.sampled_from(["", "not-a-coordinate", "[]", "[1]", "nan,inf", "x,y"])
        )
        return row
    if kind == "single_coord":
        row["hyp_projection_2d"] = [draw(st.floats(min_value=-1.0, max_value=1.0))]
        return row
    row["hyp_projections"] = {}
    return row
