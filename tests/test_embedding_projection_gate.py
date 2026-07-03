"""Hyperbolic projection read-boundary gate — no silent origin coercion.

Property: malformed governed projections raise or quarantine; never clamp to
(0, 0) and report success. See docs/ENFORCEMENT_MATRIX.md (Coordinate coercion).
"""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from hypothesis import given, settings

from science.dtie.common.embedding_projection import (
    MalformedHyperbolicCoordinateError,
    parse_embedding_projection_xy,
    validate_disc_point_xy,
)
from tests.strategies.disc_strategies import (
    malformed_embedding_row,
    valid_disc_point,
)


@pytest.mark.parametrize(
    "row",
    [
        {},
        {"hyp_projection_2d": None, "hyp_projections": None},
        {"hyp_projection_2d": "not-a-coordinate", "curvature": 1.0},
        {"hyp_projection_2d": [float("nan"), 0.2], "curvature": 1.0},
        {"hyp_projection_2d": [0.2, float("inf")], "curvature": 1.0},
        {"hyp_projection_2d": [1.5, 0.0], "curvature": 1.0},
        {"hyp_projections": {"x": "bad", "y": 0.1}, "curvature": 1.0},
    ],
    ids=[
        "missing_columns",
        "null_columns",
        "unparseable_string",
        "nan_component",
        "inf_component",
        "out_of_ball",
        "non_numeric_dict",
    ],
)
def test_malformed_rows_raise_not_origin(row: dict[str, Any]) -> None:
    with pytest.raises(MalformedHyperbolicCoordinateError):
        parse_embedding_projection_xy(row, curvature=1.0)


@given(malformed_embedding_row(curvature=1.0))
@settings(max_examples=40)
def test_property_malformed_rows_never_silent_origin(row: dict[str, Any]) -> None:
    with pytest.raises(MalformedHyperbolicCoordinateError):
        parse_embedding_projection_xy(row, curvature=1.0)


@given(valid_disc_point(curvature=1.0))
@settings(max_examples=30)
def test_property_valid_disc_points_parse(point: tuple[float, float]) -> None:
    x, y = point
    row = {"hyp_projection_2d": [x, y], "curvature": 1.0}
    assert parse_embedding_projection_xy(row, curvature=1.0) == (x, y)


def test_explicit_origin_is_valid() -> None:
    row = {"hyp_projection_2d": [0.0, 0.0], "curvature": 1.0}
    assert parse_embedding_projection_xy(row, curvature=1.0) == (0.0, 0.0)


def test_string_projection_parses() -> None:
    row = {"hyp_projection_2d": "[0.12,-0.08]", "curvature": 1.0}
    assert parse_embedding_projection_xy(row, curvature=1.0) == pytest.approx((0.12, -0.08))


def test_hyp_projections_dict_fallback() -> None:
    row = {"hyp_projections": {"x": 0.15, "y": -0.05}, "curvature": 1.0}
    assert parse_embedding_projection_xy(row, curvature=1.0) == pytest.approx((0.15, -0.05))


def test_validate_disc_point_rejects_outside_ball() -> None:
    c = 0.605
    r_ball = 1.0 / math.sqrt(c)
    with pytest.raises(MalformedHyperbolicCoordinateError):
        validate_disc_point_xy(r_ball + 0.1, 0.0, curvature=c)


@pytest.mark.asyncio
async def test_hydration_quarantines_malformed_rows() -> None:
    from agent.coordinator.routers import dashboard as dash

    rows = [
        {
            "residue_id": "kras:A:1",
            "residue_index": 1,
            "residue_name": "MET",
            "chain_label": "A",
            "hyp_projection_2d": "[0.12,-0.08]",
            "hyp_projections": None,
            "cone_depth": 3.0,
            "epistemic_uncertainty": 0.4,
            "aleatoric_uncertainty": 0.1,
            "curvature": 1.0,
        },
        {
            "residue_id": "kras:A:2",
            "residue_index": 2,
            "residue_name": "GLU",
            "chain_label": "A",
            "hyp_projection_2d": None,
            "hyp_projections": None,
            "cone_depth": 2.5,
            "epistemic_uncertainty": 0.3,
            "aleatoric_uncertainty": 0.1,
            "curvature": 1.0,
        },
    ]

    with patch(
        "agent.tools.dtie.tools.ToolDB.fetch_all",
        new_callable=AsyncMock,
        return_value=rows,
    ):
        result = await dash._fetch_embeddings_for_hydration("kras", AsyncMock())

    assert result is not None
    assert len(result["residues"]) == 1
    assert result["residues"][0]["residue_id"] == "kras:A:1"
    assert (result["residues"][0]["x"], result["residues"][0]["y"]) != (0.0, 0.0)
    quarantined = result.get("projection_quarantined", [])
    assert len(quarantined) == 1
    assert quarantined[0]["residue_id"] == "kras:A:2"
