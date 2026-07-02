"""Tests for curvature four-source probe."""

from __future__ import annotations

from science.dtie.common.curvature_registry import (
    CANONICAL_TS002_CURVATURE,
    CANONICAL_V6_CURVATURE,
    curvature_bucket,
    curvature_match,
)


def test_curvature_match_ts002_deprecated():
    assert curvature_match(0.6054343, CANONICAL_TS002_CURVATURE)
    assert curvature_bucket(CANONICAL_TS002_CURVATURE) == "ts002_deprecated"


def test_curvature_match_v6_pin():
    assert curvature_match(CANONICAL_V6_CURVATURE, CANONICAL_V6_CURVATURE)
    assert curvature_bucket(CANONICAL_V6_CURVATURE) == "v6_lever_a_canonical"


def test_curvature_mismatch_detected():
    assert not curvature_match(0.694913, CANONICAL_TS002_CURVATURE)
    assert curvature_bucket(0.694913) == "mid_c_band"
