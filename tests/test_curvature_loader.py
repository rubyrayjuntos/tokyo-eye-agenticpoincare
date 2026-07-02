"""Tests for curvature_loader SSOT module."""

from __future__ import annotations

import pytest

from science.dtie.common.curvature_loader import (
    CANONICAL_V6_CURVATURE,
    CurvatureSSOTError,
    curvature_hash_for,
    get_curvature,
    normalize_space_key,
)


def test_curvature_hash_matches_migration_population():
    expected = "c0c09c6e43aeb6e8ce097f4f301fbb7ff5884acdd754682699774d4f9967c6ca"
    assert curvature_hash_for(CANONICAL_V6_CURVATURE) == expected


def test_normalize_space_key():
    assert normalize_space_key("space_gospconemapper_v6_hyp128") == "gospconemapper_v6_hyp128"
    assert normalize_space_key("gospconemapper_v6_hyp128") == "gospconemapper_v6_hyp128"


@pytest.mark.asyncio
async def test_get_curvature_verifies_hash_mismatch_raises():
    class _FakeDB:
        async def fetch_one(self, *_args, **_kwargs):
            return {
                "curvature": CANONICAL_V6_CURVATURE,
                "curvature_hash": "deadbeef",
                "name": "gospconemapper_v6_hyp128",
                "space_id": "space_gospconemapper_v6_hyp128",
            }

    with pytest.raises(CurvatureSSOTError, match="hash mismatch"):
        await get_curvature("gospconemapper_v6_hyp128", db=_FakeDB())
    class _FakeDB:
        async def fetch_one(self, *_args, **_kwargs):
            return None

    with pytest.raises(CurvatureSSOTError):
        await get_curvature("nonexistent_space_xyz", db=_FakeDB())
