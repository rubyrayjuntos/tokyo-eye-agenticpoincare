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


def test_curvature_hash_matches_ieee754_canonical():
    expected = "90383c360f60edddafebcb6c119b4b62366543eb733d7a9536bcb54d92c2e0e8"
    assert curvature_hash_for(CANONICAL_V6_CURVATURE) == expected


def test_curvature_hash_differs_from_repr_digest():
    """repr() hashes are not cross-Python stable — do not use for CI SSOT."""
    import hashlib

    repr_hash = hashlib.sha256(repr(CANONICAL_V6_CURVATURE).encode("ascii")).hexdigest()
    assert curvature_hash_for(CANONICAL_V6_CURVATURE) != repr_hash


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
