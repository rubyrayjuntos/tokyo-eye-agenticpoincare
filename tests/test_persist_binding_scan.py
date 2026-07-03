"""Tests for binding-site scan persistence normalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from science.compute.persist_binding_scan import persist_binding_site_scan


@dataclass
class _Candidate:
    site_id: str
    residue_ids: list[str]
    centroid_xyz: tuple[float, float, float]
    site_type: str
    discovery_method: str
    druggability_score: float
    site_rank: int = 1
    composite_gnn_score: float | None = None
    fpocket_druggability: float | None = None
    volume_angstrom3: float | None = None
    provenance_gate: str | None = None
    heuristic_version: str | None = None


class _FakeNormalizer:
    def __init__(self) -> None:
        self.payload = None

    async def normalize_binding_site_scan(self, payload: Any) -> None:
        self.payload = payload


@pytest.mark.asyncio
async def test_persist_binding_scan_coerces_partial_residue_ids(monkeypatch):
    captured: dict[str, Any] = {}

    class _Normalizer:
        async def normalize_binding_site_scan(self, payload: Any) -> None:
            captured["payload"] = payload

    monkeypatch.setattr(
        "science.compute.persist_binding_scan.Normalizer",
        lambda db, caller_identity: _Normalizer(),
    )

    await persist_binding_site_scan(
        db=object(),
        run_id="scan_run_1",
        structure_id="11qe",
        candidates=[
            _Candidate(
                site_id="site_1",
                residue_ids=["11qe:A:12", "A:379"],
                centroid_xyz=(1.0, 2.0, 3.0),
                site_type="surface_pocket",
                discovery_method="hybrid",
                druggability_score=0.5,
            )
        ],
        heuristic_version="v1.0",
        model_version="binding-scan-v1",
        scan_parameters={"eps_angstrom": 8.0},
        sites_found=1,
        duration_ms=10,
        status="complete",
    )

    payload = captured["payload"]
    assert payload.sites[0].residue_ids == ["11qe:A:12", "11qe:A:379"]
