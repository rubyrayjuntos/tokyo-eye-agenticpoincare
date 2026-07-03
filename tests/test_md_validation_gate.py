"""MD validation integrity gate — synthetic SMD must not set passed status.

Property: ``md_validation_status='passed'`` only after authentic SMD output.
See docs/ENFORCEMENT_MATRIX.md (MD validation gate).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from science.compute.cryptic.md_validate import (
    smd_result_qualifies_for_passed,
    validate_site_md_in_process,
)
from science.compute.runners.base import JobRunContext
from science.compute.runners.md_validate_top_n import run_md_validate_top_n

_AUTHENTIC_SMD = {
    "success": True,
    "status": "passed",
    "work_kcal_mol": 12.4,
    "strain_delta": -2.1,
    "duration_ms": 45_000,
    "notes": "Three-phase SMD complete. 120 force samples collected.",
}


class _CrypticSiteDB:
    """Minimal fake DB tracking fact_cryptic_site md_validation_status."""

    def __init__(self, site: dict[str, Any]) -> None:
        self.site = dict(site)

    async def fetch_one(self, query: str, params: dict[str, Any]) -> dict[str, Any] | None:
        if "fact_cryptic_site" not in query:
            return None
        if "md_validation_status FROM" in query:
            return {"md_validation_status": self.site.get("md_validation_status", "pending")}
        if "site_id, structure_id" in query:
            return self.site
        return None

    async def execute(self, query: str, params: dict[str, Any]) -> None:
        if "UPDATE fact_cryptic_site" in query:
            self.site["md_validation_status"] = params["status"]


@pytest.mark.parametrize(
    "smd_payload",
    [
        {"success": True},
        {"success": True, "status": "passed"},
        {
            "success": True,
            "work_kcal_mol": 10.0,
            "duration_ms": 500,
            "notes": "STUB: Synthetic SMD result — no real physics performed.",
        },
        {
            "success": True,
            "work_kcal_mol": 8.0,
            "duration_ms": 0,
            "notes": "Finished",
        },
        {"success": False, "status": "failed", "work_kcal_mol": 0.0, "duration_ms": 100},
        {"success": True, "status": "timeout"},
    ],
    ids=[
        "bare_success_flag",
        "bare_success_with_status",
        "stub_marked_result",
        "zero_duration",
        "explicit_failure",
        "timeout_status",
    ],
)
def test_synthetic_smd_never_qualifies_for_passed(smd_payload: dict[str, Any]) -> None:
    assert not smd_result_qualifies_for_passed(smd_payload)


def test_authentic_smd_result_qualifies_for_passed() -> None:
    assert smd_result_qualifies_for_passed(_AUTHENTIC_SMD)


def test_stub_mode_env_never_qualifies_even_with_authentic_shape(monkeypatch) -> None:
    monkeypatch.setenv("SMD_STUB_MODE", "1")
    from science.dtie.cryptic import stub_control

    stub_control.reset_approval()
    assert not smd_result_qualifies_for_passed(_AUTHENTIC_SMD)


@pytest.mark.asyncio
async def test_validate_site_rejects_bare_success_smd(monkeypatch) -> None:
    site_id = "site_kras_1"
    db = _CrypticSiteDB(
        {
            "site_id": site_id,
            "structure_id": "kras",
            "site_type": "surface_pocket",
            "residue_ids": ["A:12"],
            "md_validation_status": "pending",
        }
    )

    def _fake_smd(_spec_path: str, _protocol: str) -> dict[str, Any]:
        return {"success": True, "status": "passed"}

    monkeypatch.setattr("science.compute.cryptic.md_validate.run_smd", _fake_smd)

    result = await validate_site_md_in_process(site_id, db)
    assert result["md_validation_status"] != "passed"
    assert db.site["md_validation_status"] != "passed"


@pytest.mark.asyncio
async def test_validate_site_accepts_authentic_smd(monkeypatch) -> None:
    site_id = "site_kras_2"
    db = _CrypticSiteDB(
        {
            "site_id": site_id,
            "structure_id": "kras",
            "site_type": "surface_pocket",
            "residue_ids": ["A:12"],
            "md_validation_status": "pending",
        }
    )

    monkeypatch.setattr(
        "science.compute.cryptic.md_validate.run_smd",
        lambda _spec, _proto: dict(_AUTHENTIC_SMD),
    )

    result = await validate_site_md_in_process(site_id, db)
    assert result["md_validation_status"] == "passed"
    assert db.site["md_validation_status"] == "passed"


@pytest.mark.asyncio
async def test_dry_run_runner_never_emits_md_validation_artifact() -> None:
    db = AsyncMock()
    db.commit = AsyncMock()
    ctx = JobRunContext(
        structure_id="kras",
        job_id="md_validate_top_n",
        job_params={"site_id": "site_kras_dry", "dry_run": True},
    )

    result = await run_md_validate_top_n(db, ctx)

    assert "md_validation" not in result.artifacts_produced
    assert result.outputs.get("passed", 0) == 0
    dry = result.outputs.get("results", [{}])[0]
    assert dry.get("dry_run") is True
