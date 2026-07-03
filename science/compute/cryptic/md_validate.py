"""In-process MD validation for atomic compute jobs (canonical smd_runner path)."""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
from pathlib import Path
from typing import Any

from science.dtie.cryptic.smd_runner import run as run_smd
from science.dtie.cryptic.stub_control import is_stub_mode

logger = logging.getLogger(__name__)

_STUB_MARKERS = ("STUB", "SYNTHETIC", "NO REAL PHYSICS")


def smd_result_qualifies_for_passed(result: dict[str, Any]) -> bool:
    """Return True only when SMD output evidences a real simulation run.

    Synthetic/stub payloads and bare ``success`` flags must never qualify —
    ``md_validation_status='passed'`` is reserved for authentic MD completion.
    """
    if is_stub_mode():
        return False
    if result.get("status") == "timeout":
        return False
    if not result.get("success"):
        return False

    notes = str(result.get("notes") or "").upper()
    if any(marker in notes for marker in _STUB_MARKERS):
        return False

    work = result.get("work_kcal_mol")
    duration = result.get("duration_ms")
    if work is None and duration is None:
        return False
    if duration is not None and int(duration) <= 0:
        return False
    return True

SITE_TYPE_TO_PROTOCOL: dict[str, str] = {
    "cryptic_wedge": "SMD_three_phase",
    "structural_stent": "SMD_stent_stabilization",
    "dynamic_lid": "SMD_lid_restraint",
    "allosteric_clamp": "SMD_clamp_stabilization",
    "strain_relief_insert": "SMD_strain_relief",
    "surface_pocket": "SMD_three_phase",
}

VALID_STATUS_TRANSITIONS: dict[str, set[str]] = {
    "pending": {"running"},
    "running": {"passed", "failed", "timeout"},
    "passed": {"pending"},
    "failed": {"pending"},
    "timeout": {"pending"},
}


async def _update_md_status(site_id: str, new_status: str, db: Any) -> bool:
    row = await db.fetch_one(
        "SELECT md_validation_status FROM fact_cryptic_site WHERE site_id = :site_id",
        {"site_id": site_id},
    )
    if row is None:
        return False

    current_status = row.get("md_validation_status", "pending")
    allowed = VALID_STATUS_TRANSITIONS.get(current_status, set())
    if new_status not in allowed:
        logger.warning(
            "Invalid MD status transition for site %s: %s → %s",
            site_id,
            current_status,
            new_status,
        )
        return False

    await db.execute(
        "UPDATE fact_cryptic_site SET md_validation_status = :status WHERE site_id = :site_id",
        {"status": new_status, "site_id": site_id},
    )
    return True


def _write_site_spec_json(
    site_id: str,
    structure_id: str,
    residue_ids: list[str],
    output_dir: Path,
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = {
        "site_id": site_id,
        "structure_id": structure_id,
        "residue_ids": residue_ids,
    }
    spec_path = output_dir / f"{site_id}_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2))
    return str(spec_path)


async def validate_site_md_in_process(
    site_id: str,
    db: Any,
    *,
    timeout_seconds: int = 1800,
) -> dict[str, Any]:
    """Validate one site using smd_runner in-process (science container)."""
    site = await db.fetch_one(
        """
        SELECT site_id, structure_id, site_type, residue_ids, md_validation_status
        FROM fact_cryptic_site
        WHERE site_id = :site_id
        """,
        {"site_id": site_id},
    )
    if site is None:
        return {
            "site_id": site_id,
            "success": False,
            "md_validation_status": "failed",
            "error": f"Site '{site_id}' not found",
        }

    current_status = site.get("md_validation_status", "pending")
    if current_status == "running":
        return {
            "site_id": site_id,
            "success": False,
            "md_validation_status": "running",
            "error": "MD validation already in progress",
        }

    site_type = site.get("site_type", "surface_pocket")
    protocol = SITE_TYPE_TO_PROTOCOL.get(site_type, "SMD_three_phase")
    structure_id = site["structure_id"]

    if not await _update_md_status(site_id, "running", db):
        return {
            "site_id": site_id,
            "success": False,
            "md_validation_status": current_status,
            "error": f"Cannot transition from '{current_status}' to running",
        }

    residue_ids = site.get("residue_ids", [])
    if isinstance(residue_ids, str):
        residue_ids = json.loads(residue_ids)

    tmp_dir = Path(tempfile.mkdtemp(prefix="md_validate_job_"))
    spec_path = _write_site_spec_json(site_id, structure_id, residue_ids, tmp_dir)

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(run_smd, spec_path, protocol),
            timeout=timeout_seconds,
        )
    except TimeoutError:
        await _update_md_status(site_id, "timeout", db)
        return {
            "site_id": site_id,
            "structure_id": structure_id,
            "success": False,
            "md_validation_status": "timeout",
            "protocol": protocol,
        }
    except Exception as exc:
        await _update_md_status(site_id, "failed", db)
        logger.exception("SMD failed for site %s", site_id)
        return {
            "site_id": site_id,
            "structure_id": structure_id,
            "success": False,
            "md_validation_status": "failed",
            "protocol": protocol,
            "error": str(exc),
        }

    smd_status = result.get("status", "failed")
    if smd_status == "timeout":
        final_status = "timeout"
    elif smd_result_qualifies_for_passed(result):
        final_status = "passed"
    else:
        final_status = "failed"
        if result.get("success"):
            logger.warning(
                "SMD reported success for site %s but result did not qualify for passed "
                "(stub/synthetic or missing physics metrics)",
                site_id,
            )

    await _update_md_status(site_id, final_status, db)

    return {
        "site_id": site_id,
        "structure_id": structure_id,
        "site_type": site_type,
        "protocol": protocol,
        "success": final_status == "passed",
        "md_validation_status": final_status,
        "work_kcal_mol": result.get("work_kcal_mol", 0.0),
        "strain_delta": result.get("strain_delta", 0.0),
        "duration_ms": result.get("duration_ms", 0),
    }


async def validate_top_n_sites(
    structure_id: str,
    db: Any,
    *,
    top_n: int = 5,
    timeout_seconds: int = 1800,
) -> dict[str, Any]:
    """Batch MD validation for top N eligible binding sites."""
    rows = await db.fetch_all(
        """
        SELECT site_id, site_type, site_rank, md_validation_status
        FROM fact_cryptic_site
        WHERE structure_id = :structure_id
          AND md_validation_status IN ('pending', 'failed', 'timeout')
        ORDER BY site_rank ASC NULLS LAST
        LIMIT :limit
        """,
        {"structure_id": structure_id, "limit": top_n},
    )

    if not rows:
        return {
            "structure_id": structure_id,
            "validated": 0,
            "passed": 0,
            "failed": 0,
            "timed_out": 0,
            "results": [],
            "success": True,
        }

    results: list[dict[str, Any]] = []
    passed = failed = timed_out = 0

    for row in rows:
        site_result = await validate_site_md_in_process(
            row["site_id"],
            db,
            timeout_seconds=timeout_seconds,
        )
        status = site_result.get("md_validation_status", "failed")
        results.append(site_result)

        if status == "passed":
            passed += 1
        elif status == "timeout":
            timed_out += 1
        else:
            failed += 1

    return {
        "structure_id": structure_id,
        "validated": len(results),
        "passed": passed,
        "failed": failed,
        "timed_out": timed_out,
        "results": results,
        "success": passed > 0 or (failed == 0 and timed_out == 0),
    }
