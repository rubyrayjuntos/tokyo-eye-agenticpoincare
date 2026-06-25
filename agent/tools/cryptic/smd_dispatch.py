"""SMD dispatch utilities for cryptic site MD validation.

Wraps science_dispatch with GPU-appropriate timeout handling
for steered molecular dynamics jobs.

The default science_dispatch timeout (300s) is insufficient for GPU SMD jobs.
This module ensures proper timeout handling (default 3600s per site).

Requirements: 6.1
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Default timeout for GPU SMD jobs (1 hour)
DEFAULT_SMD_TIMEOUT_SECONDS = 3600

# Module path for the real SMD runner in the science container
SMD_RUNNER_MODULE = "science.dtie.cryptic.smd_runner"


async def dispatch_smd_job(
    spec_json_path: str,
    protocol: str,
    structure_id: str,
    timeout_seconds: int = DEFAULT_SMD_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Dispatch a steered MD job to the science container.

    Uses start_science_job + wait_for_science_job with GPU-appropriate
    timeout (default 3600s).

    Args:
        spec_json_path: Path to the site spec JSON (accessible inside container).
        protocol: SMD protocol name (e.g., "SMD_three_phase").
        structure_id: Structure ID for provenance tracking.
        timeout_seconds: Max seconds to wait for completion (default 3600).

    Returns:
        Dict with success, status, protocol, work_kcal_mol, strain_delta,
        duration_ms, notes, metrics.
    """
    from agent.tools.science_dispatch import (
        start_science_job,
        wait_for_science_job,
    )

    logger.info(
        "Dispatching SMD job: protocol=%s, structure=%s, timeout=%ds",
        protocol,
        structure_id,
        timeout_seconds,
    )

    job = start_science_job(
        module=SMD_RUNNER_MODULE,
        args=["--spec-json", spec_json_path, "--protocol", protocol],
        timeout=timeout_seconds,
        job_type="cryptic_md_validation",
        structure_id=structure_id,
    )

    job_id = job["job_id"]
    logger.info("SMD job started: job_id=%s", job_id)

    # Wait with GPU-appropriate timeout
    result = await wait_for_science_job(
        job_id=job_id,
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=30,
    )

    if not result.get("success"):
        status = result.get("status", "unknown")
        logger.warning("SMD job %s did not succeed: status=%s", job_id, status)

        if status == "timed_out_waiting":
            return {
                "success": False,
                "status": "timeout",
                "protocol": protocol,
                "work_kcal_mol": 0.0,
                "strain_delta": 0.0,
                "duration_ms": timeout_seconds * 1000,
                "notes": f"SMD job timed out after {timeout_seconds}s",
                "metrics": {"job_id": job_id},
            }

        return {
            "success": False,
            "status": "failed",
            "protocol": protocol,
            "work_kcal_mol": 0.0,
            "strain_delta": 0.0,
            "duration_ms": 0,
            "notes": f"SMD job failed: {result.get('error', 'unknown error')}",
            "metrics": {"job_id": job_id},
        }

    return result


async def dispatch_smd_sync(
    spec_json_path: str,
    protocol: str,
    timeout_seconds: int = DEFAULT_SMD_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Run SMD synchronously using run_science_command.

    For use in calibration scripts where we want direct results
    without the async job tracking overhead.

    Args:
        spec_json_path: Path to the site spec JSON.
        protocol: SMD protocol name.
        timeout_seconds: Max seconds to wait (default 3600).

    Returns:
        Parsed JSON result dict from the SMD runner.
    """
    import json

    from agent.tools.science_dispatch import run_science_command

    result = run_science_command(
        module=SMD_RUNNER_MODULE,
        args=["--spec-json", spec_json_path, "--protocol", protocol],
        timeout=timeout_seconds,
    )

    if not result.get("success"):
        return {
            "success": False,
            "status": "failed",
            "protocol": protocol,
            "work_kcal_mol": 0.0,
            "strain_delta": 0.0,
            "duration_ms": 0,
            "notes": f"Science command failed: {result.get('stderr', '')[:200]}",
            "metrics": {},
        }

    # Parse JSON from stdout
    stdout = result.get("stdout", "")
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        return {
            "success": False,
            "status": "failed",
            "protocol": protocol,
            "work_kcal_mol": 0.0,
            "strain_delta": 0.0,
            "duration_ms": 0,
            "notes": f"Failed to parse SMD output: {stdout[:200]}",
            "metrics": {},
        }
