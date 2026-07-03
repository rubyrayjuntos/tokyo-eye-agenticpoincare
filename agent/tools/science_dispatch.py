"""Science container dispatcher — runs compute jobs via docker compose.

The agent (launcher) is lightweight and doesn't have torch/GPU deps.
All heavy computation is dispatched to the science container via
`docker compose run science python -m <module> <args>`.

The science container shares the same database and filesystem volumes,
so results are written directly to the governed layer.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from science.dtie.common.keys import make_structure_id

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).parent.parent.parent  # tokyo-eye-agenticpoincare/
_science_jobs: dict[str, dict[str, Any]] = {}
_science_job_tasks: dict[str, asyncio.Task[None]] = {}


def _normalize_structure_id(structure_id: str) -> str:
    normalized = structure_id.strip()
    if len(normalized) == 4:
        return make_structure_id(pdb_id=normalized, source="rcsb")
    return normalized.lower()


def _build_science_command(module: str, args: list[str] | None = None) -> list[str]:
    cmd = [
        "docker", "compose",
        "-f", "/app/docker-compose.yml",
        "-p", "tokyo-eye-agenticpoincare",
        "exec",
        "-T",
        "science",
        "python", "-m", module,
    ]
    if args:
        cmd.extend(args)
    return cmd


def _trim_output(text: str, limit: int = 2000) -> str:
    normalized = (text or "").strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(limit - 3, 0)] + "..."


def _summarize_result(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if not result:
        return None
    summary: dict[str, Any] = {
        "success": result.get("success", False),
        "returncode": result.get("returncode"),
    }
    if result.get("success"):
        data = result.get("data")
        if isinstance(data, dict):
            for key in ("run_id", "phases_run", "warnings", "structure_id", "pocket_count"):
                if key in data:
                    summary[key] = data[key]
        elif "stdout" in result:
            summary["stdout"] = _trim_output(str(result.get("stdout", "")), limit=400)
    else:
        summary["error"] = result.get("error")
    return summary


def _snapshot_job(job: dict[str, Any]) -> dict[str, Any]:
    finished_at = job.get("completed_at_epoch")
    elapsed = (finished_at or time.time()) - job["started_at_epoch"]
    return {
        "job_id": job["job_id"],
        "job_type": job["job_type"],
        "module": job["module"],
        "structure_id": job.get("structure_id"),
        "status": job["status"],
        "pid": job.get("pid"),
        "started_at": job["started_at"],
        "completed_at": job.get("completed_at"),
        "elapsed_seconds": round(max(elapsed, 0.0), 1),
        "timeout_seconds": job["timeout_seconds"],
        "result_summary": _summarize_result(job.get("result")),
        "stdout_tail": job.get("stdout_tail"),
        "stderr_tail": job.get("stderr_tail"),
        "error": job.get("error"),
    }


async def _run_science_job(job_id: str) -> None:
    job = _science_jobs[job_id]
    cmd = _build_science_command(job["module"], job["args"])
    job["status"] = "running"
    logger.info("Dispatching science job %s: %s %s", job_id, job["module"], job["args"])
    process = None
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(PROJECT_DIR),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        job["pid"] = process.pid
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(),
            timeout=job["timeout_seconds"],
        )
        stdout = stdout_bytes.decode() if isinstance(stdout_bytes, bytes) else stdout_bytes
        stderr = stderr_bytes.decode() if isinstance(stderr_bytes, bytes) else stderr_bytes
        job["stdout_tail"] = _trim_output(stdout)
        job["stderr_tail"] = _trim_output(stderr)

        if process.returncode == 0:
            try:
                data = json.loads((stdout or "").strip().split("\n")[-1])
                job["result"] = {"success": True, "data": data, "returncode": 0}
            except (json.JSONDecodeError, IndexError):
                job["result"] = {"success": True, "stdout": stdout, "returncode": 0}
            job["status"] = "completed"
        else:
            error = (stderr or stdout or "").strip() or f"Science command failed with exit code {process.returncode}"
            job["result"] = {
                "success": False,
                "error": error,
                "returncode": process.returncode,
            }
            job["status"] = "failed"
            job["error"] = error
    except asyncio.TimeoutError:
        if process is not None and process.returncode is None:
            process.kill()
            await process.wait()
        job["status"] = "timed_out"
        job["error"] = f"Science container timed out after {job['timeout_seconds']}s"
        job["result"] = {"success": False, "error": job["error"]}
    except FileNotFoundError:
        job["status"] = "failed"
        job["error"] = "docker not found — is Docker installed and running?"
        job["result"] = {"success": False, "error": job["error"]}
    except Exception as e:
        job["status"] = "failed"
        job["error"] = f"{type(e).__name__}: {e}"
        job["result"] = {"success": False, "error": job["error"]}
    finally:
        job["completed_at_epoch"] = time.time()
        job["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(job["completed_at_epoch"]))


def start_science_job(
    *,
    module: str,
    args: list[str] | None = None,
    timeout: int = 300,
    job_type: str,
    structure_id: str | None = None,
) -> dict[str, Any]:
    job_id = f"science_{uuid.uuid4().hex[:12]}"
    started_at_epoch = time.time()
    job = {
        "job_id": job_id,
        "job_type": job_type,
        "module": module,
        "args": args or [],
        "structure_id": structure_id,
        "status": "queued",
        "pid": None,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started_at_epoch)),
        "started_at_epoch": started_at_epoch,
        "completed_at": None,
        "completed_at_epoch": None,
        "timeout_seconds": timeout,
        "result": None,
        "stdout_tail": None,
        "stderr_tail": None,
        "error": None,
    }
    _science_jobs[job_id] = job
    _science_job_tasks[job_id] = asyncio.create_task(_run_science_job(job_id))
    return _snapshot_job(job)


def get_science_job_status(job_id: str) -> dict[str, Any]:
    job = _science_jobs.get(job_id)
    if not job:
        return {"success": False, "job_id": job_id, "error": f"Unknown science job '{job_id}'"}
    return {"success": True, **_snapshot_job(job)}


async def wait_for_science_job(
    job_id: str,
    timeout_seconds: int = 30,
    poll_interval_seconds: int = 5,
) -> dict[str, Any]:
    deadline = time.time() + max(timeout_seconds, 1)
    poll_interval = max(1, min(poll_interval_seconds, 30))
    while True:
        snapshot = get_science_job_status(job_id)
        if not snapshot.get("success"):
            return snapshot
        if snapshot["status"] in {"completed", "failed", "timed_out"}:
            return snapshot
        if time.time() >= deadline:
            snapshot["timed_out_waiting"] = True
            return snapshot
        await asyncio.sleep(poll_interval)


def diagnose_science_job(job_id: str) -> dict[str, Any]:
    snapshot = get_science_job_status(job_id)
    if not snapshot.get("success"):
        return snapshot

    status = snapshot["status"]
    elapsed = snapshot["elapsed_seconds"]
    timeout_seconds = snapshot["timeout_seconds"]
    diagnosis = "running_normally"
    recommendation = "Use wait_for_science_job to keep waiting, or query persisted retrieval tools for any partial outputs."
    if status == "completed":
        diagnosis = "completed"
        recommendation = "Retrieve the produced data with the relevant retrieval tools."
    elif status in {"failed", "timed_out"}:
        diagnosis = "failed_or_stalled"
        recommendation = "Inspect the error, then retry or switch to a narrower run."
    elif elapsed >= timeout_seconds * 0.8:
        diagnosis = "near_timeout"
        recommendation = "This run is approaching its timeout. Check partial persisted outputs now and consider retrying with a narrower scope if progress has stalled."
    elif elapsed >= 60:
        diagnosis = "long_running"
    snapshot["diagnosis"] = diagnosis
    snapshot["recommendation"] = recommendation
    return snapshot


async def run_science_command(
    module: str,
    args: list[str] | None = None,
    timeout: int = 300,
) -> dict[str, Any]:
    """Run a Python module in the science container.

    Uses `docker compose exec` to run inside the already-running science
    container (started by `make start`). This avoids container name conflicts
    and is faster than `docker compose run` since no new container is created.

    Args:
        module: Python module path (e.g., "science.dtie.v5.gnn.runner")
        args: Additional CLI arguments
        timeout: Max seconds to wait

    Returns:
        Dict with stdout, stderr, returncode, success
    """
    cmd = _build_science_command(module, args)

    logger.info("Dispatching to science container: %s %s", module, args or [])

    try:
        result = await asyncio.to_thread(
            subprocess.run,
            cmd,
            cwd=str(PROJECT_DIR),
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode == 0:
            # Try to parse JSON output (science scripts output JSON on success)
            try:
                data = json.loads(result.stdout.strip().split("\n")[-1])
                return {"success": True, "data": data, "returncode": 0}
            except (json.JSONDecodeError, IndexError):
                return {"success": True, "stdout": result.stdout, "returncode": 0}
        else:
            return {
                "success": False,
                "error": result.stderr.strip() or result.stdout.strip(),
                "returncode": result.returncode,
            }

    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Science container timed out after {timeout}s"}
    except FileNotFoundError:
        return {"success": False, "error": "docker not found — is Docker installed and running?"}
    except Exception as e:
        return {"success": False, "error": f"{type(e).__name__}: {e}"}


def _flag(enabled: bool, name: str) -> list[str]:
    return [name] if enabled else []


async def run_small_molecule_job_via_container(
    job: str,
    *,
    structure_id: str,
    candidate_percentile: float = 75.0,
    cluster_distance_angstrom: float = 8.0,
    min_cluster_size: int = 3,
    max_pockets: int = 10,
    spatial_cutoff: float = 8.0,
    persist: bool = False,
    radius_angstrom: float = 6.0,
    pocket_index: int | None = None,
    fallback_top_pockets: int = 5,
    max_mean_depth: float = 3.0,
    max_mean_epistemic: float = 0.8,
    max_mean_total_uncertainty: float = 1.5,
    selectivity_ratio_threshold: float = 1.3,
    druggability_weight: float = 0.3,
    binding_weight: float = 0.3,
    accessibility_weight: float = 0.2,
    selectivity_weight: float = 0.2,
    top_k: int = 5,
    docking_box_padding: float = 6.0,
    pose_count: int = 5,
    compare_structures: str | None = None,
    timeout: int = 900,
) -> dict[str, Any]:
    args = [
        job,
        "--structure", _normalize_structure_id(structure_id),
        "--candidate-percentile", str(candidate_percentile),
        "--cluster-distance-angstrom", str(cluster_distance_angstrom),
        "--min-cluster-size", str(min_cluster_size),
        "--max-pockets", str(max_pockets),
        "--spatial-cutoff", str(spatial_cutoff),
        *(_flag(persist, "--persist")),
    ]
    if job in {"map_pharmacophore_features", "search_pocket_vectors", "run_docking_surrogate", "run_small_molecule_pipeline"}:
        args.extend(["--radius-angstrom", str(radius_angstrom)])
    if pocket_index is not None and job in {"map_pharmacophore_features", "screen_fragments", "run_docking_surrogate", "search_pocket_vectors", "run_small_molecule_pipeline"}:
        args.extend(["--pocket-index", str(pocket_index)])
    if job in {"screen_fragments", "run_docking_surrogate", "run_small_molecule_pipeline"}:
        args.extend(
            [
                "--fallback-top-pockets", str(fallback_top_pockets),
                "--max-mean-depth", str(max_mean_depth),
                "--max-mean-epistemic", str(max_mean_epistemic),
                "--max-mean-total-uncertainty", str(max_mean_total_uncertainty),
                "--selectivity-ratio-threshold", str(selectivity_ratio_threshold),
                "--druggability-weight", str(druggability_weight),
                "--binding-weight", str(binding_weight),
                "--accessibility-weight", str(accessibility_weight),
                "--selectivity-weight", str(selectivity_weight),
                "--top-k", str(top_k),
            ]
        )
    if job in {"run_docking_surrogate", "run_small_molecule_pipeline"}:
        args.extend(
            [
                "--docking-box-padding", str(docking_box_padding),
                "--pose-count", str(pose_count),
            ]
        )
    if job == "search_pocket_vectors":
        args.extend(["--top-k", str(top_k)])
        if compare_structures:
            args.extend(["--compare-structures", compare_structures])

    return await run_science_command(
        module="science.dtie.v5.orchestrator.small_molecule_jobs",
        args=args,
        timeout=timeout,
    )


async def check_science_container() -> dict[str, Any]:
    """Check if the science container is available and can run."""
    return await run_science_command(
        module="science.dtie.v5.gnn.runner",
        args=["--check"],
        timeout=30,
    )


def start_full_pipeline_job(structure_id: str) -> dict[str, Any]:
    canonical_structure_id = _normalize_structure_id(structure_id)
    return start_science_job(
        module="science.dtie.v5.orchestrator.pipeline",
        args=["--structure", canonical_structure_id],
        timeout=600,
        job_type="full_pipeline",
        structure_id=canonical_structure_id,
    )
