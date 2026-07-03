"""Tests for pathway executor error message extraction."""

from __future__ import annotations

from agent.tools.science_client import ScienceComputeError

from science.compute.pathway_executor import _job_exception_message


def test_job_exception_message_prefers_science_compute_detail():
    exc = ScienceComputeError(
        "/compute/jobs/strain_vulnerability_scan",
        500,
        "run_strain_vulnerability_scan() takes 2 positional arguments but 3 were given",
    )

    assert _job_exception_message(exc) == (
        "run_strain_vulnerability_scan() takes 2 positional arguments but 3 were given"
    )


def test_job_exception_message_falls_back_to_str():
    exc = RuntimeError("tier-1 job failed")

    assert _job_exception_message(exc) == "tier-1 job failed"
