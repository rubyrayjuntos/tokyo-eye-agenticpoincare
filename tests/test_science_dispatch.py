from __future__ import annotations

import asyncio

import pytest

from agent.tools.science_dispatch import (
    diagnose_science_job,
    start_full_pipeline_job,
    start_science_job,
    wait_for_science_job,
)


class _FakeProcess:
    def __init__(self, *, stdout: str = "", stderr: str = "", returncode: int = 0, delay: float = 0.0):
        self.pid = 4242
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr
        self._delay = delay
        self.killed = False

    async def communicate(self):
        if self._delay:
            await asyncio.sleep(self._delay)
        return self._stdout.encode(), self._stderr.encode()

    def kill(self):
        self.killed = True
        self.returncode = -9

    async def wait(self):
        return self.returncode


@pytest.mark.asyncio
async def test_start_full_pipeline_job_returns_background_handle(monkeypatch):
    async def _fake_create_subprocess_exec(*args, **kwargs):
        return _FakeProcess(stdout='{"run_id":"dtie_test","phases_run":["gnn_inference","phase2"]}\n')

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)

    queued = start_full_pipeline_job("4OBE")
    assert queued["status"] == "queued"
    assert queued["structure_id"] == "4obe"

    finished = await wait_for_science_job(queued["job_id"], timeout_seconds=2, poll_interval_seconds=1)
    assert finished["status"] == "completed"
    assert finished["result_summary"]["run_id"] == "dtie_test"
    assert finished["result_summary"]["phases_run"] == ["gnn_inference", "phase2"]


@pytest.mark.asyncio
async def test_diagnose_science_job_reports_timeout(monkeypatch):
    async def _fake_create_subprocess_exec(*args, **kwargs):
        return _FakeProcess(stdout="", stderr="", delay=0.05)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)

    queued = start_science_job(
        module="science.dtie.v5.orchestrator.pipeline",
        args=["--structure", "4obe"],
        timeout=0.01,
        job_type="full_pipeline",
        structure_id="4obe",
    )

    finished = await wait_for_science_job(queued["job_id"], timeout_seconds=1, poll_interval_seconds=1)
    assert finished["status"] == "timed_out"

    diagnosis = diagnose_science_job(queued["job_id"])
    assert diagnosis["diagnosis"] == "failed_or_stalled"
    assert "timed out" in diagnosis["error"]
