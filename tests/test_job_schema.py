"""Tests for compute job schema catalog."""

from __future__ import annotations

from science.compute.job_schema import (
    SCHEMA_PATH,
    build_job_catalog,
    load_job_schema,
    validate_schema_against_registry,
)
from science.compute.registry import JOB_REGISTRY


class TestJobSchema:
    def test_schema_file_exists(self):
        assert SCHEMA_PATH.is_file()

    def test_schema_matches_registry(self):
        errors = validate_schema_against_registry()
        assert errors == []

    def test_all_jobs_have_destinations(self):
        schema = load_job_schema()
        for job_id, entry in schema["jobs"].items():
            assert entry.get("destinations"), f"{job_id} missing destinations"
            assert entry.get("execution"), f"{job_id} missing execution"
            assert set(entry["requires"]) == set(JOB_REGISTRY[job_id].requires)

    def test_canonical_dispatch_documented(self):
        schema = load_job_schema()
        assert schema["canonical_dispatch"] == "POST /compute/jobs/{job_id}"

    def test_build_is_deterministic_job_count(self):
        catalog = build_job_catalog()
        assert len(catalog["jobs"]) == len(JOB_REGISTRY)
