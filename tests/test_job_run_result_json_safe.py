"""Tests for JSON-safe serialization of compute job outputs."""

from __future__ import annotations

import json

import numpy as np
import pytest

from science.compute.runners.base import JobRunResult, json_safe


def test_json_safe_coerces_numpy_scalars_and_arrays():
    payload = {
        "count": np.int64(42),
        "score": np.float64(0.75),
        "labels": np.array([1, 2, 3]),
        "nested": {"size": np.int32(7)},
    }

    safe = json_safe(payload)

    assert safe == {
        "count": 42,
        "score": 0.75,
        "labels": [1, 2, 3],
        "nested": {"size": 7},
    }
    json.dumps(safe)


def test_job_run_result_to_dict_is_json_serializable():
    result = JobRunResult(
        job_id="hyperbolic_motifs",
        run_id="run_1",
        structure_id="11qe",
        success=True,
        artifacts_produced=["hyperbolic_motifs"],
        outputs={
            "motif_count": np.int64(3),
            "centroid": np.array([0.1, 0.2]),
        },
    )

    encoded = json.dumps(result.to_dict())

    assert '"motif_count": 3' in encoded
    assert '"centroid": [0.1, 0.2]' in encoded
