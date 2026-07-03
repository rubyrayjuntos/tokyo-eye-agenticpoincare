"""Tests for witness embedding input loader."""

from __future__ import annotations

import numpy as np
import pytest

from science.compute.gnn_witness_inputs import load_witness_embedding_inputs


class _FakeDB:
    def __init__(self, rows):
        self._rows = rows

    async def fetch_all(self, _query, _params):
        return self._rows


@pytest.mark.asyncio
async def test_load_witness_embedding_inputs_builds_phase1_dict():
    rows = [
        {
            "residue_id": "kras:wt:A:12",
            "embedding_double": [0.1, 0.2, 0.3],
            "embedding": None,
            "epistemic_uncertainty": 0.4,
            "aleatoric_uncertainty": 0.6,
            "run_id": "run_gnn_1",
            "space_curvature": 1.25,
            "ca_x": 1.0,
            "ca_y": 2.0,
            "ca_z": 3.0,
            "chain_label": "A",
            "residue_index": 12,
        }
    ]
    inputs = await load_witness_embedding_inputs(_FakeDB(rows), "kras")
    assert inputs is not None
    assert inputs.gnn_output["gdp_x_routed_hyp"].shape == (1, 3)
    assert inputs.ingestion_data["no_midpoints"].shape == (1, 3)
    assert inputs.curvature_c == 1.25
    assert inputs.gnn_run_id == "run_gnn_1"
