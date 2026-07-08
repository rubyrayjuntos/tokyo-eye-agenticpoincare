"""Tests for DB-first training load helpers."""

from __future__ import annotations

import numpy as np
import pytest

from science.dtie.common.load_graph_from_db import build_protein_graph_dict


def test_build_protein_graph_dict_from_db_rows() -> None:
    rows = [
        {
            "residue_id": "4obe:A:1",
            "residue_index": i,
            "chain_label": "A",
            "rho": float(i),
            "tau_flag": 1.0 if i < 5 else 0.0,
            "ss_type": 0.0,
            "sasa": 50.0 + i,
            "ca_x": float(i),
            "ca_y": 0.0,
            "ca_z": 0.0,
            "ca_b_factor": 20.0,
        }
        for i in range(1, 15)
    ]
    graph = build_protein_graph_dict("4OBE", "A", rows)
    assert graph is not None
    assert graph["n_residues"] == 14
    assert graph["source"] == "db"
    x = graph["data"].x.numpy()
    assert x.shape == (14, 4)
    assert np.allclose(x[:, 0], np.arange(1, 15, dtype=np.float64))


@pytest.mark.asyncio
async def test_check_master_features_ready_empty() -> None:
    from science.dtie.common.structure_readiness import check_master_features_ready

    class FakeDB:
        async def fetch_one(self, query, params):
            return {"residue_count": 0, "with_sasa_sse": 0, "with_ingestion_facts": 0}

    status = await check_master_features_ready(FakeDB(), "4obe", "A")
    assert not status.ready
    assert status.reason == "no_dim_residue_rows"
