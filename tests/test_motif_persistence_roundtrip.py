"""Property-based test for motif persistence round-trip.

Feature: science-container-api, Property 5: Motif persistence round-trip

For any successful motif analysis, querying fact_hyperbolic_motif for the
returned run_id SHALL yield exactly motif_count rows, each with valid
residue_ids, medoid_residue_id, and angular_sector.

**Validates: Requirements 5.3, 5.4, 8.1, 8.3**
"""

from __future__ import annotations

import asyncio
import math
import uuid
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from science.api.routers.compute import (
    _classify_angular_sector,
    _compute_poincare_distance_matrix,
    _persist_motifs,
)
from data.normalizer.core import Normalizer


# ---------------------------------------------------------------------------
# Mock DB that captures fact_hyperbolic_motif inserts
# ---------------------------------------------------------------------------


class MotifPersistenceMockDB:
    """In-memory mock DB that tracks fact_hyperbolic_motif inserts."""

    def __init__(self):
        self.motif_rows: list[dict[str, Any]] = []
        self.provenance_runs: list[dict[str, Any]] = []
        self.governed_assets: list[dict[str, Any]] = []

    async def execute(self, query: str, params: dict[str, Any] | None = None) -> None:
        if params is None:
            return
        q = query.strip().lower()
        if "fact_hyperbolic_motif" in q:
            self.motif_rows.append(dict(params))
        elif "provenance_run" in q:
            self.provenance_runs.append(dict(params))

    async def execute_many(self, query: str, params_list: list[dict[str, Any]]) -> None:
        q = query.strip().lower()
        if "governed_asset" in q:
            for p in params_list:
                self.governed_assets.append(dict(p))

    async def fetch_one(
        self, query: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        return None

    async def begin(self) -> None:
        pass

    async def commit(self) -> None:
        pass

    async def rollback(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

VALID_SECTORS = {"core", "N", "NE", "E", "SE", "S", "SW", "W", "NW"}


@st.composite
def motif_list_strategy(draw):
    """Generate a list of motifs as produced by the motif analysis endpoint.

    Generates between 1 and 8 motifs, each with random cluster data
    that simulates what the HDBSCAN clustering would produce.
    """
    n_motifs = draw(st.integers(min_value=1, max_value=8))
    run_id = f"run_{uuid.uuid4().hex[:12]}"
    structure_id = draw(st.sampled_from(["4obe", "6q21", "1crn", "3pqr"]))

    motifs = []
    for cluster_id in range(n_motifs):
        # Each motif has between 3 and 15 residues
        motif_size = draw(st.integers(min_value=3, max_value=15))
        chain = draw(st.sampled_from(["A", "B", "C"]))
        residue_ids = [
            f"{structure_id}_{chain}_{i}"
            for i in draw(
                st.lists(
                    st.integers(min_value=1, max_value=500),
                    min_size=motif_size,
                    max_size=motif_size,
                    unique=True,
                )
            )
        ]
        medoid_residue_id = draw(st.sampled_from(residue_ids))

        # Generate centroid in polar coords on Poincaré disc
        centroid_radius = draw(
            st.floats(min_value=0.0, max_value=0.95, allow_nan=False, allow_infinity=False)
        )
        centroid_angle_deg = draw(
            st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False)
        )

        angular_sector = _classify_angular_sector(centroid_angle_deg, centroid_radius)

        motifs.append({
            "motif_id": f"motif_{run_id}_{cluster_id}",
            "cluster_id": cluster_id,
            "residue_ids": residue_ids,
            "medoid_residue_id": medoid_residue_id,
            "centroid_angle_deg": round(centroid_angle_deg, 2),
            "centroid_radius": round(centroid_radius, 4),
            "motif_size": motif_size,
            "angular_sector": angular_sector,
            "classification": None,
        })

    return run_id, structure_id, motifs


# ---------------------------------------------------------------------------
# Property Test
# ---------------------------------------------------------------------------


class TestMotifPersistenceRoundTrip:
    """Property-based test for motif persistence round-trip.

    # Feature: science-container-api, Property 5: Motif persistence round-trip
    """

    @settings(max_examples=100)
    @given(data=motif_list_strategy())
    def test_property_5_motif_persistence_roundtrip(
        self, data: tuple[str, str, list[dict[str, Any]]]
    ):
        """For any successful motif analysis, querying fact_hyperbolic_motif
        for the returned run_id SHALL yield exactly motif_count rows, each
        with valid residue_ids, medoid_residue_id, and angular_sector.

        **Validates: Requirements 5.3, 5.4, 8.1, 8.3**
        """
        run_id, structure_id, motifs = data
        motif_count = len(motifs)

        db = MotifPersistenceMockDB()
        normalizer = Normalizer(db=db, caller_identity="test")  # type: ignore[arg-type]

        # Persist motifs through the same path the endpoint uses
        asyncio.run(_persist_motifs(normalizer, db, run_id, structure_id, motifs))  # type: ignore[arg-type]

        # --- Round-trip verification ---

        # 1. Exactly motif_count rows persisted
        assert len(db.motif_rows) == motif_count, (
            f"Expected {motif_count} rows, got {len(db.motif_rows)}"
        )

        # 2. All rows have the correct run_id
        for row in db.motif_rows:
            assert row["run_id"] == run_id

        # 3. Each row has valid residue_ids (non-empty list)
        for row in db.motif_rows:
            assert isinstance(row["residue_ids"], list)
            assert len(row["residue_ids"]) > 0

        # 4. Each row has a medoid_residue_id that is in its residue_ids
        for row in db.motif_rows:
            assert row["medoid_residue_id"] in row["residue_ids"], (
                f"medoid {row['medoid_residue_id']} not in residue_ids {row['residue_ids']}"
            )

        # 5. Each row has a valid angular_sector
        for row in db.motif_rows:
            assert row["angular_sector"] in VALID_SECTORS, (
                f"Invalid angular_sector: {row['angular_sector']}"
            )

        # 6. Unique constraint: each (run_id, cluster_id) pair is unique
        seen_keys = set()
        for row in db.motif_rows:
            key = (row["run_id"], row["cluster_id"])
            assert key not in seen_keys, f"Duplicate key: {key}"
            seen_keys.add(key)

        # 7. Provenance run was created
        assert len(db.provenance_runs) >= 1
        assert any(p["run_id"] == run_id for p in db.provenance_runs)

        # 8. Governed assets registered (one per motif)
        assert len(db.governed_assets) == motif_count
