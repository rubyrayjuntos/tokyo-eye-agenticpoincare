"""Property-based test for GNN persistence completeness.

Feature: science-container-api, Property 2: GNN persistence completeness

For any valid structure_id with residues in dim_residue, calling the GNN
inference + persist flow SHALL result in fact_gnn_node_embedding rows equal
to the structure's residue count, each with non-null hyp_projections and
cone_depth.

**Validates: Requirements 2.2, 2.3**
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from data.normalizer.core import Normalizer
from science.dtie.common.adapters import GNNOutputAdapter
from science.dtie.common.interfaces import GNNInferenceResult, GNNNodeOutput
from science.dtie.common.keys import make_residue_id


# ---------------------------------------------------------------------------
# Mock DB that captures GNN node embeddings
# ---------------------------------------------------------------------------


class GNNPersistenceMockDB:
    """In-memory mock DB that tracks fact_gnn_node_embedding inserts."""

    def __init__(self):
        self.embeddings: list[dict[str, Any]] = []
        self.provenance_runs: list[dict[str, Any]] = []
        self.embedding_spaces: list[dict[str, Any]] = []
        self.governed_assets: list[dict[str, Any]] = []
        self.audit_records: list[dict[str, Any]] = []

    async def execute(self, query: str, params: dict[str, Any] | None = None) -> None:
        if params is None:
            return
        q = query.strip().lower()
        if "fact_gnn_node_embedding" in q:
            self.embeddings.append(dict(params))
        elif "provenance_run" in q:
            self.provenance_runs.append(dict(params))
        elif "embedding_space" in q and "insert" in q:
            self.embedding_spaces.append(dict(params))
        elif "normalization_audit" in q:
            self.audit_records.append(dict(params))

    async def execute_many(self, query: str, params_list: list[dict[str, Any]]) -> None:
        q = query.strip().lower()
        if "governed_asset" in q:
            for p in params_list:
                self.governed_assets.append(dict(p))

    async def fetch_one(self, query: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
        q = query.strip().lower()
        if "embedding_space" in q and "select" in q:
            # Return None first time (space doesn't exist), then it gets created
            name = params.get("name", "") if params else ""
            for space in self.embedding_spaces:
                if space.get("name") == name:
                    return {"space_id": space["space_id"]}
            return None
        if "provenance_run" in q:
            return None
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

STRUCTURE_ID = "4obe"


@st.composite
def gnn_inference_result_strategy(draw):
    """Generate a valid GNNInferenceResult with random node counts.

    Simulates what V5GNNRunner.run_inference() would return — a set of
    per-node outputs with hyperbolic embeddings, cone depth/width, and
    uncertainty values.
    """
    n_nodes = draw(st.integers(min_value=1, max_value=50))
    hidden_dim = 128

    nodes = []
    for i in range(1, n_nodes + 1):
        chain = draw(st.sampled_from(["A", "B"]))
        input_features = np.array([
            draw(st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False)),  # rho
            draw(st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False)),  # tau
            draw(st.sampled_from([0.0, 1.0, 2.0])),  # ss_type
            draw(st.floats(0.0, 300.0, allow_nan=False, allow_infinity=False)),  # sasa
        ], dtype=np.float32)

        # V5 produces 64-dim Euclidean projections
        projections = np.random.randn(64).astype(np.float32)
        # V5 produces hyperbolic embeddings in the Poincaré ball
        x_hyp = np.random.randn(hidden_dim).astype(np.float32)
        x_routed_hyp = np.random.randn(hidden_dim).astype(np.float32)
        # 2D disc projection
        hyp_2d = np.random.randn(2).astype(np.float32)
        hyp_2d = hyp_2d / (np.linalg.norm(hyp_2d) + 1e-6) * 0.8  # Keep inside disc

        # Cone depth and width must be non-zero floats
        cone_depth = float(draw(st.floats(0.01, 10.0, allow_nan=False, allow_infinity=False)))
        cone_width = float(draw(st.floats(0.01, 5.0, allow_nan=False, allow_infinity=False)))

        expert_weights = np.random.dirichlet(np.ones(4)).astype(np.float32)

        nodes.append(GNNNodeOutput(
            residue_index=i,
            chain_label=chain,
            input_features=input_features,
            projections=projections,
            cone_depth=cone_depth,
            cone_width=cone_width,
            epistemic_uncertainty=float(np.random.rand()),
            aleatoric_uncertainty=float(np.random.rand()),
            total_uncertainty=float(np.random.rand()),
            x_hyp=x_hyp,
            x_routed_hyp=x_routed_hyp,
            hyp_projections=hyp_2d,
            expert_weights=expert_weights,
        ))

    curvature = float(draw(st.floats(0.1, 5.0, allow_nan=False, allow_infinity=False)))

    return GNNInferenceResult(
        structure_id=STRUCTURE_ID,
        model_version="GOSPConeMapper-v5",
        checkpoint_path="checkpoints_v5/v5_stage4_11prot.pt",
        nodes=nodes,
        curvature=curvature,
        embedding_dim=hidden_dim,
        space_type="hyperbolic",
        metadata={"device": "cpu", "num_nodes": n_nodes},
    )


# ---------------------------------------------------------------------------
# Property Test
# ---------------------------------------------------------------------------


class TestGNNPersistenceCompleteness:
    """Property-based test for GNN persistence completeness.

    # Feature: science-container-api, Property 2: GNN persistence completeness
    """

    @settings(max_examples=100)
    @given(result=gnn_inference_result_strategy())
    def test_property_2_gnn_persistence_completeness(self, result: GNNInferenceResult):
        """For any valid structure_id with N residues, calling the GNN persist
        flow SHALL result in fact_gnn_node_embedding rows equal to N per
        embedding space, each with non-null hyp_projections and cone_depth.

        The V5 adapter produces two spaces (hyperbolic + euclidean), so we
        expect 2*N total embedding rows — N in the hyperbolic space (with
        hyp_projections) and N in the euclidean space (with cone_depth).

        **Validates: Requirements 2.2, 2.3**
        """
        db = GNNPersistenceMockDB()
        normalizer = Normalizer(db=db, caller_identity="test")  # type: ignore[arg-type]
        adapter = GNNOutputAdapter(normalizer=normalizer)

        n_nodes = len(result.nodes)
        run_id = f"run_test_{id(result)}"

        normalizer_results = asyncio.run(adapter.normalize(result, run_id=run_id))

        # V5 (hyperbolic) should produce 2 NormalizerResult (hyp + euc)
        assert len(normalizer_results) == 2

        # Total embeddings persisted = N (hyp) + N (euc) = 2*N
        assert len(db.embeddings) == 2 * n_nodes

        # Separate by space (hyp vs euc) using the run_id suffix
        hyp_embeddings = [e for e in db.embeddings if "_hyp" in e.get("run_id", "")]
        euc_embeddings = [e for e in db.embeddings if "_euc" in e.get("run_id", "")]

        # Each space gets exactly N rows
        assert len(hyp_embeddings) == n_nodes
        assert len(euc_embeddings) == n_nodes

        # Hyperbolic embeddings MUST have non-null hyp_projections and cone_depth
        for emb in hyp_embeddings:
            assert emb["hyp_projections"] is not None, (
                f"hyp_projections is None for residue {emb.get('residue_id')}"
            )
            assert emb["cone_depth"] is not None, (
                f"cone_depth is None for residue {emb.get('residue_id')}"
            )

        # All embeddings (hyp + euc) MUST have non-null cone_depth
        for emb in db.embeddings:
            assert emb["cone_depth"] is not None, (
                f"cone_depth is None for residue {emb.get('residue_id')}"
            )

        # Each NormalizerResult should report success and correct asset count
        for nr in normalizer_results:
            assert nr.success is True
            assert nr.assets_created == n_nodes
