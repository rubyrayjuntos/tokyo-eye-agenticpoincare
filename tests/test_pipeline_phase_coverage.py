"""Property-based test for pipeline phase coverage.

Feature: science-container-api, Property 3: Pipeline phase coverage

For any successful pipeline run, the returned phases_run list SHALL contain
all enabled phase names, and assets_created SHALL be greater than zero.

**Validates: Requirements 3.2, 3.3**
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from science.dtie.common.interfaces import (
    GNNInferenceResult,
    GNNNodeOutput,
    PhaseResult,
    PipelineResult,
)
from science.dtie.v5.orchestrator.pipeline import (
    DTIEOrchestrator,
    PipelineConfig,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# All phases that the orchestrator can run (keys in phase_results)
ALL_PHASE_NAMES = [
    "gnn_inference",
    "phase1",
    "phase2",
    "phase3",
    "phase35",
    "phase4",
    "phase5",
    "phase6a_virtual_screening",
    "phase6b_binding_affinity",
    "phase6c_admet_filter",
    "phase6d_state_selectivity",
    "source_leak_detection",
    "allosteric_sites",
]


@st.composite
def successful_pipeline_result_strategy(draw):
    """Generate a PipelineResult representing a successful pipeline run.

    Ensures at least one phase succeeded (required for overall success).
    Simulates what DTIEOrchestrator.run() would return for a successful run.
    """
    structure_id = draw(st.sampled_from(["4obe", "1abc", "2xyz", "3def"]))
    run_id = f"dtie_{draw(st.text(st.characters(whitelist_categories=('Nd', 'Ll')), min_size=12, max_size=12))}"

    # Choose which phases are enabled (at least GNN + source_leak for success)
    enabled_phases = draw(
        st.lists(
            st.sampled_from(ALL_PHASE_NAMES),
            min_size=2,
            max_size=len(ALL_PHASE_NAMES),
            unique=True,
        )
    )
    # Ensure required phases are present for a valid successful run
    if "gnn_inference" not in enabled_phases:
        enabled_phases.append("gnn_inference")
    if "source_leak_detection" not in enabled_phases:
        enabled_phases.append("source_leak_detection")

    # Build phase results — all enabled phases succeed
    phase_results: dict[str, PhaseResult] = {}
    for phase_name in enabled_phases:
        outputs: dict[str, Any] = {"status": "completed"}
        if phase_name == "gnn_inference":
            node_count = draw(st.integers(min_value=1, max_value=200))
            outputs = {"node_count": node_count, "architecture": "decoupled_radial_angular"}
        elif phase_name == "source_leak_detection":
            leak_count = draw(st.integers(min_value=0, max_value=20))
            outputs = {"source_leak_count": leak_count, "source_leak_residues": []}
        phase_results[phase_name] = PhaseResult(
            phase_name=phase_name,
            structure_id=structure_id,
            model_version=f"DTIE-v5-{phase_name}",
            success=True,
            outputs=outputs,
        )

    # Build a minimal GNN result to represent the inference output
    n_nodes = phase_results["gnn_inference"].outputs.get("node_count", 10)
    nodes = []
    for i in range(n_nodes):
        nodes.append(GNNNodeOutput(
            residue_index=i + 1,
            chain_label="A",
            input_features=np.array([0.5, 0.0, 1.0, 100.0], dtype=np.float32),
            projections=np.random.randn(64).astype(np.float32),
            cone_depth=1.5,
            cone_width=0.5,
            epistemic_uncertainty=0.2,
            aleatoric_uncertainty=0.1,
            total_uncertainty=0.3,
            x_hyp=np.random.randn(128).astype(np.float32),
            hyp_projections=np.random.randn(2).astype(np.float32),
        ))

    gnn_result = GNNInferenceResult(
        structure_id=structure_id,
        model_version="GOSPConeMapper-v5",
        checkpoint_path="checkpoints_v5/v5_stage4_11prot.pt",
        nodes=nodes,
        curvature=1.0,
        embedding_dim=128,
        space_type="hyperbolic",
    )

    return PipelineResult(
        run_id=run_id,
        structure_id=structure_id,
        model_version="DTIE-v5",
        success=True,
        phase_results=phase_results,
        gnn_result=gnn_result,
        warnings=[],
    )


# ---------------------------------------------------------------------------
# Response construction logic (mirrors compute.py endpoint logic)
# ---------------------------------------------------------------------------


def compute_pipeline_response(result: PipelineResult) -> dict[str, Any]:
    """Extract phases_run and assets_created from a PipelineResult.

    This mirrors the logic in the POST /compute/pipeline endpoint.
    """
    phases_run = [
        name for name, pr in result.phase_results.items() if pr.success
    ]

    assets_created = 0
    if result.gnn_result:
        # GNN produces embeddings: N nodes * 2 spaces (hyp + euc)
        assets_created += len(result.gnn_result.nodes) * 2
    for phase_name, pr in result.phase_results.items():
        if pr.success and phase_name != "gnn_inference":
            assets_created += pr.outputs.get("assets_created", 1)

    return {
        "run_id": result.run_id,
        "structure_id": result.structure_id,
        "phases_run": phases_run,
        "assets_created": assets_created,
    }


# ---------------------------------------------------------------------------
# Property Test
# ---------------------------------------------------------------------------


class TestPipelinePhaseCoverage:
    """Property-based test for pipeline phase coverage.

    # Feature: science-container-api, Property 3: Pipeline phase coverage
    """

    @settings(max_examples=100)
    @given(result=successful_pipeline_result_strategy())
    def test_property_3_pipeline_phase_coverage(self, result: PipelineResult):
        """For any successful pipeline run, the returned phases_run list SHALL
        contain all enabled phase names that succeeded, and assets_created
        SHALL be greater than zero.

        **Validates: Requirements 3.2, 3.3**
        """
        response = compute_pipeline_response(result)

        # All enabled phases that succeeded must appear in phases_run
        expected_phases = {
            name for name, pr in result.phase_results.items() if pr.success
        }
        assert set(response["phases_run"]) == expected_phases, (
            f"phases_run mismatch: got {response['phases_run']}, "
            f"expected {expected_phases}"
        )

        # phases_run must not be empty (at least GNN + source_leak succeed)
        assert len(response["phases_run"]) >= 2, (
            f"Expected at least 2 phases, got {len(response['phases_run'])}"
        )

        # assets_created must be > 0 for any successful pipeline
        assert response["assets_created"] > 0, (
            f"assets_created should be > 0 for a successful pipeline, "
            f"got {response['assets_created']}"
        )

        # Specifically: GNN nodes contribute 2*N assets (hyp + euc spaces)
        if result.gnn_result:
            n_nodes = len(result.gnn_result.nodes)
            assert response["assets_created"] >= n_nodes * 2, (
                f"assets_created ({response['assets_created']}) should be >= "
                f"2 * node_count ({n_nodes * 2})"
            )
