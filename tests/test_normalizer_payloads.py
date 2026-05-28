"""Tests for Normalizer payload validation (Priority 2: payload schemas).

These tests validate that the payload schemas correctly accept v4 GNN output
shapes and reject malformed data.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from science.dtie.common.keys import make_residue_id, make_structure_id
from science.dtie.common.normalizer_payloads import (
    GNNNodeResult,
    GNNOutputPayload,
    NormalizerResult,
    PersistenceBarcode,
    Phase3PersistencePayload,
    Phase3ResidueContribution,
    ProvenanceContext,
    RunType,
    SourceType,
    SpaceType,
)


@pytest.fixture
def provenance() -> ProvenanceContext:
    return ProvenanceContext(
        run_id="run_test_001",
        structure_id="4obe",
        model_version="GOSPConeMapper-v4",
        pipeline_name="dtie_v4",
        run_type=RunType.INFERENCE,
        source_type=SourceType.PROBABILISTIC,
        checkpoint_uri="s3://checkpoints/tokyo_eyes_v4.pt",
        checkpoint_sha256="abc123def456",
        code_version="a1b2c3d",
    )


@pytest.fixture
def v4_node_result() -> GNNNodeResult:
    """A realistic v4 GNN node output for one residue."""
    return GNNNodeResult(
        residue_id="4obe:A:12",
        residue_index=12,
        chain_label="A",
        input_rho=0.85,
        input_tau_flag=1.0,
        input_ss_type=2.0,
        input_sasa=45.3,
        embedding=[0.1] * 64,  # 64-dim Euclidean projection
        cone_depth=2.34,
        cone_width=0.78,
        epistemic_uncertainty=0.12,
        aleatoric_uncertainty=0.08,
        total_uncertainty=0.20,
        x_hyp=[0.05] * 32,  # 32-dim hyperbolic embedding
        hyp_projections=[0.3, -0.2],  # 2D disc projection
        x_routed_hyp=[0.06] * 32,
        expert_weights=[0.4, 0.3, 0.2, 0.1],
    )


class TestProvenanceContext:
    def test_valid_provenance(self, provenance: ProvenanceContext):
        assert provenance.run_id == "run_test_001"
        assert provenance.run_type == RunType.INFERENCE

    def test_missing_run_id_raises(self):
        with pytest.raises(ValidationError):
            ProvenanceContext(
                structure_id="4obe",
                model_version="v4",
                pipeline_name="test",
            )  # type: ignore[call-arg]

    def test_training_run_type(self):
        prov = ProvenanceContext(
            run_id="train_001",
            structure_id="4obe",
            model_version="v4",
            pipeline_name="training",
            run_type=RunType.TRAINING,
            source_type=SourceType.PROBABILISTIC,
        )
        assert prov.run_type == RunType.TRAINING


class TestGNNNodeResult:
    def test_valid_v4_node(self, v4_node_result: GNNNodeResult):
        assert v4_node_result.residue_id == "4obe:A:12"
        assert len(v4_node_result.embedding) == 64
        assert v4_node_result.hyp_projections == [0.3, -0.2]

    def test_v3_node_without_hyperbolic(self):
        """V3 nodes don't have hyperbolic outputs — should still validate."""
        node = GNNNodeResult(
            residue_id="4obe:A:12",
            residue_index=12,
            chain_label="A",
            input_rho=0.85,
            input_tau_flag=1.0,
            input_ss_type=2.0,
            input_sasa=45.3,
            embedding=[0.1] * 64,
            cone_depth=2.34,
            cone_width=0.78,
            epistemic_uncertainty=0.12,
            # No aleatoric, no hyp fields
        )
        assert node.x_hyp is None
        assert node.hyp_projections is None
        assert node.aleatoric_uncertainty is None

    def test_empty_embedding_raises(self):
        with pytest.raises(ValidationError, match="embedding must not be empty"):
            GNNNodeResult(
                residue_id="4obe:A:12",
                residue_index=12,
                chain_label="A",
                input_rho=0.85,
                input_tau_flag=1.0,
                input_ss_type=2.0,
                input_sasa=45.3,
                embedding=[],  # Empty!
            )


class TestGNNOutputPayload:
    def test_valid_v4_payload(
        self, provenance: ProvenanceContext, v4_node_result: GNNNodeResult
    ):
        payload = GNNOutputPayload(
            provenance=provenance,
            space_type=SpaceType.HYPERBOLIC,
            space_name="gospcone_v4_hyp32",
            dimensionality=32,
            curvature=1.0,
            nodes=[v4_node_result],
        )
        assert payload.space_type == SpaceType.HYPERBOLIC
        assert len(payload.nodes) == 1

    def test_hyperbolic_requires_curvature(
        self, provenance: ProvenanceContext, v4_node_result: GNNNodeResult
    ):
        with pytest.raises(ValidationError, match="curvature is required"):
            GNNOutputPayload(
                provenance=provenance,
                space_type=SpaceType.HYPERBOLIC,
                space_name="test",
                dimensionality=32,
                curvature=None,  # Missing!
                nodes=[v4_node_result],
            )

    def test_euclidean_no_curvature_needed(
        self, provenance: ProvenanceContext, v4_node_result: GNNNodeResult
    ):
        payload = GNNOutputPayload(
            provenance=provenance,
            space_type=SpaceType.EUCLIDEAN,
            space_name="gospcone_v4_euc64",
            dimensionality=64,
            curvature=None,
            nodes=[v4_node_result],
        )
        assert payload.curvature is None

    def test_empty_nodes_raises(self, provenance: ProvenanceContext):
        with pytest.raises(ValidationError):
            GNNOutputPayload(
                provenance=provenance,
                space_type=SpaceType.EUCLIDEAN,
                space_name="test",
                dimensionality=64,
                nodes=[],  # Empty!
            )

    def test_multi_node_payload(self, provenance: ProvenanceContext):
        """Simulate a real structure with multiple residues."""
        nodes = [
            GNNNodeResult(
                residue_id=make_residue_id("4obe", "A", i),
                residue_index=i,
                chain_label="A",
                input_rho=0.5 + i * 0.01,
                input_tau_flag=1.0,
                input_ss_type=float(i % 3),
                input_sasa=40.0 + i,
                embedding=[0.1 * i] * 64,
                cone_depth=float(i) / 10,
                cone_width=0.5,
                epistemic_uncertainty=0.1,
            )
            for i in range(1, 166)  # ~165 residues (realistic for a small protein)
        ]
        payload = GNNOutputPayload(
            provenance=provenance,
            space_type=SpaceType.EUCLIDEAN,
            space_name="gospcone_v3_euc64",
            dimensionality=64,
            nodes=nodes,
        )
        assert len(payload.nodes) == 165


class TestPhase3PersistencePayload:
    def test_valid_v3_persistence(self, provenance: ProvenanceContext):
        payload = Phase3PersistencePayload(
            provenance=provenance,
            barcodes=[
                PersistenceBarcode(birth=0.0, death=1.5, dimension=0),
                PersistenceBarcode(birth=0.2, death=0.8, dimension=1),
            ],
            max_alpha=2.0,
            n_witnesses=500,
            n_landmarks=50,
            hyperbolic_distances_used=False,
        )
        assert len(payload.barcodes) == 2
        assert payload.hyperbolic_distances_used is False

    def test_valid_v4_persistence_with_hyperbolic(self, provenance: ProvenanceContext):
        payload = Phase3PersistencePayload(
            provenance=provenance,
            barcodes=[
                PersistenceBarcode(
                    birth=0.0,
                    death=2.1,
                    dimension=0,
                    generator_residues=["4obe:A:12", "4obe:A:15"],
                ),
            ],
            max_alpha=3.0,
            n_witnesses=1000,
            n_landmarks=100,
            hyperbolic_distances_used=True,
            curvature_c=1.0,
            residue_contributions=[
                Phase3ResidueContribution(
                    residue_id="4obe:A:12",
                    persistence_score=0.85,
                    max_barcode_length=2.1,
                    topological_significance=0.92,
                ),
            ],
            landmark_to_residue={"0": "4obe:A:12", "1": "4obe:A:15"},
        )
        assert payload.hyperbolic_distances_used is True
        assert payload.curvature_c == 1.0
        assert len(payload.residue_contributions) == 1


class TestNormalizerResult:
    def test_success_result(self):
        result = NormalizerResult(
            success=True,
            run_id="run_001",
            assets_created=5,
            asset_ids=["a1", "a2", "a3", "a4", "a5"],
        )
        assert result.success is True
        assert result.assets_created == 5

    def test_result_with_warnings(self):
        result = NormalizerResult(
            success=True,
            run_id="run_001",
            assets_created=3,
            asset_ids=["a1", "a2", "a3"],
            warnings=["Skipped invalid residue_id: bad_id"],
        )
        assert len(result.warnings) == 1
