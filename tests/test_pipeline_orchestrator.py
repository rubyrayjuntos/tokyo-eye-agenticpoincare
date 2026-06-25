"""Tests for the DTIE pipeline orchestrator."""

from __future__ import annotations

import pytest

from science.dtie.v5.orchestrator.pipeline import (
    DTIEOrchestrator,
    PipelineConfig,
    PHASE_REGISTRY,
    _REQUIRED_PHASES,
    _load_phase_function,
)
from science.dtie.common.interfaces import PhaseResult


class TestPipelineConfig:
    def test_source_leak_only_disables_optional_phases(self):
        config = PipelineConfig(structure_id="4obe", source_leak_only=True)
        assert config.run_phase1 is False
        assert config.run_phase2 is False
        assert config.run_phase35 is False
        assert config.run_phase4 is False
        assert config.run_phase5 is False
        assert config.run_phase6 is False
        # These should remain enabled
        assert config.run_gnn is True
        assert config.run_phase3 is True
        assert config.detect_source_leaks is True

    def test_is_full_pipeline_property(self):
        full = PipelineConfig(structure_id="4obe", source_leak_only=False)
        assert full.is_full_pipeline is True

        partial = PipelineConfig(structure_id="4obe", source_leak_only=True)
        assert partial.is_full_pipeline is False

    def test_default_thresholds(self):
        config = PipelineConfig(structure_id="4obe")
        assert config.uncertainty_threshold == 0.3
        assert config.depth_threshold == 1.5
        assert config.n_landmarks == 50


class TestPhaseRegistry:
    def test_all_expected_phases_registered(self):
        expected = [
            "phase1_witness_embedding",
            "phase2_vulnerability_scan",
            "phase35_topological_lift",
            "phase4_resistance_mapping",
            "phase5_pharmacophore",
            "phase6a_virtual_screening",
            "phase6b_binding_affinity",
            "phase6c_admet_filter",
            "phase6d_state_selectivity",
        ]
        for phase in expected:
            assert phase in PHASE_REGISTRY, f"Phase {phase} not in registry"

    def test_registry_entries_are_valid_tuples(self):
        for name, entry in PHASE_REGISTRY.items():
            assert isinstance(entry, tuple), f"{name} entry is not a tuple"
            assert len(entry) == 2, f"{name} entry should be (module, func)"
            module_path, func_name = entry
            assert "." in module_path, f"{name} module_path looks invalid"
            assert func_name.startswith("execute_") or func_name.startswith("run_"), \
                f"{name} func_name '{func_name}' doesn't follow naming convention"

    def test_load_unknown_phase_returns_none(self):
        result = _load_phase_function("nonexistent_phase")
        assert result is None


class TestSuccessEvaluation:
    @pytest.fixture
    def orchestrator(self, validating_mock_db):
        return DTIEOrchestrator(db=validating_mock_db)

    def test_all_required_pass_is_success(self, orchestrator):
        config = PipelineConfig(structure_id="4obe")
        results = {
            "gnn_inference": PhaseResult(
                phase_name="gnn_inference", structure_id="4obe",
                model_version="v5", success=True, outputs={},
            ),
            "source_leak_detection": PhaseResult(
                phase_name="source_leak_detection", structure_id="4obe",
                model_version="v5", success=True, outputs={},
            ),
            "phase1": PhaseResult(
                phase_name="phase1", structure_id="4obe",
                model_version="v5", success=False, outputs={},
            ),
        }
        # Optional phase1 failed, but required phases passed
        assert orchestrator._evaluate_success(results, config) is True

    def test_required_phase_failure_is_failure(self, orchestrator):
        config = PipelineConfig(structure_id="4obe")
        results = {
            "gnn_inference": PhaseResult(
                phase_name="gnn_inference", structure_id="4obe",
                model_version="v5", success=True, outputs={},
            ),
            "source_leak_detection": PhaseResult(
                phase_name="source_leak_detection", structure_id="4obe",
                model_version="v5", success=False, outputs={},
            ),
        }
        assert orchestrator._evaluate_success(results, config) is False

    def test_empty_results_is_failure(self, orchestrator):
        config = PipelineConfig(structure_id="4obe")
        assert orchestrator._evaluate_success({}, config) is False

    def test_only_optional_phases_with_one_success(self, orchestrator):
        config = PipelineConfig(structure_id="4obe")
        results = {
            "phase1": PhaseResult(
                phase_name="phase1", structure_id="4obe",
                model_version="v5", success=True, outputs={},
            ),
        }
        # No required phases present, but at least one succeeded
        assert orchestrator._evaluate_success(results, config) is True


class TestOrchestratorSourceLeakDetection:
    @pytest.mark.asyncio
    async def test_detect_source_leaks_queries_correct_table(self, validating_mock_db):
        """Source-leak detection should query fact_gnn_node_embedding with filters."""
        validating_mock_db.register_response("fact_gnn_node_embedding", [
            {"residue_id": "4obe:A:12", "cone_depth": 2.1, "epistemic_uncertainty": 0.45},
        ])

        orchestrator = DTIEOrchestrator(db=validating_mock_db)
        config = PipelineConfig(
            structure_id="4obe",
            uncertainty_threshold=0.3,
            depth_threshold=1.5,
        )

        result = await orchestrator._detect_source_leaks(config, "test_run")
        assert result.success is True
        assert result.outputs["source_leak_count"] == 1
        assert "4obe:A:12" in result.outputs["source_leak_residues"]

        # Verify the query hit the right table with right filters
        validating_mock_db.assert_query_executed_containing("fact_gnn_node_embedding")
        validating_mock_db.assert_query_executed_containing("epistemic_uncertainty")


class TestOrchestratorBindingSiteScan:
    @pytest.mark.asyncio
    async def test_binding_site_scan_not_in_orchestrator_results(self, validating_mock_db):
        """Binding-site scan should remain a separate compute call, not an orchestrator phase."""
        validating_mock_db.register_response("embedding_space", [])

        orchestrator = DTIEOrchestrator(db=validating_mock_db)
        config = PipelineConfig(
            structure_id="4obe",
            run_gnn=False,
            run_phase1=False,
            run_phase2=False,
            run_phase3=False,
            run_phase35=False,
            run_phase4=False,
            run_phase5=False,
            run_phase6=False,
            detect_source_leaks=False,
            identify_allosteric_sites=False,
        )

        result = await orchestrator.run(config)
        assert "binding_site_scan" not in result.phase_results
