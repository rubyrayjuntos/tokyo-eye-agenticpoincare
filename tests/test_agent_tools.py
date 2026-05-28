"""Tests for agent DTIE tools and viewport models."""

from __future__ import annotations

import pytest

from agent.models.viewport import (
    DirectiveAction,
    HighlightGroup,
    HighlightStyle,
    ViewportCapabilities,
    ViewportDirective,
    ViewportRegistration,
    ViewportState,
)
from agent.tools.dtie.tools import (
    ToolResult,
    compare_wt_mutant,
    get_high_uncertainty_residues,
    get_residue_state,
    get_source_leaks,
    run_gnn_inference,
    run_phase,
)


class TestViewportModels:
    def test_highlight_group(self):
        group = HighlightGroup(
            residue_ids=["4obe:A:12", "4obe:A:13"],
            color="#ff0000",
            style=HighlightStyle.PULSE,
            label="Source Leak",
        )
        assert len(group.residue_ids) == 2
        assert group.style == HighlightStyle.PULSE

    def test_viewport_directive(self):
        directive = ViewportDirective(
            action=DirectiveAction.HIGHLIGHT,
            structure_id="4obe",
            highlight_groups=[
                HighlightGroup(
                    residue_ids=["4obe:A:12"],
                    color="#ff6b6b",
                    label="Test",
                )
            ],
            message="Testing highlight",
        )
        assert directive.action == DirectiveAction.HIGHLIGHT
        assert directive.structure_id == "4obe"
        assert len(directive.highlight_groups) == 1

    def test_viewport_state(self):
        state = ViewportState(
            viewport_id="viewer_1",
            structure_id="4obe",
            current_metric="cone_depth",
            curvature=1.0,
        )
        assert state.viewport_id == "viewer_1"
        assert state.current_metric == "cone_depth"

    def test_viewport_registration(self):
        reg = ViewportRegistration(
            viewport_id="viewer_1",
            capabilities=ViewportCapabilities(
                supports_3d=True,
                max_nodes=3000,
            ),
            current_structure="4obe",
        )
        assert reg.capabilities.supports_3d is True
        assert reg.capabilities.max_nodes == 3000

    def test_all_directive_actions(self):
        """All directive actions should be valid enum values."""
        actions = [
            DirectiveAction.HIGHLIGHT,
            DirectiveAction.FOCUS,
            DirectiveAction.CLEAR_HIGHLIGHTS,
            DirectiveAction.SET_METRIC,
            DirectiveAction.SET_CURVATURE,
            DirectiveAction.TOGGLE_LABELS,
            DirectiveAction.FILTER_BY_SITE,
            DirectiveAction.SHOW_UNCERTAINTY,
            DirectiveAction.COMPARE_RUNS,
            DirectiveAction.ANNOTATE,
        ]
        assert len(actions) == 10


class TestDTIETools:
    """Test tools with a query-validating mock database."""

    @pytest.fixture
    def mock_db(self, validating_mock_db):
        """Query-validating mock DB with realistic responses registered."""
        validating_mock_db.register_response("v_agent_high_uncertainty", [
            {
                "residue_id": "4obe:A:12",
                "residue_index": 12,
                "residue_name": "G",
                "chain_label": "A",
                "epistemic_uncertainty": 0.45,
                "aleatoric_uncertainty": 0.12,
                "total_uncertainty": 0.57,
                "cone_depth": 2.1,
                "model_version": "GOSPConeMapper-v4",
                "run_id": "run_001",
                "pipeline_name": "dtie_v4",
            }
        ])
        validating_mock_db.register_response("v_agent_residue_state", [
            {
                "residue_id": "4obe:A:12",
                "residue_index": 12,
                "residue_name": "G",
                "chain_label": "A",
                "cone_depth": 2.1,
                "epistemic_uncertainty": 0.45,
            }
        ])
        validating_mock_db.register_response("fact_gnn_node_embedding", [
            {
                "residue_id": "4obe:A:12",
                "embedding": [0.1] * 32,
                "cone_depth": 2.1,
                "epistemic_uncertainty": 0.45,
                "residue_index": 12,
                "chain_label": "A",
                "cnt": 165,
            }
        ])
        return validating_mock_db

    @pytest.mark.asyncio
    async def test_run_gnn_inference_no_db(self):
        """Without DB, tool returns failure gracefully."""
        result = await run_gnn_inference(structure_id="4obe", model_version="v4")
        assert result.success is False
        assert "No database" in result.message

    @pytest.mark.asyncio
    async def test_run_phase(self, mock_db):
        result = await run_phase(
            structure_id="4obe",
            phase="phase3_persistence",
            model_version="v4",
            db=mock_db,
        )
        assert result.success is True
        assert result.data["phase"] == "phase3_persistence"

    @pytest.mark.asyncio
    async def test_get_source_leaks(self, mock_db):
        result = await get_source_leaks(
            structure_id="4obe",
            uncertainty_threshold=0.3,
            min_depth=1.5,
            db=mock_db,
        )
        assert result.success is True
        assert len(result.viewport_directives) == 1
        directive = result.viewport_directives[0]
        assert directive.action == DirectiveAction.HIGHLIGHT

    @pytest.mark.asyncio
    async def test_get_high_uncertainty(self, mock_db):
        result = await get_high_uncertainty_residues(
            structure_id="4obe",
            top_n=10,
            uncertainty_type="epistemic",
            db=mock_db,
        )
        assert result.success is True
        assert result.data["count"] == 1
        assert len(result.viewport_directives) == 1
        assert result.viewport_directives[0].action == DirectiveAction.SHOW_UNCERTAINTY

    @pytest.mark.asyncio
    async def test_get_residue_state(self, mock_db):
        result = await get_residue_state(
            structure_id="4obe",
            residue_ids=["4obe:A:12"],
            db=mock_db,
        )
        assert result.success is True
        assert result.data["count"] == 1

    @pytest.mark.asyncio
    async def test_compare_wt_mutant(self, mock_db):
        result = await compare_wt_mutant(
            wt_structure_id="4obe",
            mutant_structure_id="4obe_g12d",
            db=mock_db,
        )
        assert result.success is True
        assert len(result.viewport_directives) == 1
        assert result.viewport_directives[0].action == DirectiveAction.COMPARE_RUNS
