"""Tests for GNN interactive NGL viewer generation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from science.dtie.common.interfaces import GNNInferenceResult, GNNNodeOutput
from shared.gnn_viewer_paths import interactive_viewer_enabled
from science.dtie.v6.visualization.interactive_viewer import (
    build_residue_channel_lookup,
    write_interactive_html,
    _format_pdb_atom_name,
    _residue_name_3letter,
)


def _sample_node(res_index: int, *, epistemic: float, depth: float) -> GNNNodeOutput:
    return GNNNodeOutput(
        residue_index=res_index,
        chain_label="A",
        input_features=np.zeros(4),
        projections=np.zeros(8),
        cone_depth=depth,
        cone_width=0.5,
        epistemic_uncertainty=epistemic,
        aleatoric_uncertainty=0.1,
        expert_weights=np.array([0.1, 0.7, 0.1, 0.1]),
    )


class TestInteractiveViewerHelpers:
    def test_build_residue_channel_lookup(self) -> None:
        nodes = [_sample_node(1, epistemic=1.0, depth=2.0), _sample_node(2, epistemic=2.0, depth=4.0)]
        lookup = build_residue_channel_lookup(nodes)
        assert lookup[("A", 1)].epistemic_uncertainty == 1.0
        assert lookup[("A", 2)].cone_depth == 4.0

    def test_residue_name_3letter_from_one_letter(self) -> None:
        assert _residue_name_3letter("V") == "VAL"
        assert _residue_name_3letter("ALA") == "ALA"

    def test_format_pdb_atom_name_locant_vs_backbone(self) -> None:
        assert _format_pdb_atom_name("CG1") == "CG1 "
        assert _format_pdb_atom_name("CA") == " CA "
        assert _format_pdb_atom_name("N") == " N  "

    def test_write_interactive_html_contains_ngl_and_pdb(self, tmp_path: Path) -> None:
        pdb = (
            "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  0.50 88.00           C\n"
            "END\n"
        )
        out = tmp_path / "9est_interactive.html"
        write_interactive_html(
            structure_id="9est",
            pdb_text=pdb,
            model_version="GOSPConeMapper-v6",
            output_path=out,
            checkpoint_path="checkpoints/v6/runs/lever_a_clean_slate_v1/v6_best_disc.pt",
        )
        text = out.read_text(encoding="utf-8")
        assert "NGL.Stage" in text
        assert "addRepresentation(\"cartoon\"" in text
        assert "sele: \"polymer\"" in text
        assert "9EST" in text
        assert "GOSPConeMapper-v6" in text
        assert "ATOM      1  CA  ALA A   1" in text

    def test_interactive_viewer_enabled_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GNN_INTERACTIVE_HTML", raising=False)
        assert interactive_viewer_enabled() is True
        monkeypatch.setenv("GNN_INTERACTIVE_HTML", "false")
        assert interactive_viewer_enabled() is False


@pytest.mark.asyncio
async def test_generate_gnn_interactive_viewer_registers_asset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GNN_VIEWER_OUTPUT_DIR", str(tmp_path))

    nodes = [_sample_node(i, epistemic=1.0 + i * 0.1, depth=float(i)) for i in range(1, 4)]
    gnn_result = GNNInferenceResult(
        structure_id="9est",
        model_version="GOSPConeMapper-v6",
        checkpoint_path="checkpoints/v6/test.pt",
        nodes=nodes,
        curvature=1.2,
    )

    db = MagicMock()
    db.fetch_all = AsyncMock(
        return_value=[
            {
                "atom_name": "CA",
                "element": "C",
                "x": 1.0,
                "y": 2.0,
                "z": 3.0,
                "residue_index": i,
                "residue_name": "ALA",
                "chain_label": "A",
            }
            for i in range(1, 4)
        ]
    )

    normalizer = MagicMock()
    normalizer.register_file_asset = AsyncMock()

    import data.normalizer.core as core_mod

    monkeypatch.setattr(core_mod, "Normalizer", lambda db, caller_identity: normalizer)

    from science.dtie.v6.visualization.interactive_viewer import generate_gnn_interactive_viewer

    meta = await generate_gnn_interactive_viewer(
        db,
        gnn_result=gnn_result,
        run_id="run_test_001",
        code_version="test",
    )

    html_path = Path(meta["html_path"])
    pdb_path = Path(meta["pdb_path"])
    assert html_path.is_file()
    assert pdb_path.is_file()
    assert meta["viewer_url"] == "/api/structures/9est/gnn-viewer"
    normalizer.register_file_asset.assert_awaited_once()
