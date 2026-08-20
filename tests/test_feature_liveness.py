"""Unit tests for feature liveness probes and barcode+slim refuse."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from science.training.config import TrainingConfig
from science.training.feature_liveness import (
    probe_barcode_liveness,
    probe_mp_liveness,
)
from science.training.mlflow_governance import build_governance_params


def _toy_prot(*, node_dim: int = 15) -> dict:
    n = 8
    x = torch.randn(n, node_dim)
    if node_dim > 4:
        x[:, 4:] = torch.randn(n, node_dim - 4) * 2.0
    edge = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    data = Data(x=x, edge_index=edge)
    data.degree = torch.ones(n)
    data.rho = x[:, 0]
    data.ss_onehot = torch.zeros(n, 3)
    data.ss_onehot[:, 2] = 1.0
    data.clustering = torch.zeros(n)
    return {"pdb_id": "4OBE", "chain": "A", "data": data, "residue_ids": [f"A:{i}:" for i in range(n)]}


class _ToyModel(torch.nn.Module):
    """Minimal model: cone_depth = sum of node features (barcode-sensitive)."""

    def __init__(self, in_dim: int = 15) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(in_dim, 1)
        self.log_c = torch.nn.Parameter(torch.tensor(0.5))

    @property
    def curvature(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.log_c) + 1e-4

    def forward(self, data: Data) -> dict:
        depth = self.lin(data.x)
        xy = data.x[:, :2]
        w = torch.softmax(data.x[:, :4], dim=-1)
        return {
            "cone_depth": depth,
            "hyp_projections_2d": xy,
            "expert_weights": w,
            "uncertainty": {"epistemic": depth.abs()},
        }


def test_barcode_liveness_detects_alive_channel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "experiments.training.v6.train_loop.prepare_training_batch",
        lambda model, prot, device, structural_disc_frozen=False: prot["data"],
    )
    model = _ToyModel(15)
    # Make barcode columns matter: weight them non-zero
    with torch.no_grad():
        model.lin.weight.zero_()
        model.lin.weight[0, 5] = 1.0
    report = probe_barcode_liveness(model, _toy_prot(node_dim=15), "cpu")
    assert report["alive"] is True
    assert float(report["delta_cone_depth"]) > 1e-6


def test_barcode_liveness_detects_dead_zero_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "experiments.training.v6.train_loop.prepare_training_batch",
        lambda model, prot, device, structural_disc_frozen=False: prot["data"],
    )
    model = _ToyModel(15)
    with torch.no_grad():
        model.lin.weight.zero_()
        model.lin.weight[0, 0] = 1.0  # only ρ column
    report = probe_barcode_liveness(model, _toy_prot(node_dim=15), "cpu")
    assert report["alive"] is False


def test_governance_emits_geometry_mode_params() -> None:
    cfg = TrainingConfig(
        corpus_manifest="manifests/v6_corpus_stage_a_small_v1.json",
        master_cold_lineage=True,
        structural_disc_frozen=False,
        use_dehydron_barcode=True,
    )
    params = build_governance_params(cfg, proteins=[{"pdb_id": "4OBE"}])
    assert params["structural_disc_frozen"] == "false"
    assert params["disc_layout_source"] == "gnn_learned"
    assert params["backbone_trainable"] == "true"
    assert params["use_dehydron_barcode"] == "true"


def test_slim_governance_marks_backbone_frozen() -> None:
    cfg = TrainingConfig(
        corpus_manifest="manifests/v6_corpus_stage_a_small_v1.json",
        slim_moe_structural_ssot=True,
        structural_disc_frozen=True,
    )
    params = build_governance_params(cfg, proteins=[{"pdb_id": "4OBE"}])
    assert params["structural_disc_frozen"] == "true"
    assert params["disc_layout_source"] == "structural_ssot_frozen"
    assert params["backbone_trainable"] == "false"
