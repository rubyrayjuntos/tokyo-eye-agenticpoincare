"""Containment-edge liveness + metrics.json persistence contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from science.dtie.v66.chem_edge_graph import attach_chem_edge_graph
from science.dtie.v66.containment_edge_graph import (
    NUM_ROLE_RELATIONS_WITH_CONTAINMENT,
    ROLE_CONTAIN_DOWN,
    attach_containment_edge_graph,
)
from science.dtie.v66.role_edge_graph import attach_role_edge_graph
from science.dtie.v66.thermo_edge_features import GEO_DIM
from science.training.checkpoint import CheckpointManager
from science.training.feature_liveness import probe_containment_edge_liveness


def _role_chem_contain_prot(
    *,
    with_helix: bool,
    pdb_id: str = "1LYZ",
) -> dict:
    n = 10
    coords = torch.zeros(n, 3)
    coords[:, 0] = torch.arange(n, dtype=torch.float32) * 3.8
    x = torch.zeros(n, 4)
    x[:, 0] = torch.linspace(20, 8, n)
    x[:, 1] = (x[:, 0] < 13).float()
    data = Data(
        x=x,
        edge_index=torch.zeros(2, 0, dtype=torch.long),
        edge_attr=torch.zeros(0, 4),
    )
    data.rho = x[:, 0].clone()
    residue_ids = [f"A:{i + 1}:" for i in range(n)]
    pdb_text = (
        "HELIX    1   1 ALA A    2  ALA A    8  1                                   7\n"
        "END\n"
        if with_helix
        else "END\n"
    )
    return {
        "pdb_id": pdb_id,
        "chain": "A",
        "structure_id": pdb_id.lower(),
        "data": data,
        "ca_coords": coords,
        "residue_ids": residue_ids,
        "covalent_bonds": [],
        "pdb_text": pdb_text,
    }


class _ContainmentAwareToy(torch.nn.Module):
    """Cone depth depends on contain_down one-hot mass when containment_edge_mp is on."""

    def __init__(self, *, containment_edge_mp: bool = True) -> None:
        super().__init__()
        self.containment_edge_mp = containment_edge_mp
        self.chem_edge_mp = True
        self.role_edge_mp = True
        self.role_coupling_edges = False
        self.log_c = torch.nn.Parameter(torch.tensor(0.5))
        self.convs = torch.nn.ModuleList([torch.nn.Module()])
        self.convs[0].radial_mlps = torch.nn.ModuleList(
            [torch.nn.Linear(1, 4) for _ in range(NUM_ROLE_RELATIONS_WITH_CONTAINMENT)]
        )

    @property
    def curvature(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.log_c) + 1e-4

    def forward(self, data: Data) -> dict:
        contain_mass = data.edge_attr[:, GEO_DIM + ROLE_CONTAIN_DOWN].sum()
        n = data.x.size(0)
        depth = data.x[:, :1] + contain_mass
        return {
            "cone_depth": depth,
            "hyp_projections_2d": data.x[:, :2],
            "expert_weights": torch.softmax(data.x[:, :4], dim=-1),
            "uncertainty": {"epistemic": depth.abs()},
        }


def _prep_containment(model, prot, device, structural_disc_frozen=False):
    data = prot["data"].clone()
    data = attach_role_edge_graph(data, prot["ca_coords"], residue_ids=prot["residue_ids"])
    data = attach_chem_edge_graph(
        data,
        prot["ca_coords"],
        prot["covalent_bonds"],
        residue_ids=prot["residue_ids"],
        structure_id=prot["structure_id"],
        chain_label=prot["chain"],
    )
    return attach_containment_edge_graph(
        data,
        prot["ca_coords"],
        prot["pdb_text"],
        residue_ids=prot["residue_ids"],
        force=True,
    )


def test_containment_liveness_alive_when_helix_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "experiments.training.v6.train_loop.prepare_training_batch",
        _prep_containment,
    )
    model = _ContainmentAwareToy(containment_edge_mp=True)
    report = probe_containment_edge_liveness(
        model, [_role_chem_contain_prot(with_helix=True)], "cpu"
    )
    assert report["skipped"] is False
    assert report["alive"] is True
    assert float(report["n_contain_down_edges"]) >= 1.0
    assert float(report["n_contain_up_edges"]) >= 1.0
    assert float(report["delta_cone_depth"]) > 1e-6


def test_containment_liveness_skips_empty_containment_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "experiments.training.v6.train_loop.prepare_training_batch",
        _prep_containment,
    )
    model = _ContainmentAwareToy(containment_edge_mp=True)
    report = probe_containment_edge_liveness(
        model, [_role_chem_contain_prot(with_helix=False, pdb_id="4OBE")], "cpu"
    )
    assert report["skipped"] is True
    assert report["reason"] == "no_containment_edges_in_corpus_sample"


def test_containment_liveness_keys_persist_into_metrics_json(tmp_path: Path) -> None:
    """Regression: liveness must land in metrics.json as liveness_containment_*."""
    log_metrics = {
        "liveness_containment_alive": 1.0,
        "liveness_containment_delta_cone_depth": 0.42,
        "liveness_containment_radial_var_contain_down": 0.01,
        "liveness_containment_n_contain_down_edges": 3.0,
        "loss": 1.23,
    }
    entry: dict = {"epoch": 1, "phase": 1}
    for _lk, _lv in log_metrics.items():
        if _lk.startswith("liveness_") and isinstance(_lv, (int, float)):
            entry[_lk] = float(_lv)

    mgr = CheckpointManager(tmp_path, checkpoint_prefix="contain", protein_count=12)
    path = mgr.write_metrics_log([entry])
    loaded = json.loads(path.read_text())
    assert loaded[0]["liveness_containment_alive"] == 1.0
    assert loaded[0]["liveness_containment_delta_cone_depth"] == 0.42
    assert loaded[0]["liveness_containment_radial_var_contain_down"] == 0.01
    assert "loss" not in loaded[0]
