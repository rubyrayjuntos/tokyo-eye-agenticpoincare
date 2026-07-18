"""Chem-edge liveness + metrics.json persistence + chem init-seed isolation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from science.dtie.common.isolated_init import isolated_torch_seed
from science.dtie.v66.chem_edge_graph import (
    EDGE_ATTR_CHEM_DIM,
    NUM_ROLE_RELATIONS_WITH_CHEM,
    ROLE_DISULF,
    attach_chem_edge_graph,
)
from science.dtie.v66.gnn.model import GOSPConeMapperV66
from science.dtie.v66.role_edge_graph import attach_role_edge_graph
from science.dtie.v66.thermo_edge_features import GEO_DIM
from science.training.checkpoint import CheckpointManager
from science.training.feature_liveness import probe_chem_edge_liveness


def _role_chem_prot(
    *,
    with_disulf: bool,
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
    bonds = []
    if with_disulf:
        bonds = [
            {
                "residue_id_1": f"{pdb_id.lower()}:A:2",
                "residue_id_2": f"{pdb_id.lower()}:A:8",
                "bond_type": "disulf",
            }
        ]
    return {
        "pdb_id": pdb_id,
        "chain": "A",
        "structure_id": pdb_id.lower(),
        "data": data,
        "ca_coords": coords,
        "residue_ids": residue_ids,
        "covalent_bonds": bonds,
    }


class _ChemAwareToy(torch.nn.Module):
    """Cone depth depends on chem one-hot mass when chem_edge_mp is on."""

    def __init__(self, *, chem_edge_mp: bool = True) -> None:
        super().__init__()
        self.chem_edge_mp = chem_edge_mp
        self.role_edge_mp = True
        self.role_coupling_edges = False
        self.log_c = torch.nn.Parameter(torch.tensor(0.5))
        # Minimal radial MLP bank so variance probe has something to call.
        self.convs = torch.nn.ModuleList(
            [
                torch.nn.Module()
            ]
        )
        self.convs[0].radial_mlps = torch.nn.ModuleList(
            [torch.nn.Linear(1, 4) for _ in range(NUM_ROLE_RELATIONS_WITH_CHEM)]
        )

    @property
    def curvature(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.log_c) + 1e-4

    def forward(self, data: Data) -> dict:
        chem_mass = data.edge_attr[:, GEO_DIM + ROLE_DISULF].sum()
        n = data.x.size(0)
        depth = data.x[:, :1] + chem_mass
        return {
            "cone_depth": depth,
            "hyp_projections_2d": data.x[:, :2],
            "expert_weights": torch.softmax(data.x[:, :4], dim=-1),
            "uncertainty": {"epistemic": depth.abs()},
        }


def test_chem_liveness_alive_when_disulf_present(monkeypatch: pytest.MonkeyPatch) -> None:
    def _prep(model, prot, device, structural_disc_frozen=False):
        data = prot["data"].clone()
        data = attach_role_edge_graph(data, prot["ca_coords"], residue_ids=prot["residue_ids"])
        return attach_chem_edge_graph(
            data,
            prot["ca_coords"],
            prot["covalent_bonds"],
            residue_ids=prot["residue_ids"],
            structure_id=prot["structure_id"],
            chain_label=prot["chain"],
        )

    monkeypatch.setattr(
        "experiments.training.v6.train_loop.prepare_training_batch",
        _prep,
    )
    model = _ChemAwareToy(chem_edge_mp=True)
    report = probe_chem_edge_liveness(
        model, [_role_chem_prot(with_disulf=True)], "cpu"
    )
    assert report["skipped"] is False
    assert report["alive"] is True
    assert float(report["n_disulf_edges"]) == 2.0  # directed
    assert float(report["delta_cone_depth"]) > 1e-6


def test_chem_liveness_skips_empty_chem_set(monkeypatch: pytest.MonkeyPatch) -> None:
    def _prep(model, prot, device, structural_disc_frozen=False):
        data = prot["data"].clone()
        data = attach_role_edge_graph(data, prot["ca_coords"], residue_ids=prot["residue_ids"])
        return attach_chem_edge_graph(
            data,
            prot["ca_coords"],
            prot["covalent_bonds"],
            residue_ids=prot["residue_ids"],
            structure_id=prot["structure_id"],
            chain_label=prot["chain"],
        )

    monkeypatch.setattr(
        "experiments.training.v6.train_loop.prepare_training_batch",
        _prep,
    )
    model = _ChemAwareToy(chem_edge_mp=True)
    report = probe_chem_edge_liveness(
        model, [_role_chem_prot(with_disulf=False, pdb_id="4OBE")], "cpu"
    )
    assert report["skipped"] is True
    assert report["reason"] == "no_chem_edges_in_corpus_sample"


def test_chem_liveness_keys_persist_into_metrics_json(tmp_path: Path) -> None:
    """Regression for barcode-class gap: liveness must land in metrics.json."""
    # Mirror stage_runner entry assembly for chem keys.
    log_metrics = {
        "liveness_chem_alive": 1.0,
        "liveness_chem_delta_cone_depth": 0.42,
        "liveness_chem_radial_var_disulf": 0.01,
        "liveness_chem_n_disulf_edges": 8.0,
        "loss": 1.23,  # non-liveness — must NOT be auto-copied by the liveness filter
    }
    entry: dict = {"epoch": 1, "phase": 1}
    for _lk, _lv in log_metrics.items():
        if _lk.startswith("liveness_") and isinstance(_lv, (int, float)):
            entry[_lk] = float(_lv)

    mgr = CheckpointManager(tmp_path, checkpoint_prefix="chem", protein_count=12)
    path = mgr.write_metrics_log([entry])
    loaded = json.loads(path.read_text())
    assert loaded[0]["liveness_chem_alive"] == 1.0
    assert loaded[0]["liveness_chem_delta_cone_depth"] == 0.42
    assert loaded[0]["liveness_chem_radial_var_disulf"] == 0.01
    assert "loss" not in loaded[0]  # only liveness_* copied by that loop


def test_prototype_bank_identical_chem_on_vs_off_with_init_seed() -> None:
    """Prerequisite #2: 7-rel chem MLP bank must not scramble prototype init."""
    kwargs = dict(
        node_dim=4,
        hidden=64,
        num_layers=2,
        num_experts=4,
        role_edge_mp=True,
        hyperbolic_gate=True,
        topology_only_gate=True,
        init_seed=17,
    )
    with isolated_torch_seed(17):
        off = GOSPConeMapperV66(chem_edge_mp=False, **kwargs)
    with isolated_torch_seed(17):
        on = GOSPConeMapperV66(chem_edge_mp=True, **kwargs)

    p_off = off.gate.prototype_bank.prototype_tangent
    p_on = on.gate.prototype_bank.prototype_tangent
    assert torch.equal(p_off, p_on), (
        f"prototype_tangent diverged chem-off vs chem-on "
        f"(max |Δ|={(p_off - p_on).abs().max().item()})"
    )
    assert torch.equal(
        off.gate.topo_encoder[0].weight,
        on.gate.topo_encoder[0].weight,
    )
    assert len(on.convs[0].radial_mlps) == NUM_ROLE_RELATIONS_WITH_CHEM
    assert len(off.convs[0].radial_mlps) < NUM_ROLE_RELATIONS_WITH_CHEM
