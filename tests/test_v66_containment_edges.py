"""Path B containment: SSE parent nodes + contain_up/down directed edges."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from experiments.training.v6.corpus import stamp_prot_pdb_paths
from experiments.training.v6.train_loop import (
    _pad_containment_parent_node_attrs,
    _slice_residue_outputs,
    prepare_training_batch,
    residue_node_count,
    resolve_prot_pdb_path,
)
from science.dtie.common.isolated_init import isolated_torch_seed
from science.dtie.v66.chem_edge_graph import EDGE_ATTR_CHEM_DIM
from science.dtie.v66.gnn.model import GOSPConeMapperV66
from science.dtie.v66.containment_edge_graph import (
    EDGE_ATTR_CONTAIN_DIM,
    NUM_ROLE_RELATIONS_WITH_CONTAINMENT,
    ROLE_CONTAIN_DOWN,
    ROLE_CONTAIN_UP,
    attach_containment_edge_graph,
    pad_chem_edge_attr_for_containment,
)
from science.dtie.v66.thermo_edge_features import GEO_DIM


def _toy_chem_ready_data(n: int = 6, feat_dim: int = 3) -> Data:
    x = torch.randn(n, feat_dim)
    ei = torch.zeros(2, 0, dtype=torch.long)
    ea = torch.zeros(0, EDGE_ATTR_CHEM_DIM)
    data = Data(x=x, edge_index=ei, edge_attr=ea)
    data.role_edge_graph = True
    data.chem_edge_graph = True
    return data


def test_pad_chem_edge_attr_inserts_containment_onehot_slots() -> None:
    ea = torch.zeros(2, EDGE_ATTR_CHEM_DIM)
    ea[:, GEO_DIM + 3] = 1.0  # ribbon slot
    aux_col = GEO_DIM + 7
    ea[:, aux_col] = 0.42
    padded = pad_chem_edge_attr_for_containment(ea)
    assert padded.shape == (2, EDGE_ATTR_CONTAIN_DIM)
    assert torch.allclose(
        padded[:, GEO_DIM : GEO_DIM + 7],
        ea[:, GEO_DIM : GEO_DIM + 7],
    )
    assert torch.allclose(
        padded[:, GEO_DIM + 7 : GEO_DIM + 9],
        torch.zeros(2, 2),
    )
    assert torch.allclose(padded[:, GEO_DIM + 9], ea[:, aux_col])


def test_attach_appends_parent_and_directed_relations() -> None:
    data = _toy_chem_ready_data()
    ca = torch.randn(6, 3)
    residue_ids = [f"A:{i}:" for i in range(10, 16)]
    pdb = (
        "HELIX    1   1 ALA A   10  ALA A   12  1                                   3\n"
        "END\n"
    )
    out = attach_containment_edge_graph(
        data, ca, pdb, residue_ids=residue_ids, force=True
    )
    assert out.n_residue_nodes == 6
    assert out.n_parent_nodes == 1
    assert out.containment_edge_graph is True
    assert out.x.shape[0] == 7
    assert torch.allclose(out.x[6], data.x[:3].mean(0), atol=1e-5)
    oh = out.edge_attr[:, GEO_DIM:]
    assert (oh[:, ROLE_CONTAIN_DOWN] > 0).any()
    assert (oh[:, ROLE_CONTAIN_UP] > 0).any()
    assert out.containment_edge_counts["contain_down"] == 3
    assert out.containment_edge_counts["contain_up"] == 3


def test_attach_sets_n_residue_nodes_for_loss_slice() -> None:
    """After attach with 1 parent, n_residue_nodes == original N (loss slice key)."""
    data = _toy_chem_ready_data()
    ca = torch.randn(6, 3)
    residue_ids = [f"A:{i}:" for i in range(10, 16)]
    pdb = (
        "HELIX    1   1 ALA A   10  ALA A   12  1                                   3\n"
        "END\n"
    )
    out = attach_containment_edge_graph(
        data, ca, pdb, residue_ids=residue_ids, force=True
    )
    assert out.n_residue_nodes == 6
    assert out.x.shape[0] == 7
    assert residue_node_count(out, {"n_residues": 6}) == 6


def test_slice_residue_outputs_drops_parent_rows() -> None:
    n_res, n_tot = 6, 7
    output = {
        "cone_depth": torch.randn(n_tot, 1),
        "x_hyp": torch.randn(n_tot, 8),
        "hyp_projections_2d": torch.randn(n_tot, 2),
        "expert_weights": torch.randn(n_tot, 2),
        "uncertainty": {
            "epistemic": torch.randn(n_tot, 1),
            "aleatoric": torch.randn(n_tot, 1),
        },
        "capacity_loss": torch.tensor(0.1),
        "routing_entropy": torch.tensor(1.2),
    }
    sliced = _slice_residue_outputs(output, n_res)
    assert sliced["cone_depth"].shape[0] == n_res
    assert sliced["x_hyp"].shape[0] == n_res
    assert sliced["hyp_projections_2d"].shape[0] == n_res
    assert sliced["expert_weights"].shape[0] == n_res
    assert sliced["uncertainty"]["epistemic"].shape[0] == n_res
    assert sliced["uncertainty"]["aleatoric"].shape[0] == n_res
    assert sliced["capacity_loss"].shape == ()
    assert torch.equal(sliced["cone_depth"], output["cone_depth"][:n_res])


def test_pad_containment_parent_attrs_matches_grown_x() -> None:
    data = _toy_chem_ready_data()
    n = 6
    data.structural_z_disc = torch.randn(n, 2)
    data.clustering = torch.rand(n)
    data.degree = torch.ones(n)
    data.ss_onehot = torch.zeros(n, 3)
    data.ss_onehot[:, 2] = 1.0
    data.rho = data.x[:, 0].clone()
    ca = torch.randn(n, 3)
    residue_ids = [f"A:{i}:" for i in range(10, 16)]
    pdb = (
        "HELIX    1   1 ALA A   10  ALA A   12  1                                   3\n"
        "END\n"
    )
    out = attach_containment_edge_graph(
        data, ca, pdb, residue_ids=residue_ids, force=True
    )
    assert out.x.shape[0] == 7
    assert out.structural_z_disc.shape[0] == n  # not padded yet
    padded = _pad_containment_parent_node_attrs(out)
    assert padded.structural_z_disc.shape[0] == 7
    assert torch.allclose(padded.structural_z_disc[n], torch.zeros(2))
    assert padded.clustering.shape[0] == 7
    assert padded.degree.shape[0] == 7
    assert padded.ss_onehot.shape[0] == 7
    assert padded.rho.shape[0] == 7


def test_collect_edge_telemetry_residue_slice_with_parent_edges() -> None:
    """Path B: residue-sliced outputs + parent MP edges must not IndexError."""
    from science.training.edge_telemetry import collect_edge_telemetry

    n_res, n_parent = 8, 2
    n_total = n_res + n_parent
    # Residue–residue contacts + containment edges into parents.
    ei = torch.tensor(
        [
            [0, 1, 2, 3, 0, 1, 4, 5],
            [1, 2, 3, 0, n_res, n_res + 1, n_res, n_res + 1],
        ],
        dtype=torch.long,
    )
    ea = torch.zeros(ei.size(1), EDGE_ATTR_CONTAIN_DIM)
    ea[:, 3] = 1.0
    data = Data(x=torch.randn(n_total, 3), edge_index=ei, edge_attr=ea)
    data.n_residue_nodes = n_res
    data.n_parent_nodes = n_parent
    data.containment_edge_graph = True

    class _Conv(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.radial_mlp = torch.nn.Sequential(
                torch.nn.Linear(1, 4),
                torch.nn.ReLU(),
                torch.nn.Linear(4, 8),
            )

    class _Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.convs = torch.nn.ModuleList([_Conv()])

    out = {
        "uncertainty": {
            "epistemic": torch.rand(n_res, 1),
            "aleatoric": torch.rand(n_res, 1),
        },
        "cone_depth": torch.rand(n_res, 1),
        "expert_weights": torch.softmax(torch.randn(n_res, 4), dim=-1),
    }
    ca = np.random.default_rng(0).normal(size=(n_res, 3)).astype(np.float64)
    rec = collect_edge_telemetry(
        _Model(),
        data,
        out,
        structure_id="TOY",
        ca_coords=ca,
    )
    assert rec.n_nodes == n_res
    assert rec.n_edges == 4  # containment edges dropped
    assert math.isfinite(rec.edge_epistemic_var_mean)


def test_stamp_prot_pdb_paths_restamps_cache_without_pdb_path(tmp_path: Path) -> None:
    """Corpus caches omit pdb_path; Path B needs restamp from --pdb-dir."""
    pdb = tmp_path / "1MBN.pdb"
    pdb.write_text("END\n")
    proteins = [{"pdb_id": "1MBN", "pdb_path": None, "pdb_dir": None}]
    stamp_prot_pdb_paths(proteins, tmp_path)
    assert proteins[0]["pdb_dir"] == str(tmp_path)
    assert proteins[0]["pdb_path"] == str(pdb)
    assert resolve_prot_pdb_path(proteins[0]) == str(pdb)


def test_resolve_prot_pdb_path_via_pdb_dir_when_path_missing(tmp_path: Path) -> None:
    """resolve_prot_pdb_path uses prot['pdb_dir'] when pdb_path is unset."""
    pdb = tmp_path / "4OBE.pdb"
    pdb.write_text("END\n")
    prot = {"pdb_id": "4OBE", "pdb_dir": str(tmp_path)}
    assert resolve_prot_pdb_path(prot) == str(pdb)


def test_prepare_training_batch_containment_attach_and_pad(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """prepare_training_batch wires containment after chem and pads structural_z."""
    n = 6
    x = torch.randn(n, 3)
    ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    ea = torch.zeros(2, EDGE_ATTR_CHEM_DIM)
    data = Data(x=x, edge_index=ei, edge_attr=ea)
    data.structural_z_disc = torch.randn(n, 2)
    data.structural_z_disc_frozen = True
    data.clustering = torch.rand(n)

    pdb_text = (
        "HELIX    1   1 ALA A   10  ALA A   12  1                                   3\n"
        "END\n"
    )
    pdb_path = tmp_path / "TOY1.pdb"
    pdb_path.write_text(pdb_text)

    prot = {
        "pdb_id": "TOY1",
        "chain": "A",
        "data": data,
        "ca_coords": torch.randn(n, 3),
        "residue_ids": [f"A:{i}:" for i in range(10, 16)],
        "n_residues": n,
        "pdb_path": str(pdb_path),
        "covalent_bonds": [],
        "target_dehydron": torch.zeros(n),
    }

    def _fake_role(data_in, *_args, **_kwargs):
        data_in.role_edge_graph = True
        return data_in

    def _fake_chem(data_in, *_args, **_kwargs):
        data_in.chem_edge_graph = True
        if data_in.edge_attr is None or data_in.edge_attr.size(-1) < EDGE_ATTR_CHEM_DIM:
            data_in.edge_attr = torch.zeros(
                data_in.edge_index.size(1), EDGE_ATTR_CHEM_DIM
            )
        return data_in

    monkeypatch.setattr(
        "science.dtie.v66.role_edge_graph.attach_role_edge_graph", _fake_role
    )
    monkeypatch.setattr(
        "science.dtie.v66.role_edge_graph.resolve_residue_records_for_prot",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        "science.dtie.v66.chem_edge_graph.attach_chem_edge_graph", _fake_chem
    )

    model = SimpleNamespace(
        role_edge_mp=True,
        chem_edge_mp=True,
        containment_edge_mp=True,
        hyperbolic_mp_graph=False,
        role_coupling_edges=False,
        dehydron_exclusivity=True,
        spoke_edge_scale=1.0,
        ribbon_edge_scale=1.0,
        rim_fanout_forward=False,
        thermo_edge_message_gate=False,
        multi_rel_edge_mp=False,
        geometric_angular_prior=False,
        curvature=torch.tensor(1.0),
    )
    out = prepare_training_batch(model, prot, "cpu", structural_disc_frozen=False)
    assert out.n_residue_nodes == n
    assert out.x.shape[0] == n + 1
    assert out.structural_z_disc.shape[0] == n + 1
    assert torch.allclose(out.structural_z_disc[n], torch.zeros(2))
    assert out.containment_edge_graph is True


def test_empty_sse_pads_attr_but_does_not_grow_n() -> None:
    data = _toy_chem_ready_data()
    ca = torch.randn(6, 3)
    residue_ids = [f"A:{i}:" for i in range(10, 16)]
    out = attach_containment_edge_graph(
        data, ca, "END\n", residue_ids=residue_ids, force=True
    )
    assert out.x.shape[0] == 6
    assert out.n_parent_nodes == 0
    assert out.edge_attr.size(-1) == EDGE_ATTR_CONTAIN_DIM
    assert out.containment_edge_graph is True


def test_containment_radial_mlp_count_is_nine() -> None:
    with isolated_torch_seed(123):
        m = GOSPConeMapperV66(
            node_dim=3,
            hidden=32,
            num_layers=1,
            num_experts=2,
            role_edge_mp=True,
            chem_edge_mp=True,
            containment_edge_mp=True,
            init_seed=123,
        )
    assert len(m.convs[0].radial_mlps) == 9


def test_containment_edge_mp_requires_chem_edge_mp() -> None:
    with pytest.raises(ValueError, match="containment_edge_mp requires chem_edge_mp"):
        GOSPConeMapperV66(
            node_dim=3,
            hidden=32,
            num_layers=1,
            num_experts=2,
            role_edge_mp=True,
            chem_edge_mp=False,
            containment_edge_mp=True,
        )


def test_containment_radial_mlp_expansion_preserves_gate_init_under_isolated_seed() -> None:
    """New contain MLP slots must not scramble gate/prototype under init_seed discipline.

    Hard prerequisite before any cold containment run — same as Chem-MVP.
    Required: max |Δ| = 0 on gate/prototype tensors (torch.equal).
    """

    def _gate_proto_snapshot(model: GOSPConeMapperV66) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for name, tensor in model.state_dict().items():
            if name.startswith("gate.") or "prototype" in name.lower():
                out[name] = tensor.detach().clone()
        return out

    with isolated_torch_seed(123):
        baseline = GOSPConeMapperV66(
            node_dim=4,
            hidden=32,
            num_layers=1,
            num_experts=2,
            role_edge_mp=True,
            chem_edge_mp=True,
            containment_edge_mp=False,
            init_seed=123,
        )
        base_snap = _gate_proto_snapshot(baseline)

    with isolated_torch_seed(123):
        contain = GOSPConeMapperV66(
            node_dim=4,
            hidden=32,
            num_layers=1,
            num_experts=2,
            role_edge_mp=True,
            chem_edge_mp=True,
            containment_edge_mp=True,
            init_seed=123,
        )
        contain_snap = _gate_proto_snapshot(contain)

    assert base_snap.keys() == contain_snap.keys()
    for key in base_snap:
        assert torch.equal(base_snap[key], contain_snap[key]), (
            f"{key} diverged containment-off vs on "
            f"(max |Δ|={(base_snap[key] - contain_snap[key]).abs().max().item()})"
        )
    assert len(contain.convs[0].radial_mlps) == NUM_ROLE_RELATIONS_WITH_CONTAINMENT
    assert len(baseline.convs[0].radial_mlps) == NUM_ROLE_RELATIONS_WITH_CONTAINMENT - 2


def test_empty_containment_forward_no_nan() -> None:
    n = 6
    data = _toy_chem_ready_data(n=n, feat_dim=4)
    ca = torch.randn(n, 3)
    residue_ids = [f"A:{i}:" for i in range(10, 10 + n)]
    data = attach_containment_edge_graph(
        data, ca, "END\n", residue_ids=residue_ids, force=True
    )
    assert data.n_parent_nodes == 0
    assert data.edge_attr.size(-1) == EDGE_ATTR_CONTAIN_DIM

    data.clustering = torch.rand(n)
    data.degree = torch.ones(n)
    data.ss_onehot = torch.zeros(n, 3)
    data.rho = data.x[:, 0].clone()

    model = GOSPConeMapperV66(
        node_dim=4,
        hidden=32,
        num_layers=1,
        num_experts=2,
        role_edge_mp=True,
        chem_edge_mp=True,
        containment_edge_mp=True,
    )
    model.eval()
    with torch.no_grad():
        out = model(data)

    for key in ("x_hyp", "cone_depth", "hyp_projections_2d"):
        assert torch.isfinite(out[key]).all(), key
    for ukey, utensor in out["uncertainty"].items():
        assert torch.isfinite(utensor).all(), f"uncertainty.{ukey}"


def test_load_checkpoint_restores_containment_num_relations(tmp_path: Path) -> None:
    """Save tiny containment-on checkpoint; reload must restore num_relations=9."""
    from science.training.gnn_lineage import load_model_from_checkpoint

    with isolated_torch_seed(123):
        model = GOSPConeMapperV66(
            node_dim=3,
            hidden=32,
            num_layers=1,
            num_experts=2,
            role_edge_mp=True,
            chem_edge_mp=True,
            containment_edge_mp=True,
            init_seed=123,
        )
    ckpt_path = tmp_path / "containment_tiny.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "architecture": {
                "version": "v6.6",
                "node_dim": 3,
                "hidden": 32,
                "num_experts": 2,
                "role_edge_mp": True,
                "chem_edge_mp": True,
                "containment_edge_mp": True,
            },
            "training_config": {
                "gnn_lineage": "v6.6",
                "role_edge_mp": True,
                "chem_edge_mp": True,
                "containment_edge_mp": True,
            },
        },
        ckpt_path,
    )
    loaded = load_model_from_checkpoint(ckpt_path, "cpu", lineage_id="v6.6")
    assert getattr(loaded, "containment_edge_mp", False) is True
    assert len(loaded.convs[0].radial_mlps) == NUM_ROLE_RELATIONS_WITH_CONTAINMENT
