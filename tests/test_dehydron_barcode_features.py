"""Unit tests for dehydron barcode features (Tasks 1–4)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from science.dtie.common.dehydron_barcode_features import (
    BARCODE_FEATURE_VERSION,
    BINNED_DIM,
    SCALAR_DIM,
    DehydronMidpoint,
    StructureAtom,
    extract_dehydron_midpoints,
    residue_map_key,
)


def _make_backbone_pair(
    *,
    donor_residue_index: int = 3,
    acceptor_residue_index: int = 2,
    n_coord: np.ndarray | None = None,
    o_coord: np.ndarray | None = None,
    wrapping_carbons: list[np.ndarray] | None = None,
    chain: str = "A",
) -> tuple[list[StructureAtom], dict[tuple[str, int], int]]:
    """Two-residue backbone H-bond with optional apolar wrapping carbons."""
    n_coord = np.asarray(n_coord if n_coord is not None else [5.5, 0.0, 0.0], dtype=np.float64)
    o_coord = np.asarray(o_coord if o_coord is not None else [3.2, 0.0, 0.0], dtype=np.float64)

    atoms: list[StructureAtom] = [
        StructureAtom("N", "N", n_coord, "ALA", chain, donor_residue_index),
        StructureAtom("O", "O", o_coord, "ALA", chain, acceptor_residue_index),
        StructureAtom("CA", "C", n_coord + [0.0, 1.0, 0.0], "ALA", chain, donor_residue_index),
        StructureAtom("CA", "C", o_coord + [0.0, 1.0, 0.0], "ALA", chain, acceptor_residue_index),
    ]
    for i, coord in enumerate(wrapping_carbons or []):
        atoms.append(
            StructureAtom(
                f"CB{i}",
                "C",
                np.asarray(coord, dtype=np.float64),
                "LEU",
                chain,
                99 + i,
            )
        )

    residue_index_map = {
        residue_map_key(chain, acceptor_residue_index): 0,
        residue_map_key(chain, donor_residue_index): 1,
    }
    return atoms, residue_index_map


def test_constants_defined():
    from science.dtie.common.dehydron_barcode_features import (
        LONG_LIVED_PERSISTENCE_ANGSTROM,
    )

    assert BARCODE_FEATURE_VERSION == "dehydron_barcode_v1_2"
    assert SCALAR_DIM == 3
    assert BINNED_DIM == 40
    assert LONG_LIVED_PERSISTENCE_ANGSTROM == 3.11


def test_current_version_changes_writer_and_loader_sidecar_path(tmp_path):
    from experiments.training.v6._data import _barcode_sidecar_path
    from experiments.training.v6.precompute_dehydron_barcodes import _sidecar_path

    expected = tmp_path / "4OBE_A_dehydron_barcode_v1_2.pt"
    stale = tmp_path / "4OBE_A_dehydron_barcode_v1.pt"

    assert _sidecar_path(tmp_path, "4obe", "A") == expected
    assert _barcode_sidecar_path(tmp_path, "4obe", "A") == expected
    assert expected != stale


def test_extract_midpoints_requires_inter_residue_hbond():
    atoms, residue_index_map = _make_backbone_pair(wrapping_carbons=[])
    midpoints = extract_dehydron_midpoints(atoms, residue_index_map)

    assert isinstance(midpoints, list)
    assert len(midpoints) == 1
    mp = midpoints[0]
    assert isinstance(mp, DehydronMidpoint)
    assert mp.donor_idx != mp.acceptor_idx
    assert mp.donor_idx == 1
    assert mp.acceptor_idx == 0
    assert mp.wrapping_count < 13.0
    expected_mid = (np.array([5.5, 0.0, 0.0]) + np.array([3.2, 0.0, 0.0])) / 2.0
    np.testing.assert_allclose(mp.coord, expected_mid)


def test_extract_midpoints_empty_when_overwrapped():
    midpoint = (np.array([5.5, 0.0, 0.0]) + np.array([3.2, 0.0, 0.0])) / 2.0
    # Pack 13+ apolar side-chain carbons within 6.5 Å of the H-bond midpoint.
    wrapping = [midpoint + np.array([1.0, 0.0, 0.0]) * (k % 3) + np.array([0.0, (k % 4) * 0.5, 0.0]) for k in range(14)]
    atoms, residue_index_map = _make_backbone_pair(wrapping_carbons=wrapping)

    midpoints = extract_dehydron_midpoints(atoms, residue_index_map)
    assert midpoints == []


def test_extract_midpoints_skips_same_residue_n_o():
    coord = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    atoms = [
        StructureAtom("N", "N", coord, "ALA", "A", 1),
        StructureAtom("O", "O", coord + [1.0, 0.0, 0.0], "ALA", "A", 1),
    ]
    residue_index_map = {residue_map_key("A", 1): 0}

    midpoints = extract_dehydron_midpoints(atoms, residue_index_map)
    assert midpoints == []


@pytest.fixture
def simple_midpoints() -> list[DehydronMidpoint]:
    """Five spread 3D midpoints — enough for k-means landmarks and witness complex."""
    coords = [
        np.array([0.0, 0.0, 0.0]),
        np.array([5.0, 0.0, 0.0]),
        np.array([2.5, 4.0, 0.0]),
        np.array([2.5, 1.5, 3.0]),
        np.array([8.0, 2.0, 1.0]),
    ]
    return [
        DehydronMidpoint(
            coord=c,
            donor_idx=i,
            acceptor_idx=(i + 1) % len(coords),
            wrapping_count=5.0,
        )
        for i, c in enumerate(coords)
    ]


@pytest.fixture
def simple_bars(simple_midpoints):
    from science.dtie.common.dehydron_barcode_features import compute_witness_persistence

    return compute_witness_persistence(simple_midpoints, max_alpha_angstrom=20.0)


def test_witness_persistence_empty_midpoints():
    from science.dtie.common.dehydron_barcode_features import compute_witness_persistence

    assert compute_witness_persistence([]) == []


def test_witness_persistence_single_midpoint_returns_empty():
    from science.dtie.common.dehydron_barcode_features import compute_witness_persistence

    one = DehydronMidpoint(
        coord=np.array([0.0, 0.0, 0.0]),
        donor_idx=0,
        acceptor_idx=1,
        wrapping_count=5.0,
    )
    assert compute_witness_persistence([one]) == []


def test_witness_persistence_returns_nonneg_persistence(simple_midpoints):
    from science.dtie.common.dehydron_barcode_features import compute_witness_persistence

    bars = compute_witness_persistence(simple_midpoints, max_alpha_angstrom=20.0)
    assert len(bars) > 0
    assert all(b.persistence >= 0.0 for b in bars)
    assert all(b.dim in (0, 1) for b in bars)


def test_aggregate_missing_mask_when_no_midpoints():
    from science.dtie.common.dehydron_barcode_features import aggregate_residue_barcode_features

    out = aggregate_residue_barcode_features(5, [], [])
    assert out["scalars"].shape == (5, 3)
    assert out["binned"] is None
    assert out["missing"].shape == (5, 1)
    assert np.allclose(out["missing"], 1.0)


def test_aggregate_binned_shape_when_enabled(simple_midpoints, simple_bars):
    from science.dtie.common.dehydron_barcode_features import aggregate_residue_barcode_features

    out = aggregate_residue_barcode_features(10, simple_midpoints, simple_bars, use_binned=True)
    assert out["binned"].shape == (10, 40)


def test_aggregate_structure_stats_not_kfold_duplicated():
    """Residue touching K midpoints must not K-fold duplicate structure-level bar stats."""
    from science.dtie.common.dehydron_barcode_features import (
        PersistenceBar,
        aggregate_residue_barcode_features,
    )

    midpoints = [
        DehydronMidpoint(
            coord=np.array([0.0, 0.0, 0.0]),
            donor_idx=0,
            acceptor_idx=1,
            wrapping_count=5.0,
        ),
        DehydronMidpoint(
            coord=np.array([5.0, 0.0, 0.0]),
            donor_idx=0,
            acceptor_idx=2,
            wrapping_count=5.0,
        ),
    ]
    bars = [
        PersistenceBar(dim=0, birth=0.0, death=20.0, persistence=20.0),
        PersistenceBar(dim=1, birth=0.0, death=2.0, persistence=2.0),
        PersistenceBar(dim=1, birth=0.0, death=3.0, persistence=3.0),
    ]
    out = aggregate_residue_barcode_features(5, midpoints, bars)
    scalars = out["scalars"]

    total_h1 = 5.0
    assert scalars[0, 0] == pytest.approx(np.log1p(total_h1))
    assert scalars[0, 0] != pytest.approx(np.log1p(2 * total_h1))
    # local count stays residue-local (2 midpoints touch residue 0)
    assert scalars[0, 2] == pytest.approx(np.log1p(2))


def test_edge_barcode_features_are_pair_local(simple_midpoints, simple_bars):
    from science.dtie.common.dehydron_barcode_features import (
        EDGE_BARCODE_DIM,
        aggregate_dehydron_edge_barcode_features,
    )

    out = aggregate_dehydron_edge_barcode_features(simple_midpoints, simple_bars)
    assert out["edge_pairs"].shape[1] == 2
    assert out["edge_scalars"].shape == (len(simple_midpoints), EDGE_BARCODE_DIM)
    assert out["edge_scalars"][0, 0] >= 0.0
    assert out["edge_scalars"][0, 1] >= 0.0


def test_align_edge_barcode_to_graph_resseq():
    from science.dtie.common.dehydron_barcode_features import (
        EDGE_BARCODE_DIM,
        align_edge_barcode_to_graph,
    )

    pairs = np.array([[0, 1], [2, 3]], dtype=np.int32)
    scalars = np.array(
        [
            [1.0, 0.1, 0.2, 0.3, 0.4],
            [0.5, 0.1, 0.2, 0.3, 0.4],
        ],
        dtype=np.float32,
    )
    barcode_resseq = [10, 11, 12, 13]
    graph_resseq = [10, 11, 12, 13]
    lookup = align_edge_barcode_to_graph(pairs, scalars, barcode_resseq, graph_resseq)
    assert (0, 1) in lookup
    assert lookup[(0, 1)].shape == (EDGE_BARCODE_DIM,)


def test_stack_dims_scalars_only():
    from science.dtie.common.dehydron_barcode_features import stack_node_features_with_barcode

    x = np.zeros((8, 3), np.float32)
    barcode = {
        "scalars": np.zeros((8, 3), np.float32),
        "binned": None,
        "missing": np.ones((8, 1), np.float32),
    }
    y = stack_node_features_with_barcode(x, barcode, use_binned=False)
    assert y.shape == (8, 7)


def test_stack_dims_full():
    from science.dtie.common.dehydron_barcode_features import stack_node_features_with_barcode

    x = np.zeros((8, 3), np.float32)
    barcode = {
        "scalars": np.zeros((8, 3), np.float32),
        "binned": np.zeros((8, 40), np.float32),
        "missing": np.zeros((8, 1), np.float32),
    }
    y = stack_node_features_with_barcode(x, barcode, use_binned=True)
    assert y.shape == (8, 47)


def test_feature_set_ids():
    from science.dtie.common.residue_features import (
        GnnInputMode,
        gnn_feature_set_id,
        gnn_feature_set_id_for_barcode,
        gnn_input_dim_for_barcode,
    )

    assert (
        gnn_feature_set_id_for_barcode(
            use_barcode=False,
            use_binned=False,
            mode=GnnInputMode.TOPOLOGY_THREE_VECTOR,
        )
        == gnn_feature_set_id(GnnInputMode.TOPOLOGY_THREE_VECTOR)
    )
    assert "dbh_scalars_v1_2" in gnn_feature_set_id_for_barcode(
        use_barcode=True,
        use_binned=False,
        mode=GnnInputMode.TOPOLOGY_THREE_VECTOR,
    )
    assert "dbh_full_v1_2" in gnn_feature_set_id_for_barcode(
        use_barcode=True,
        use_binned=True,
        mode=GnnInputMode.TOPOLOGY_THREE_VECTOR,
    )
    assert (
        gnn_input_dim_for_barcode(
            use_barcode=True,
            use_binned=False,
            mode=GnnInputMode.TOPOLOGY_THREE_VECTOR,
        )
        == 7
    )
    assert (
        gnn_input_dim_for_barcode(
            use_barcode=True,
            use_binned=True,
            mode=GnnInputMode.TOPOLOGY_THREE_VECTOR,
        )
        == 47
    )


def test_attach_missing_sidecar_zeros_shape(tmp_path):
    import torch
    from torch_geometric.data import Data

    from experiments.training.v6._data import attach_dehydron_barcode_features

    prot = {
        "pdb_id": "4OBE",
        "chain": "A",
        "residue_ids": ["A:1:", "A:2:", "A:3:", "A:4:"],
        "data": Data(x=torch.ones(4, 3)),
    }

    attach_dehydron_barcode_features(prot, barcode_dir=tmp_path, use_binned=False)

    x = prot["data"].x
    assert x.shape == (4, 7)
    assert torch.allclose(x[:, :3], torch.ones(4, 3))
    assert torch.allclose(x[:, 3:6], torch.zeros(4, 3))
    assert torch.allclose(x[:, 14:], torch.ones(4, 1))


def test_precompute_sidecar_round_trips_through_attach(tmp_path):
    import torch
    from torch_geometric.data import Data

    from experiments.training.v6._data import attach_dehydron_barcode_features
    from experiments.training.v6.precompute_dehydron_barcodes import _save_payload
    from science.dtie.common.dehydron_barcode_features import barcode_sidecar_filename

    sidecar = tmp_path / barcode_sidecar_filename("4OBE", "A")
    _save_payload(
        sidecar,
        {
            "scalars": np.full((4, 3), 2.0, dtype=np.float32),
            "binned": np.full((4, 40), 3.0, dtype=np.float32),
            "missing": np.zeros((4, 1), dtype=np.float32),
            "residue_indices": np.asarray([10, 11, 12, 13], dtype=np.int32),
            "metadata": {"version": BARCODE_FEATURE_VERSION},
        },
    )

    weights_only_payload = torch.load(sidecar, map_location="cpu", weights_only=True)
    assert isinstance(weights_only_payload["scalars"], torch.Tensor)
    assert weights_only_payload["scalars"].dtype == torch.float32

    prot = {
        "pdb_id": "4OBE",
        "chain": "A",
        "residue_ids": ["A:10:", "A:11:", "A:12:", "A:13:"],
        "data": Data(x=torch.ones(4, 3)),
    }
    attach_dehydron_barcode_features(prot, barcode_dir=tmp_path, use_binned=True)

    x = prot["data"].x
    assert x.shape == (4, 47)
    assert torch.allclose(x[:, :3], torch.ones(4, 3))
    assert torch.allclose(x[:, 3:6], torch.full((4, 3), 2.0))
    assert torch.allclose(x[:, 6:46], torch.full((4, 40), 3.0))
    assert torch.allclose(x[:, 46:], torch.zeros(4, 1))


def test_align_barcode_to_longer_graph_marks_extra_missing():
    from science.dtie.common.dehydron_barcode_features import align_barcode_to_graph_residues

    barcode = {
        "scalars": np.full((2, 3), 2.0, dtype=np.float32),
        "missing": np.zeros((2, 1), dtype=np.float32),
        "binned": np.full((2, 40), 3.0, dtype=np.float32),
        "residue_indices": np.asarray([10, 12], dtype=np.int32),
    }
    aligned = align_barcode_to_graph_residues(
        barcode,
        [10, 11, 12],
        use_binned=True,
    )
    assert aligned["scalars"].shape == (3, 3)
    assert np.allclose(aligned["scalars"][0], 2.0)
    assert np.allclose(aligned["scalars"][1], 0.0)
    assert np.allclose(aligned["scalars"][2], 2.0)
    assert aligned["missing"][0, 0] == 0.0
    assert aligned["missing"][1, 0] == 1.0
    assert aligned["missing"][2, 0] == 0.0


def test_attach_sidecar_prefers_weights_only_load(tmp_path, monkeypatch):
    import torch
    from torch_geometric.data import Data

    from experiments.training.v6 import _data
    from science.dtie.common.dehydron_barcode_features import barcode_sidecar_filename

    sidecar = tmp_path / barcode_sidecar_filename("4OBE", "A")
    sidecar.write_bytes(b"placeholder")
    load_calls: list[bool | None] = []

    def fake_load(path, *, map_location=None, weights_only=None):
        assert path == sidecar
        assert map_location == "cpu"
        load_calls.append(weights_only)
        return {
            "scalars": torch.full((4, 3), 2.0),
            "binned": None,
            "missing": torch.zeros((4, 1)),
            "residue_indices": torch.tensor([1, 2, 3, 4], dtype=torch.int32),
        }

    monkeypatch.setattr(_data.torch, "load", fake_load)

    prot = {
        "pdb_id": "4OBE",
        "chain": "A",
        "residue_ids": ["A:1:", "A:2:", "A:3:", "A:4:"],
        "data": Data(x=torch.ones(4, 3)),
    }
    _data.attach_dehydron_barcode_features(prot, barcode_dir=tmp_path, use_binned=False)

    assert load_calls == [True]
    assert prot["data"].x.shape == (4, 7)


FOUROBE_PDB = Path(
    "/home/rswan/Documents/tokyo-eyes-consolidation/tokyo-eye-agenticpoincare/"
    "checkpoints/v6/runs/slim_moe_structural_ssot_cold_v1/viewers/4obe/4obe_gosp_native.pdb"
)


@pytest.mark.skipif(not FOUROBE_PDB.is_file(), reason="4obe PDB not available")
def test_featurize_4obe_chain_a_smoke():
    from science.dtie.common.dehydron_barcode_features import (
        BARCODE_FEATURE_VERSION,
        featurize_chain_dehydron_barcode,
    )

    out = featurize_chain_dehydron_barcode(FOUROBE_PDB, "A")
    assert out["metadata"]["version"] == BARCODE_FEATURE_VERSION
    assert out["metadata"]["n_midpoints"] >= 1
    assert out["metadata"]["n_bars"] >= 0
    n = out["scalars"].shape[0]
    assert out["scalars"].shape == (n, 3)
    assert out["missing"].shape == (n, 1)
    assert out["edge_pairs"].shape[0] == out["metadata"]["n_midpoints"]
    assert out["edge_scalars"].shape[0] == out["metadata"]["n_midpoints"]
    assert np.all(np.isfinite(out["scalars"]))
    assert np.all(np.isfinite(out["missing"]))
