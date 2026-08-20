"""Unit tests for KRAS G12 residue-12 graft helpers."""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch_geometric.data import Data

from science.dtie.common.kras_g12_graft import (
    graft_direction_verdict,
    graft_resseq_from_donor,
    rebuild_ca_graph,
)


def _toy_prot(pdb_id: str, ca: np.ndarray, x_row0: float) -> dict:
    n = ca.shape[0]
    x = torch.zeros(n, 3)
    x[:, 0] = x_row0
    for i in range(n):
        x[i, 1] = float(i)
    data = Data(
        x=x,
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_attr=torch.zeros(2, 4),
    )
    return {
        "pdb_id": pdb_id,
        "chain": "A",
        "data": data,
        "ca_coords": torch.tensor(ca, dtype=torch.float32),
        "residue_ids": [f"A:{10 + i}:" for i in range(n)],
    }


def test_graft_copies_features_and_ca_at_target() -> None:
    # resseqs 10,11,12,13 — target index 2 is resseq 12
    ca_wt = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.float64)
    ca_mut = np.array([[0, 0, 0], [1, 0, 0], [9, 9, 9], [3, 0, 0]], dtype=np.float64)
    wt = _toy_prot("WT", ca_wt, 0.1)
    mut = _toy_prot("MUT", ca_mut, 0.9)
    mut["data"].x[2] = torch.tensor([0.5, 0.6, 0.7])

    grafted = graft_resseq_from_donor(wt, mut, target_resseq=12, donor_resseq=12)
    assert torch.allclose(grafted["data"].x[2], torch.tensor([0.5, 0.6, 0.7]))
    ca = grafted["ca_coords"].numpy()
    assert np.allclose(ca[2], [9, 9, 9])
    # Other sites unchanged
    assert torch.allclose(grafted["data"].x[0], wt["data"].x[0])


def test_scramble_uses_distal_donor() -> None:
    ca_wt = np.zeros((4, 3), dtype=np.float64)
    ca_mut = np.zeros((4, 3), dtype=np.float64)
    ca_mut[0] = [7, 7, 7]  # resseq 10
    wt = _toy_prot("WT", ca_wt, 0.0)
    mut = _toy_prot("MUT", ca_mut, 0.0)
    mut["data"].x[0] = torch.tensor([1.0, 2.0, 3.0])

    # Donor resseq 10 onto target 12
    g = graft_resseq_from_donor(wt, mut, target_resseq=12, donor_resseq=10)
    assert torch.allclose(g["data"].x[2], torch.tensor([1.0, 2.0, 3.0]))
    assert np.allclose(g["ca_coords"].numpy()[2], [7, 7, 7])
    assert g["graft_meta"]["donor_resseq"] == 10


def test_graft_direction_verdict() -> None:
    ok = graft_direction_verdict(rho_wt_mut=0.2, rho_graft_mut=0.5, rho_scramble_mut=0.3)
    assert ok["pass"] is True
    fail = graft_direction_verdict(rho_wt_mut=0.5, rho_graft_mut=0.4, rho_scramble_mut=0.1)
    assert fail["pass"] is False
    fail2 = graft_direction_verdict(rho_wt_mut=0.2, rho_graft_mut=0.5, rho_scramble_mut=0.6)
    assert fail2["pass"] is False


def test_neighborhood_n12_and_scramble_donors() -> None:
    from science.dtie.common.kras_g12_graft import (
        distal_scramble_donors,
        graft_neighborhood_matched,
        neighborhood_n12,
        neighborhood_verdict,
    )

    # Linear chain: resseqs 10..20 at 0,3,6,... Å — neighbors of 12 within 10Å
    n = 11
    ca = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        ca[i, 0] = float(i * 3.0)
    wt = _toy_prot("WT", ca, 0.0)
    mut = _toy_prot("MUT", ca.copy(), 0.0)
    # Remap residue_ids to 10..20
    wt["residue_ids"] = [f"A:{10 + i}:" for i in range(n)]
    mut["residue_ids"] = [f"A:{10 + i}:" for i in range(n)]
    # Put distinctive features on mut 12 and neighbors
    mut["data"].x[2] = torch.tensor([0.9, 0.8, 0.7])  # resseq 12

    n12 = neighborhood_n12(mut, wt, seed_resseq=12, include_switch_lock=False)
    assert 12 in n12
    # 3Å spacing → within 10Å: 12±1,±2,±3
    assert set(n12) >= {12}

    donors = distal_scramble_donors(mut, n_needed=len(n12))
    assert len(donors) == len(n12)
    assert 12 not in donors

    grafted = graft_neighborhood_matched(wt, mut, [12])
    assert torch.allclose(grafted["data"].x[2], torch.tensor([0.9, 0.8, 0.7]))

    v_ok = neighborhood_verdict(
        rho_wt_mut=0.5,
        rho_single_mut=0.51,
        rho_neighborhood_mut=0.60,
        rho_scramble_mut=0.52,
    )
    assert v_ok["pass"] is True
    v_flat = neighborhood_verdict(
        rho_wt_mut=0.5,
        rho_single_mut=0.55,
        rho_neighborhood_mut=0.52,
        rho_scramble_mut=0.40,
    )
    assert v_flat["pass"] is False


def test_rebuild_preserves_x() -> None:
    ca = np.array([[0, 0, 0], [5, 0, 0], [10, 0, 0]], dtype=np.float64)
    prot = _toy_prot("X", ca, 0.3)
    out = rebuild_ca_graph(prot, ca)
    assert out["data"].x.shape == prot["data"].x.shape
    assert out["data"].edge_index.shape[0] == 2


def test_splice_x_hyp_and_hyp_latent_verdict() -> None:
    from science.dtie.common.kras_g12_graft import (
        hyp_latent_neighborhood_matched,
        hyp_latent_verdict,
        splice_x_hyp,
    )

    # Toy ball points near origin (safe for project/dist0).
    x_wt = torch.zeros(4, 8)
    x_mut = torch.zeros(4, 8)
    x_mut[2, 0] = 0.2  # resseq 12 → deeper
    map_ids = {10: 0, 11: 1, 12: 2, 13: 3}
    out, depth, pairs = splice_x_hyp(
        x_wt,
        map_ids,
        x_mut,
        map_ids,
        target_to_donor={12: 12},
        curvature=1.0,
    )
    assert len(pairs) == 1
    assert torch.allclose(out[2], x_mut[2])
    assert float(depth[2]) > float(depth[0])

    _, depth_n, meta = hyp_latent_neighborhood_matched(
        x_wt, map_ids, x_mut, map_ids, [12], curvature=1.0
    )
    assert meta["kind"] == "hyp_neighborhood_matched"
    assert float(depth_n[2]) == float(depth[2])

    v_ok = hyp_latent_verdict(
        rho_wt_mut=0.50,
        rho_hyp_graft_mut=0.60,
        rho_euc_neigh_mut=0.55,
        rho_hyp_scramble_mut=0.52,
    )
    assert v_ok["pass"] is True
    v_fail_euc = hyp_latent_verdict(
        rho_wt_mut=0.50,
        rho_hyp_graft_mut=0.54,
        rho_euc_neigh_mut=0.55,
        rho_hyp_scramble_mut=0.40,
    )
    assert v_fail_euc["pass"] is False
    v_fail_scr = hyp_latent_verdict(
        rho_wt_mut=0.50,
        rho_hyp_graft_mut=0.60,
        rho_euc_neigh_mut=0.55,
        rho_hyp_scramble_mut=0.61,
    )
    assert v_fail_scr["pass"] is False
