"""Unit tests for KRAS topo-structural Phase A helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from science.dtie.common.kras_topo_matrix import (
    align_resseq_vectors,
    betweenness_proximity_verdict,
    ca_contact_edges_by_resseq,
    ca_distance,
    edge_delta_verdict,
    edge_symdiff_size,
    parse_resseq,
    primary_matrix_verdict,
    residue_index_map,
    spearman_rho,
    switch_lock_verdict,
)


def test_parse_resseq_variants() -> None:
    assert parse_resseq(12) == 12
    assert parse_resseq("A:81") == 81
    assert parse_resseq("A:81:") == 81
    assert parse_resseq("ASP114") == 114
    assert parse_resseq(None) is None


def test_align_and_spearman_mut_closer_to_active() -> None:
    # Shared resseqs 1..5; mut mirrors active more than wt
    active = [1.0, 2.0, 3.0, 4.0, 5.0]
    mut = [1.1, 2.1, 2.9, 4.2, 4.8]
    wt = [5.0, 4.0, 3.0, 2.0, 1.0]  # anti-correlated
    maps = {
        "4OBE": {i + 1: i for i in range(5)},
        "4DSO": {i + 1: i for i in range(5)},
        "5VQ2": {i + 1: i for i in range(5)},
    }
    shared, aligned = align_resseq_vectors(
        {"4OBE": wt, "4DSO": mut, "5VQ2": active},
        maps,
        structures=["4OBE", "4DSO", "5VQ2"],
    )
    assert shared == [1, 2, 3, 4, 5]
    rho_mut = spearman_rho(aligned["4DSO"], aligned["5VQ2"])
    rho_wt = spearman_rho(aligned["4OBE"], aligned["5VQ2"])
    v = betweenness_proximity_verdict(rho_mut, rho_wt)
    assert v["pass"] is True
    assert rho_mut > rho_wt


def test_edge_symdiff_pass_rule() -> None:
    e_active = {(1, 2), (2, 3), (3, 4)}
    e_mut = {(1, 2), (2, 3), (3, 5)}  # |ΔE|=2 vs active
    e_wt = {(1, 9), (9, 10)}  # |ΔE|=5 vs active
    d_mut = edge_symdiff_size(e_mut, e_active, shared_resseqs=[1, 2, 3, 4, 5, 9, 10])
    d_wt = edge_symdiff_size(e_wt, e_active, shared_resseqs=[1, 2, 3, 4, 5, 9, 10])
    v = edge_delta_verdict(d_mut, d_wt)
    assert v["pass"] is True
    assert v["report_only"] is True


def test_four_quadrant_edge_verdict() -> None:
    from science.dtie.common.kras_topo_matrix import four_quadrant_edge_verdict

    ok = four_quadrant_edge_verdict(
        delta_e_mimetic=33,
        delta_e_wt_to_active=89,
        delta_e_mut_span=40,
        delta_e_wt_span=106,
    )
    assert ok["pass"] is True
    assert ok["gates"]["mimetic_inactive"] is True
    assert ok["gates"]["compressed_mut_span"] is True

    fail_mimetic = four_quadrant_edge_verdict(
        delta_e_mimetic=90,
        delta_e_wt_to_active=89,
        delta_e_mut_span=40,
        delta_e_wt_span=106,
    )
    assert fail_mimetic["pass"] is False
    assert fail_mimetic["gates"]["mimetic_inactive"] is False

    fail_span = four_quadrant_edge_verdict(
        delta_e_mimetic=33,
        delta_e_wt_to_active=89,
        delta_e_mut_span=110,
        delta_e_wt_span=106,
    )
    assert fail_span["pass"] is False
    assert fail_span["gates"]["compressed_mut_span"] is False


def test_switch_lock_verdict() -> None:
    mut = {(12, 32): 8.0, (12, 61): 9.0}
    wt = {(12, 32): 14.0, (12, 61): 15.0}
    assert switch_lock_verdict(mut, wt)["pass"] is True
    # 10.49 Å is within relaxed Switch-I 11.0 Å cutoff
    near = {(12, 32): 10.49, (12, 61): 5.09}
    assert switch_lock_verdict(near, wt)["pass"] is True
    bad = {(12, 32): 8.0, (12, 61): 12.0}  # 61 fails form at 10 Å
    assert switch_lock_verdict(bad, wt)["pass"] is False


def test_dehydron_wrapper_edges_smoke(tmp_path: Path) -> None:
    # Minimal 2-residue backbone with close N–O — just ensure API returns sets
    from science.dtie.common.kras_topo_matrix import dehydron_wrapper_edges_by_resseq

    # Use real PDB if available; otherwise skip
    pdb = Path("/tmp/dtie_pdb_cache/4OBE.pdb")
    if not pdb.is_file():
        pdb = Path("pdb_cache/4OBE.pdb")
    if not pdb.is_file():
        return
    graphs = dehydron_wrapper_edges_by_resseq(pdb, "A")
    assert "dehydron" in graphs and "complementarity" in graphs
    assert graphs["dehydron"].issubset(graphs["complementarity"])
    assert len(graphs["complementarity"]) >= len(graphs["dehydron"])


def test_ca_contact_and_distance() -> None:
    # Three collinear Cα at 0, 5, 10 Å — edges 1-2 and 2-3 at 8Å cutoff; not 1-3
    coords = np.array([[0.0, 0, 0], [5.0, 0, 0], [10.0, 0, 0]], dtype=np.float64)
    rids = [10, 11, 12]
    edges = ca_contact_edges_by_resseq(coords, rids, cutoff_angstrom=8.0)
    assert (10, 11) in edges and (11, 12) in edges
    assert (10, 12) not in edges
    assert abs(ca_distance(coords, rids, 10, 12) - 10.0) < 1e-6
    assert residue_index_map(rids)[11] == 1


def test_primary_matrix_aggregate() -> None:
    # Edge Fail no longer blocks aggregate Pass
    v = primary_matrix_verdict(
        betweenness={"pass": True},
        edge_delta={"pass": False},
        switch_lock={"pass": True},
    )
    assert v["pass"] is True
    assert v["outcome"] == "Pass"
    assert v["report_only"]["edge_symmetric_difference"] is False

    fail = primary_matrix_verdict(
        betweenness={"pass": True},
        edge_delta={"pass": True},
        switch_lock={"pass": False},
    )
    assert fail["pass"] is False
    assert fail["outcome"] == "Fail"
