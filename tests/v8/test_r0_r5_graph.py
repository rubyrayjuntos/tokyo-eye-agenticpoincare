"""Sprint 2: R0–R5 dual-layer biophysical graph contract (v8 frozen)."""

from __future__ import annotations

import numpy as np
import pytest

from science.tokyo_eye.v8.types import AtomRecord, ResidueRecord
from science.tokyo_eye.v8.r0_r5_graph import (
    DEHYDRON_WRAP_MAX,
    R0_COVALENT,
    R1_HBOND,
    R2_DEHYDRON,
    R3_HYDROPHOBIC_PI,
    R4_SALT_BRIDGE,
    R5_LOCAL_NEIGHBORHOOD,
    WRAPPING_RADIUS_A,
    build_r0_r5_graph,
    classify_hbond_vs_dehydron,
    directed_pairs_for_type,
)


def _atom(
    name: str,
    xyz: tuple[float, float, float],
    res: str,
    *,
    element: str | None = None,
) -> AtomRecord:
    el = element if element is not None else (
        name[0] if name[0].isalpha() else "C"
    )
    return AtomRecord(
        atom_name=name,
        element=el,
        coord=np.array(xyz, dtype=np.float64),
        parent_residue_name=res,
    )


def _ala(idx: int, origin: tuple[float, float, float]) -> ResidueRecord:
    ox, oy, oz = origin
    return ResidueRecord(
        chain_label="A",
        residue_index=idx,
        residue_name="ALA",
        atoms=(
            _atom("N", (ox, oy, oz), "ALA", element="N"),
            _atom("CA", (ox + 1.5, oy, oz), "ALA", element="C"),
            _atom("C", (ox + 2.5, oy, oz), "ALA", element="C"),
            _atom("O", (ox + 2.5, oy + 1.2, oz), "ALA", element="O"),
            _atom("CB", (ox + 1.5, oy + 1.5, oz), "ALA", element="C"),
        ),
    )


def test_constants_match_frozen_contract() -> None:
    assert DEHYDRON_WRAP_MAX == 19
    assert WRAPPING_RADIUS_A == pytest.approx(6.5)
    assert R0_COVALENT == 0
    assert R5_LOCAL_NEIGHBORHOOD == 5


def test_classify_wrap_gate_leq_19_is_dehydron() -> None:
    assert classify_hbond_vs_dehydron(19) == R2_DEHYDRON
    assert classify_hbond_vs_dehydron(0) == R2_DEHYDRON
    assert classify_hbond_vs_dehydron(20) == R1_HBOND


def test_dual_layer_r0_and_r1_coexist_for_adjacent_hbond() -> None:
    """Δseq=±1 + DSSP-gated chemistry must emit both R0 and Layer-B rows."""
    prev = ResidueRecord(
        chain_label="A",
        residue_index=9,
        residue_name="ALA",
        atoms=(
            _atom("N", (-3.0, 0.0, 0.0), "ALA", element="N"),
            _atom("CA", (-1.5, 0.0, 0.0), "ALA", element="C"),
            _atom("C", (-0.5, 1.0, 0.0), "ALA", element="C"),
            _atom("O", (-0.5, 2.2, 0.0), "ALA", element="O"),
        ),
    )
    r10 = ResidueRecord(
        chain_label="A",
        residue_index=10,
        residue_name="ALA",
        atoms=(
            _atom("N", (0.0, 0.0, 0.0), "ALA", element="N"),
            _atom("H", (0.0, 0.0, 1.01), "ALA", element="H"),
            _atom("CA", (1.45, 0.0, 0.0), "ALA", element="C"),
            _atom("C", (2.2, 1.0, 0.0), "ALA", element="C"),
            _atom("O", (2.2, 2.2, 0.0), "ALA", element="O"),
            _atom("CB", (1.45, -1.5, 0.0), "ALA", element="C"),
        ),
    )
    r11 = ResidueRecord(
        chain_label="A",
        residue_index=11,
        residue_name="ALA",
        atoms=(
            _atom("N", (3.0, 1.0, 2.0), "ALA", element="N"),
            _atom("CA", (2.5, 0.5, 1.5), "ALA", element="C"),
            _atom("C", (1.2, 0.0, 2.85), "ALA", element="C"),
            _atom("O", (0.0, 0.0, 2.85), "ALA", element="O"),
            _atom("CB", (3.5, 0.5, 1.5), "ALA", element="C"),
        ),
    )
    result = build_r0_r5_graph([prev, r10, r11])
    # Indices: prev=0, r10=1, r11=2 — Layer-B H-bond is between 1 and 2
    ei = result.edge_index
    et = result.edge_type
    assert ei.shape[0] == 2
    assert ei.shape[1] == et.shape[0]

    r0 = directed_pairs_for_type(ei, et, R0_COVALENT)
    layer_b = set(directed_pairs_for_type(ei, et, R1_HBOND)) | set(
        directed_pairs_for_type(ei, et, R2_DEHYDRON)
    )
    assert (1, 2) in r0 and (2, 1) in r0
    assert (1, 2) in layer_b and (2, 1) in layer_b
    types_12 = sorted(
        int(et[k])
        for k in range(et.shape[0])
        if int(ei[0, k]) == 1 and int(ei[1, k]) == 2
    )
    assert R0_COVALENT in types_12
    assert R1_HBOND in types_12 or R2_DEHYDRON in types_12


def test_layer_b_priority_pi_beats_salt_no_duplicate_undirected() -> None:
    """Within Layer B, lowest ID wins; no duplicate undirected chemistry rows."""
    from science.tokyo_eye.v8.r0_r5_graph import resolve_layer_b_primary

    primary, flags = resolve_layer_b_primary(
        has_hbond=False,
        is_dehydron=False,
        has_pi_or_hydrophobic=True,
        has_salt=True,
        has_r5_neighborhood=True,
    )
    assert primary == R3_HYDROPHOBIC_PI
    assert flags & (1 << R4_SALT_BRIDGE) != 0
    assert flags & (1 << R5_LOCAL_NEIGHBORHOOD) != 0


def test_r5_suppressed_when_chemistry_claims_pair() -> None:
    from science.tokyo_eye.v8.r0_r5_graph import resolve_layer_b_primary

    primary, _ = resolve_layer_b_primary(
        has_hbond=True,
        is_dehydron=True,
        has_pi_or_hydrophobic=False,
        has_salt=False,
        has_r5_neighborhood=True,
    )
    assert primary == R2_DEHYDRON


def test_bidirectional_invariance() -> None:
    records = [_ala(i, (float(i) * 3.8, 0.0, 0.0)) for i in range(1, 5)]
    result = build_r0_r5_graph(records)
    ei, et = result.edge_index, result.edge_type
    # Every (i→j, R) has matching (j→i, R).
    seen: set[tuple[int, int, int]] = set()
    for k in range(et.shape[0]):
        seen.add((int(ei[0, k]), int(ei[1, k]), int(et[k])))
    for i, j, r in list(seen):
        assert (j, i, r) in seen


def test_r0_backbone_for_sequence_neighbors() -> None:
    records = [_ala(i, (float(i) * 3.8, 0.0, 0.0)) for i in (10, 11, 12)]
    result = build_r0_r5_graph(records)
    r0 = directed_pairs_for_type(result.edge_index, result.edge_type, R0_COVALENT)
    assert (0, 1) in r0 and (1, 0) in r0
    assert (1, 2) in r0 and (2, 1) in r0
    assert (0, 2) not in r0


def test_wrap_count_uses_65A_double_cone() -> None:
    """Few on-axis wraps → R2; many on-axis wraps → R1 (DSSP-gated pair)."""
    donor = ResidueRecord(
        chain_label="A",
        residue_index=10,
        residue_name="ALA",
        atoms=(
            _atom("N", (0.0, 0.0, 0.0), "ALA", element="N"),
            _atom("H", (0.0, 0.0, 1.01), "ALA", element="H"),
            _atom("CA", (1.45, 0.0, 0.0), "ALA", element="C"),
            _atom("C", (2.2, 1.0, 0.0), "ALA", element="C"),
            _atom("O", (2.2, 2.2, 0.0), "ALA", element="O"),
            _atom("CB", (1.45, -1.5, 0.0), "ALA", element="C"),
        ),
    )
    acceptor = ResidueRecord(
        chain_label="A",
        residue_index=14,
        residue_name="ALA",
        atoms=(
            _atom("N", (3.0, 1.0, 2.0), "ALA", element="N"),
            _atom("CA", (2.5, 0.5, 1.5), "ALA", element="C"),
            _atom("C", (1.2, 0.0, 2.85), "ALA", element="C"),
            _atom("O", (0.0, 0.0, 2.85), "ALA", element="O"),
            _atom("CB", (3.5, 0.5, 1.5), "ALA", element="C"),
        ),
    )

    result = build_r0_r5_graph([donor, acceptor])
    r2 = directed_pairs_for_type(result.edge_index, result.edge_type, R2_DEHYDRON)
    assert (0, 1) in r2 and (1, 0) in r2

    # Midpoint of H–O ≈ (0, 0, 1.93); axis +z. Place 25 carbons on-axis inside 6.5Å.
    wrap_atoms = tuple(
        _atom(f"X{k}", (0.0, 0.0, 1.93 + 0.15 * (k - 12)), "ALA", element="C")
        for k in range(25)
    )
    crowded = ResidueRecord(
        chain_label="A",
        residue_index=99,
        residue_name="ALA",
        atoms=(_atom("CA", (20.0, 0.0, 0.0), "ALA", element="C"),) + wrap_atoms,
    )
    result1 = build_r0_r5_graph([donor, acceptor, crowded])
    r1 = directed_pairs_for_type(result1.edge_index, result1.edge_type, R1_HBOND)
    r2_bad = directed_pairs_for_type(
        result1.edge_index, result1.edge_type, R2_DEHYDRON
    )
    assert (0, 1) in r1
    assert (0, 1) not in r2_bad
