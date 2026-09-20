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
    assert DEHYDRON_WRAP_MAX == 1
    assert WRAPPING_RADIUS_A == pytest.approx(6.5)
    assert R0_COVALENT == 0
    assert R5_LOCAL_NEIGHBORHOOD == 5


def test_classify_wrap_gate_leq_frozen_is_dehydron() -> None:
    assert classify_hbond_vs_dehydron(DEHYDRON_WRAP_MAX) == R2_DEHYDRON
    assert classify_hbond_vs_dehydron(0) == R2_DEHYDRON
    assert classify_hbond_vs_dehydron(DEHYDRON_WRAP_MAX + 1) == R1_HBOND


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


def test_set_dehydron_wrap_max_frozen_at_amended_value() -> None:
    from science.tokyo_eye.v8.r0_r5_graph import set_dehydron_wrap_max

    set_dehydron_wrap_max(DEHYDRON_WRAP_MAX)
    with pytest.raises(ValueError, match=f"frozen at {DEHYDRON_WRAP_MAX}"):
        set_dehydron_wrap_max(19)


def test_layer_b_exclusivity_property_and_fractions() -> None:
    from science.tokyo_eye.v8.r0_r5_graph import (
        assert_layer_b_exclusivity,
        edge_type_directed_fractions,
    )

    records = [_ala(i, (float(i) * 3.8, 0.0, 0.0)) for i in range(1, 6)]
    result = build_r0_r5_graph(records)
    assert_layer_b_exclusivity(result.edge_index, result.edge_type)
    fr = edge_type_directed_fractions(result.edge_type)
    for r in range(6):
        assert f"edge_frac_r{r}" in result.meta
        assert f"edge_frac_r{r}" in fr
    assert "n_r0" in result.meta
    # Duplicate Layer-B types on one undirected pair must fail.
    ei = np.array([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=np.int64)
    et = np.array([1, 1, 2, 2], dtype=np.int64)
    with pytest.raises(AssertionError, match="exclusivity"):
        assert_layer_b_exclusivity(ei, et)


def test_chemistry_gate_features_shape() -> None:
    from science.tokyo_eye.v8.r0_r5_graph import chemistry_gate_features

    records = [_ala(i, (float(i) * 3.8, 0.0, 0.0)) for i in range(1, 5)]
    result = build_r0_r5_graph(records)
    coords = np.stack(
        [np.array((float(i) * 3.8 + 1.5, 0.0, 0.0)) for i in range(1, 5)]
    )
    chem = chemistry_gate_features(
        result.num_nodes, result.edge_index, result.edge_type, coords
    )
    assert chem.shape == (4, 7)
    assert np.isfinite(chem).all()


# --- Freeze §5.2 property oracle (independent of resolve_layer_b_primary) ------

def _spec52_layer_b_primary(
    *,
    has_hbond: bool,
    wrap_count: int,
    has_pi_or_hydrophobic: bool,
    has_salt: bool,
    has_r5_neighborhood: bool,
    wrap_max: int = DEHYDRON_WRAP_MAX,
) -> int | None:
    """§5.2 descending Layer-B resolver written from the freeze, not the impl.

    1. R1 vs R2 via wrap (only if an H-bond exists)
    2. else R3 if hydrophobic / π
    3. else R4 if salt
    4. else R5 if Cβ neighborhood
    5. else no Layer-B edge
    Overlap: lowest numeric ID wins; R0 is not in this function.
    """
    if has_hbond:
        return R2_DEHYDRON if int(wrap_count) <= int(wrap_max) else R1_HBOND
    if has_pi_or_hydrophobic:
        return R3_HYDROPHOBIC_PI
    if has_salt:
        return R4_SALT_BRIDGE
    if has_r5_neighborhood:
        return R5_LOCAL_NEIGHBORHOOD
    return None


def _spec52_secondary_flags(
    *,
    has_hbond: bool,
    wrap_count: int,
    has_pi_or_hydrophobic: bool,
    has_salt: bool,
    has_r5_neighborhood: bool,
    wrap_max: int = DEHYDRON_WRAP_MAX,
) -> int:
    flags = 0
    if has_hbond:
        flags |= 1 << (
            R2_DEHYDRON if int(wrap_count) <= int(wrap_max) else R1_HBOND
        )
    if has_pi_or_hydrophobic:
        flags |= 1 << R3_HYDROPHOBIC_PI
    if has_salt:
        flags |= 1 << R4_SALT_BRIDGE
    if has_r5_neighborhood:
        flags |= 1 << R5_LOCAL_NEIGHBORHOOD
    return flags


def test_section_52_exhaustive_chemistry_bitmask() -> None:
    """Every 4-bit chemistry combination × wrap {0, τ, τ+1} vs the §5.2 oracle."""
    from science.tokyo_eye.v8.r0_r5_graph import resolve_layer_b_primary

    thr = DEHYDRON_WRAP_MAX
    for mask in range(16):
        has_hbond = bool(mask & 1)
        has_pi = bool(mask & 2)
        has_salt = bool(mask & 4)
        has_r5 = bool(mask & 8)
        for wrap in (0, thr, thr + 1):
            is_dehydron = has_hbond and wrap <= thr
            got, flags = resolve_layer_b_primary(
                has_hbond=has_hbond,
                is_dehydron=is_dehydron,
                has_pi_or_hydrophobic=has_pi,
                has_salt=has_salt,
                has_r5_neighborhood=has_r5,
            )
            want = _spec52_layer_b_primary(
                has_hbond=has_hbond,
                wrap_count=wrap,
                has_pi_or_hydrophobic=has_pi,
                has_salt=has_salt,
                has_r5_neighborhood=has_r5,
            )
            want_flags = _spec52_secondary_flags(
                has_hbond=has_hbond,
                wrap_count=wrap,
                has_pi_or_hydrophobic=has_pi,
                has_salt=has_salt,
                has_r5_neighborhood=has_r5,
            )
            assert got == want, (
                f"mask={mask} wrap={wrap}: got {got} want {want}"
            )
            assert flags == want_flags
            if want is not None:
                # Lowest numeric ID among all qualified Layer-B types.
                qualified = []
                if has_hbond:
                    qualified.append(R2_DEHYDRON if wrap <= thr else R1_HBOND)
                if has_pi:
                    qualified.append(R3_HYDROPHOBIC_PI)
                if has_salt:
                    qualified.append(R4_SALT_BRIDGE)
                if has_r5:
                    qualified.append(R5_LOCAL_NEIGHBORHOOD)
                assert want == min(qualified)
                if any(q <= R4_SALT_BRIDGE for q in qualified):
                    assert want != R5_LOCAL_NEIGHBORHOOD


def test_section_52_random_pair_distance_and_chemistry_fuzz() -> None:
    """Random pairwise seq/wrap/centroid/salt/Cβ inputs → §5.2 emission rules."""
    from science.tokyo_eye.v8.r0_r5_graph import (
        assert_layer_b_exclusivity,
        classify_hbond_vs_dehydron,
        resolve_layer_b_primary,
    )

    rng = np.random.default_rng(52)
    n_trials = 400
    n_nodes = 8
    for trial in range(n_trials):
        src: list[int] = []
        dst: list[int] = []
        types: list[int] = []
        # Layer A: peptide ±1 on a linear chain (always).
        for i in range(n_nodes - 1):
            src.extend([i, i + 1])
            dst.extend([i + 1, i])
            types.extend([R0_COVALENT, R0_COVALENT])

        # Random undirected pairs with independent chemistry + distances.
        seen_pairs: set[tuple[int, int]] = set()
        n_pairs = int(rng.integers(1, 12))
        for _ in range(n_pairs):
            i, j = (int(rng.integers(0, n_nodes)), int(rng.integers(0, n_nodes)))
            if i == j:
                continue
            key = (i, j) if i < j else (j, i)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)

            seq_delta = abs(i - j)  # chain index == residue index in this fuzz
            wrap = int(rng.integers(0, 28))
            # Distances relative to freeze §5.1 gates.
            centroid = float(rng.uniform(2.0, 8.0))
            salt_no = float(rng.uniform(2.0, 6.5))
            cb = float(rng.uniform(3.0, 12.0))
            dssp_e = float(rng.uniform(-1.2, 0.2))

            has_hbond = dssp_e <= -0.5
            has_pi = centroid <= 5.0
            has_salt = salt_no <= 4.0
            has_r5 = cb <= 8.0
            is_dehydron = has_hbond and wrap <= DEHYDRON_WRAP_MAX

            if has_hbond:
                assert classify_hbond_vs_dehydron(wrap) == (
                    R2_DEHYDRON if wrap <= DEHYDRON_WRAP_MAX else R1_HBOND
                )

            primary, flags = resolve_layer_b_primary(
                has_hbond=has_hbond,
                is_dehydron=is_dehydron,
                has_pi_or_hydrophobic=has_pi,
                has_salt=has_salt,
                has_r5_neighborhood=has_r5,
            )
            want = _spec52_layer_b_primary(
                has_hbond=has_hbond,
                wrap_count=wrap,
                has_pi_or_hydrophobic=has_pi,
                has_salt=has_salt,
                has_r5_neighborhood=has_r5,
            )
            assert primary == want, f"trial={trial} pair={key}"
            assert flags == _spec52_secondary_flags(
                has_hbond=has_hbond,
                wrap_count=wrap,
                has_pi_or_hydrophobic=has_pi,
                has_salt=has_salt,
                has_r5_neighborhood=has_r5,
            )

            seq_adjacent = seq_delta == 1
            if seq_adjacent:
                # Dual-layer: R0 already emitted; Layer B must coexist, not replace.
                r0_here = [
                    types[k]
                    for k in range(len(types))
                    if {src[k], dst[k]} == {i, j} and types[k] == R0_COVALENT
                ]
                assert len(r0_here) == 2
            if primary is None:
                continue
            src.extend([i, j])
            dst.extend([j, i])
            types.extend([primary, primary])

        ei = np.array([src, dst], dtype=np.int64)
        et = np.array(types, dtype=np.int64)
        assert_layer_b_exclusivity(ei, et)

        # Bidirectional invariance.
        seen = {(int(ei[0, k]), int(ei[1, k]), int(et[k])) for k in range(et.shape[0])}
        for a, b, r in list(seen):
            assert (b, a, r) in seen

        # Dual-layer: every seq-adjacent pair has R0; Layer B never swallows it.
        for i in range(n_nodes - 1):
            types_ij = sorted(
                int(et[k])
                for k in range(et.shape[0])
                if int(ei[0, k]) == i and int(ei[1, k]) == i + 1
            )
            assert R0_COVALENT in types_ij
            layer_b = [t for t in types_ij if t != R0_COVALENT]
            assert len(set(layer_b)) <= 1
            if layer_b:
                assert layer_b[0] >= R1_HBOND


def test_oracle_source_is_not_a_copy_of_the_loader() -> None:
    """Independence: oracle and implementation must remain distinct documents."""
    import inspect

    from science.tokyo_eye.v8.r0_r5_graph import resolve_layer_b_primary

    impl = "".join(inspect.getsource(resolve_layer_b_primary).split())
    oracle = "".join(inspect.getsource(_spec52_layer_b_primary).split())
    assert impl != oracle


def test_mutated_loader_r5_steals_r2_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    """Known-bad live mutation: R5 winning a qualifying R2 tie must fail §5.2.

    Same discipline as ``pure_hyp_pass``: the check is shown to fail on the
    bad case, not assumed trustworthy because 13/13 went green.
    """
    import science.tokyo_eye.v8.r0_r5_graph as g

    orig = g.resolve_layer_b_primary

    def r5_wins_ties(**kwargs):  # type: ignore[no-untyped-def]
        primary, flags = orig(**kwargs)
        if kwargs.get("has_r5_neighborhood") and kwargs.get("has_hbond"):
            return g.R5_LOCAL_NEIGHBORHOOD, flags
        return primary, flags

    monkeypatch.setattr(g, "resolve_layer_b_primary", r5_wins_ties)
    with pytest.raises(AssertionError, match=r"got 5 want 2"):
        test_section_52_exhaustive_chemistry_bitmask()

