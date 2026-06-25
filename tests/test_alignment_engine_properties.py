"""Property-based tests for the alignment engine.

Feature: structure-ingestion-normalization
Properties: 11, 12
Validates: Requirements 6.2, 6.3, 7.1, 7.2, 7.3

These tests validate:
- Alignment completeness with reason codes (Property 11)
- Kabsch superposition validity (Property 12)
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from science.dtie.alignment.kabsch_aligner import (
    ComparableCore,
    ResidueCoordinate,
    SuperpositionResult,
    build_comparable_core,
    compute_kabsch_superposition,
    MIN_COMMON_RESIDUES,
)
from science.dtie.alignment.sifts_mapper import (
    _build_alignment_records,
    _classify_unmapped_reason,
)
from science.dtie.common.ingest_payloads import ResidueAlignmentRecord
from science.dtie.common.keys import make_residue_id


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@st.composite
def st_residue_id_tuple(draw: st.DrawFn) -> tuple[int, str | None]:
    """Generate a (auth_seq_id, insertion_code) tuple."""
    seq_id = draw(st.integers(min_value=1, max_value=500))
    ins_code = draw(st.one_of(st.none(), st.sampled_from(list("ABCD"))))
    return (seq_id, ins_code)


@st.composite
def st_sifts_mapping(
    draw: st.DrawFn,
    residue_ids: list[tuple[int, str | None]],
    uniprot_accession: str = "P12345",
) -> tuple[list[dict], list[tuple[int, str | None]]]:
    """Generate a SIFTS mapping covering some subset of residue_ids.

    Returns (raw_mapping, residue_ids) where raw_mapping maps some
    residues to UniProt positions and leaves others unmapped.
    """
    # Decide how many residues are mapped (at least 1, at most all)
    n_mapped = draw(st.integers(min_value=1, max_value=len(residue_ids)))
    mapped_indices = draw(
        st.lists(
            st.integers(min_value=0, max_value=len(residue_ids) - 1),
            min_size=n_mapped,
            max_size=n_mapped,
            unique=True,
        )
    )

    raw_mapping: list[dict] = []
    uniprot_pos = 1
    for idx in sorted(mapped_indices):
        seq_id, _ = residue_ids[idx]
        raw_mapping.append({
            "pdb_seq_id": seq_id,
            "uniprot_position": uniprot_pos,
            "mapping_confidence": 1.0,
        })
        uniprot_pos += 1

    return raw_mapping, residue_ids


@st.composite
def st_3d_coords(
    draw: st.DrawFn,
    n_points: int | None = None,
) -> np.ndarray:
    """Generate Nx3 coordinate arrays."""
    n = n_points or draw(st.integers(min_value=MIN_COMMON_RESIDUES, max_value=50))
    coords = draw(
        st.lists(
            st.lists(
                st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
                min_size=3,
                max_size=3,
            ),
            min_size=n,
            max_size=n,
        )
    )
    return np.array(coords, dtype=np.float64)


@st.composite
def st_rotation_matrix(draw: st.DrawFn) -> np.ndarray:
    """Generate a valid rotation matrix (orthogonal, det=+1)."""
    # Generate random rotation via QR decomposition of random matrix
    random_matrix = draw(
        st.lists(
            st.lists(
                st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                min_size=3,
                max_size=3,
            ),
            min_size=3,
            max_size=3,
        )
    )
    M = np.array(random_matrix, dtype=np.float64)
    # Use SVD to get a proper rotation
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    # Ensure det = +1
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


@st.composite
def st_translation_vector(draw: st.DrawFn) -> np.ndarray:
    """Generate a translation vector."""
    t = draw(
        st.lists(
            st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False),
            min_size=3,
            max_size=3,
        )
    )
    return np.array(t, dtype=np.float64)


# ---------------------------------------------------------------------------
# Property 11: Alignment completeness with reason codes
# Feature: structure-ingestion-normalization, Property 11: Alignment completeness with reason codes
# ---------------------------------------------------------------------------


class TestAlignmentCompleteness:
    """Property 11: For any chain with UniProt accession and available SIFTS data,
    every residue in that chain SHALL have a fact_residue_alignment record.
    Mapped residues SHALL have non-null uniprot_position.
    Unmapped residues SHALL have uniprot_position=NULL with a non-null reason_code.
    """

    @given(
        n_residues=st.integers(min_value=3, max_value=30),
        n_mapped=st.integers(min_value=1, max_value=30),
    )
    @settings(max_examples=200)
    def test_all_residues_get_alignment_record(
        self, n_residues: int, n_mapped: int
    ):
        """Every residue in the chain produces exactly one alignment record."""
        # Feature: structure-ingestion-normalization, Property 11: Alignment completeness with reason codes
        n_mapped = min(n_mapped, n_residues)
        structure_id = "test1"
        chain_label = "A"
        uniprot_accession = "P12345"

        # Generate residue IDs
        residue_ids = [(i, None) for i in range(1, n_residues + 1)]

        # Generate SIFTS mapping covering first n_mapped residues
        raw_mapping = [
            {"pdb_seq_id": i, "uniprot_position": i, "mapping_confidence": 1.0}
            for i in range(1, n_mapped + 1)
        ]

        records = _build_alignment_records(
            raw_mapping=raw_mapping,
            structure_id=structure_id,
            chain_label=chain_label,
            uniprot_accession=uniprot_accession,
            chain_residue_ids=residue_ids,
        )

        # Property: one record per residue
        assert len(records) == n_residues

    @given(
        n_residues=st.integers(min_value=3, max_value=30),
        n_mapped=st.integers(min_value=1, max_value=30),
    )
    @settings(max_examples=200)
    def test_mapped_residues_have_uniprot_position(
        self, n_residues: int, n_mapped: int
    ):
        """Mapped residues have non-null uniprot_position."""
        # Feature: structure-ingestion-normalization, Property 11: Alignment completeness with reason codes
        n_mapped = min(n_mapped, n_residues)
        structure_id = "test1"
        chain_label = "A"
        uniprot_accession = "P12345"

        residue_ids = [(i, None) for i in range(1, n_residues + 1)]
        raw_mapping = [
            {"pdb_seq_id": i, "uniprot_position": i + 10, "mapping_confidence": 1.0}
            for i in range(1, n_mapped + 1)
        ]

        records = _build_alignment_records(
            raw_mapping=raw_mapping,
            structure_id=structure_id,
            chain_label=chain_label,
            uniprot_accession=uniprot_accession,
            chain_residue_ids=residue_ids,
        )

        mapped = [r for r in records if r.uniprot_position is not None]
        assert len(mapped) == n_mapped
        for r in mapped:
            assert r.reason_code is None

    @given(
        n_residues=st.integers(min_value=3, max_value=30),
        n_mapped=st.integers(min_value=0, max_value=29),
    )
    @settings(max_examples=200)
    def test_unmapped_residues_have_reason_code(
        self, n_residues: int, n_mapped: int
    ):
        """Unmapped residues have uniprot_position=None and non-null reason_code."""
        # Feature: structure-ingestion-normalization, Property 11: Alignment completeness with reason codes
        n_mapped = min(n_mapped, n_residues)
        structure_id = "test1"
        chain_label = "A"
        uniprot_accession = "P12345"

        residue_ids = [(i, None) for i in range(1, n_residues + 1)]
        raw_mapping = [
            {"pdb_seq_id": i, "uniprot_position": i, "mapping_confidence": 1.0}
            for i in range(1, n_mapped + 1)
        ]

        records = _build_alignment_records(
            raw_mapping=raw_mapping,
            structure_id=structure_id,
            chain_label=chain_label,
            uniprot_accession=uniprot_accession,
            chain_residue_ids=residue_ids,
        )

        unmapped = [r for r in records if r.uniprot_position is None]
        expected_unmapped = n_residues - n_mapped
        assert len(unmapped) == expected_unmapped
        for r in unmapped:
            assert r.reason_code is not None
            assert r.reason_code in {"tag", "engineered", "unmapped"}


# ---------------------------------------------------------------------------
# Property 12: Kabsch superposition validity
# Feature: structure-ingestion-normalization, Property 12: Kabsch superposition validity
# ---------------------------------------------------------------------------


class TestKabschValidity:
    """Property 12: For any structural superposition, the rotation_matrix SHALL be
    orthogonal (R^T·R ≈ I within tolerance 1e-6), det(R) SHALL be +1 (no reflection),
    RMSD SHALL be non-negative, and aligned_residue_count SHALL equal the number of
    positions in the comparable_core.
    """

    @given(
        rotation=st_rotation_matrix(),
        translation=st_translation_vector(),
        data=st.data(),
    )
    @settings(max_examples=200)
    def test_rotation_matrix_is_orthogonal(
        self, rotation: np.ndarray, translation: np.ndarray, data
    ):
        """Rotation matrix from Kabsch is orthogonal (R^T R ≈ I)."""
        # Feature: structure-ingestion-normalization, Property 12: Kabsch superposition validity
        n_points = data.draw(st.integers(min_value=MIN_COMMON_RESIDUES, max_value=30))

        # Generate reference coords
        ref_coords = data.draw(
            st.lists(
                st.lists(
                    st.floats(
                        min_value=-50.0, max_value=50.0,
                        allow_nan=False, allow_infinity=False
                    ),
                    min_size=3, max_size=3,
                ),
                min_size=n_points, max_size=n_points,
            )
        )
        ref_arr = np.array(ref_coords, dtype=np.float64)

        # Generate query by applying known rotation + translation + small noise
        query_arr = (rotation @ ref_arr.T).T + translation
        noise = data.draw(
            st.lists(
                st.lists(
                    st.floats(
                        min_value=-0.1, max_value=0.1,
                        allow_nan=False, allow_infinity=False
                    ),
                    min_size=3, max_size=3,
                ),
                min_size=n_points, max_size=n_points,
            )
        )
        query_arr = query_arr + np.array(noise, dtype=np.float64)

        core = ComparableCore(
            residue_pairs=[("q", "r")] * n_points,
            uniprot_positions=list(range(1, n_points + 1)),
            criteria={"test": True},
        )

        result = compute_kabsch_superposition(query_arr, ref_arr, core)
        assert result is not None

        R = result.rotation_matrix
        # R^T @ R should be identity
        identity_check = R.T @ R
        assert np.allclose(identity_check, np.eye(3), atol=1e-6), (
            f"R^T @ R is not identity:\n{identity_check}"
        )

    @given(
        rotation=st_rotation_matrix(),
        translation=st_translation_vector(),
        data=st.data(),
    )
    @settings(max_examples=200)
    def test_rotation_determinant_is_positive_one(
        self, rotation: np.ndarray, translation: np.ndarray, data
    ):
        """det(R) = +1 (no reflection)."""
        # Feature: structure-ingestion-normalization, Property 12: Kabsch superposition validity
        n_points = data.draw(st.integers(min_value=MIN_COMMON_RESIDUES, max_value=30))

        ref_coords = data.draw(
            st.lists(
                st.lists(
                    st.floats(
                        min_value=-50.0, max_value=50.0,
                        allow_nan=False, allow_infinity=False
                    ),
                    min_size=3, max_size=3,
                ),
                min_size=n_points, max_size=n_points,
            )
        )
        ref_arr = np.array(ref_coords, dtype=np.float64)
        query_arr = (rotation @ ref_arr.T).T + translation

        core = ComparableCore(
            residue_pairs=[("q", "r")] * n_points,
            uniprot_positions=list(range(1, n_points + 1)),
            criteria={"test": True},
        )

        result = compute_kabsch_superposition(query_arr, ref_arr, core)
        assert result is not None

        det_R = np.linalg.det(result.rotation_matrix)
        assert abs(det_R - 1.0) < 1e-6, f"det(R) = {det_R}, expected +1"

    @given(
        rotation=st_rotation_matrix(),
        translation=st_translation_vector(),
        data=st.data(),
    )
    @settings(max_examples=200)
    def test_rmsd_is_non_negative(
        self, rotation: np.ndarray, translation: np.ndarray, data
    ):
        """RMSD SHALL be non-negative."""
        # Feature: structure-ingestion-normalization, Property 12: Kabsch superposition validity
        n_points = data.draw(st.integers(min_value=MIN_COMMON_RESIDUES, max_value=30))

        ref_coords = data.draw(
            st.lists(
                st.lists(
                    st.floats(
                        min_value=-50.0, max_value=50.0,
                        allow_nan=False, allow_infinity=False
                    ),
                    min_size=3, max_size=3,
                ),
                min_size=n_points, max_size=n_points,
            )
        )
        ref_arr = np.array(ref_coords, dtype=np.float64)
        query_arr = (rotation @ ref_arr.T).T + translation

        core = ComparableCore(
            residue_pairs=[("q", "r")] * n_points,
            uniprot_positions=list(range(1, n_points + 1)),
            criteria={"test": True},
        )

        result = compute_kabsch_superposition(query_arr, ref_arr, core)
        assert result is not None
        assert result.rmsd >= 0.0

    @given(
        rotation=st_rotation_matrix(),
        translation=st_translation_vector(),
        data=st.data(),
    )
    @settings(max_examples=200)
    def test_aligned_count_matches_core(
        self, rotation: np.ndarray, translation: np.ndarray, data
    ):
        """aligned_residue_count SHALL equal the number of positions in comparable_core."""
        # Feature: structure-ingestion-normalization, Property 12: Kabsch superposition validity
        n_points = data.draw(st.integers(min_value=MIN_COMMON_RESIDUES, max_value=30))

        ref_coords = data.draw(
            st.lists(
                st.lists(
                    st.floats(
                        min_value=-50.0, max_value=50.0,
                        allow_nan=False, allow_infinity=False
                    ),
                    min_size=3, max_size=3,
                ),
                min_size=n_points, max_size=n_points,
            )
        )
        ref_arr = np.array(ref_coords, dtype=np.float64)
        query_arr = (rotation @ ref_arr.T).T + translation

        core = ComparableCore(
            residue_pairs=[("q", "r")] * n_points,
            uniprot_positions=list(range(1, n_points + 1)),
            criteria={"test": True},
        )

        result = compute_kabsch_superposition(query_arr, ref_arr, core)
        assert result is not None
        assert result.aligned_residue_count == n_points
        assert result.aligned_residue_count == len(core.uniprot_positions)

    @given(data=st.data())
    @settings(max_examples=100)
    def test_skips_when_fewer_than_min_residues(self, data):
        """Returns None when fewer than MIN_COMMON_RESIDUES points."""
        # Feature: structure-ingestion-normalization, Property 12: Kabsch superposition validity
        n_points = data.draw(st.integers(min_value=1, max_value=MIN_COMMON_RESIDUES - 1))

        coords = data.draw(
            st.lists(
                st.lists(
                    st.floats(
                        min_value=-50.0, max_value=50.0,
                        allow_nan=False, allow_infinity=False
                    ),
                    min_size=3, max_size=3,
                ),
                min_size=n_points, max_size=n_points,
            )
        )
        arr = np.array(coords, dtype=np.float64)

        core = ComparableCore(
            residue_pairs=[("q", "r")] * n_points,
            uniprot_positions=list(range(1, n_points + 1)),
            criteria={"test": True},
        )

        result = compute_kabsch_superposition(arr, arr, core)
        assert result is None

    @given(
        rotation=st_rotation_matrix(),
        translation=st_translation_vector(),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_perfect_alignment_has_near_zero_rmsd(
        self, rotation: np.ndarray, translation: np.ndarray, data
    ):
        """When query is an exact rigid-body transform of ref, RMSD ≈ 0."""
        # Feature: structure-ingestion-normalization, Property 12: Kabsch superposition validity
        n_points = data.draw(st.integers(min_value=MIN_COMMON_RESIDUES, max_value=20))

        ref_coords = data.draw(
            st.lists(
                st.lists(
                    st.floats(
                        min_value=-50.0, max_value=50.0,
                        allow_nan=False, allow_infinity=False
                    ),
                    min_size=3, max_size=3,
                ),
                min_size=n_points, max_size=n_points,
            )
        )
        ref_arr = np.array(ref_coords, dtype=np.float64)

        # Exact transform — no noise
        query_arr = (rotation @ ref_arr.T).T + translation

        core = ComparableCore(
            residue_pairs=[("q", "r")] * n_points,
            uniprot_positions=list(range(1, n_points + 1)),
            criteria={"test": True},
        )

        result = compute_kabsch_superposition(query_arr, ref_arr, core)
        assert result is not None
        assert result.rmsd < 1e-6, f"Expected near-zero RMSD, got {result.rmsd}"
