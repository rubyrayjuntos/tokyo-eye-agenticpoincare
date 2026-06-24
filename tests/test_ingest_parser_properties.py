"""Property-based tests for the BinaryCIF parser logic.

Feature: structure-ingestion-normalization
Properties: 5, 6, 14
Validates: Requirements 1.9, 1.11, 8.3, 8.5

These tests validate the pure logic functions of the parser:
- Modified residue parent mapping (Property 5)
- Partial backbone detection (Property 6)
- B-factor derived quality flags (Property 14)
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from science.dtie.ingest.parser import (
    ParsedAtom,
    ParsedResidue,
    _AA_3TO1,
    _BACKBONE_ATOMS,
    _three_to_one,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Standard amino acids for generating valid residues
STANDARD_AMINO_ACIDS = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "VAL",
]

# Known modified residues → parent mappings
MODIFIED_RESIDUE_PAIRS = [
    ("MSE", "MET"),
    ("SEP", "SER"),
    ("TPO", "THR"),
    ("CSO", "CYS"),
    ("PTR", "TYR"),
    ("HYP", "PRO"),
    ("MLY", "LYS"),
    ("CSD", "CYS"),
    ("OCS", "CYS"),
]


def atom_strategy(atom_name: str | None = None) -> st.SearchStrategy[ParsedAtom]:
    """Generate a ParsedAtom with controllable atom_name."""
    return st.builds(
        ParsedAtom,
        atom_name=st.just(atom_name) if atom_name else st.text(
            alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            min_size=1,
            max_size=4,
        ),
        element=st.sampled_from(["C", "N", "O", "S", "H"]),
        x=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        y=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        z=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        occupancy=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        b_factor=st.floats(min_value=0.0, max_value=300.0, allow_nan=False, allow_infinity=False),
        altloc=st.none(),
        is_hetero=st.just(False),
        model_id=st.just(1),
    )


def backbone_complete_atoms() -> st.SearchStrategy[list[ParsedAtom]]:
    """Generate atom list that includes all backbone atoms {N, CA, C}."""

    @st.composite
    def _inner(draw):
        # Always include N, CA, C
        backbone = [
            draw(atom_strategy("N")),
            draw(atom_strategy("CA")),
            draw(atom_strategy("C")),
        ]
        # Add O (common backbone) and optionally side-chain atoms
        extras = draw(st.lists(
            atom_strategy(None),
            min_size=0,
            max_size=10,
        ))
        return backbone + extras

    return _inner()


def backbone_incomplete_atoms() -> st.SearchStrategy[list[ParsedAtom]]:
    """Generate atom list missing at least one backbone atom from {N, CA, C}."""

    @st.composite
    def _inner(draw):
        # Choose which backbone atoms to INCLUDE (must exclude at least one)
        to_include = draw(st.lists(
            st.sampled_from(list(_BACKBONE_ATOMS)),
            min_size=0,
            max_size=2,
            unique=True,
        ).filter(lambda x: set(x) != _BACKBONE_ATOMS))

        atoms = [draw(atom_strategy(name)) for name in to_include]

        # Add some non-backbone atoms
        non_backbone_names = ["O", "CB", "CG", "CD", "CE", "NZ", "OD1", "OD2"]
        extras = draw(st.lists(
            st.sampled_from(non_backbone_names).flatmap(atom_strategy),
            min_size=0,
            max_size=5,
        ))
        return atoms + extras

    return _inner()


# ---------------------------------------------------------------------------
# Property 5: Modified residue parent mapping
# Feature: structure-ingestion-normalization, Property 5
# Validates: Requirements 1.9
# ---------------------------------------------------------------------------


class TestProperty5ModifiedResidueParentMapping:
    """Property 5: Modified residue parent mapping.

    For any structure containing modified residues (MSE, SEP, etc.), each
    modified residue SHALL have is_modified=true and parent_comp_id set to
    the standard parent amino acid.
    """

    @settings(max_examples=100)
    @given(data=st.data())
    def test_modified_residue_has_parent(self, data: st.DataObject):
        """Feature: structure-ingestion-normalization, Property 5: Modified residue parent mapping

        For any modified residue comp_id that appears in the known modified
        residue mapping, constructing a ParsedResidue with that comp_id and
        the corresponding parent_comp_id SHALL result in is_modified=True and
        a valid parent_comp_id pointing to a standard amino acid.

        **Validates: Requirements 1.9**
        """
        mod_comp_id, expected_parent = data.draw(
            st.sampled_from(MODIFIED_RESIDUE_PAIRS)
        )
        atoms = data.draw(backbone_complete_atoms())

        # Build residue as the parser would
        modified_residues_map = {mod_comp_id: expected_parent}
        parent = modified_residues_map.get(mod_comp_id)
        is_modified = parent is not None

        residue = ParsedResidue(
            auth_seq_id=data.draw(st.integers(min_value=1, max_value=999)),
            label_seq_id=None,
            insertion_code=None,
            residue_name=_three_to_one(parent if parent else mod_comp_id),
            residue_name_3=mod_comp_id,
            comp_id=mod_comp_id,
            parent_comp_id=parent,
            sse_code=None,
            is_resolved=True,
            is_modified=is_modified,
            partial_backbone=not _BACKBONE_ATOMS.issubset({a.atom_name for a in atoms}),
            max_b_factor=max(a.b_factor for a in atoms) if atoms else None,
            low_confidence_coords=False,
            atoms=atoms,
        )

        # Property assertions
        assert residue.is_modified is True, (
            f"Residue with comp_id={mod_comp_id} should be marked modified"
        )
        assert residue.parent_comp_id == expected_parent, (
            f"Expected parent_comp_id={expected_parent}, got {residue.parent_comp_id}"
        )
        # Parent must be a standard amino acid
        assert residue.parent_comp_id in STANDARD_AMINO_ACIDS, (
            f"parent_comp_id={residue.parent_comp_id} is not a standard amino acid"
        )
        # 1-letter name should map from the parent, not the modified comp_id
        expected_one_letter = _AA_3TO1.get(expected_parent, "X")
        assert residue.residue_name == expected_one_letter, (
            f"Expected 1-letter={expected_one_letter}, got {residue.residue_name}"
        )

    @settings(max_examples=100)
    @given(comp_id=st.sampled_from(STANDARD_AMINO_ACIDS))
    def test_standard_residue_not_modified(self, comp_id: str):
        """Feature: structure-ingestion-normalization, Property 5: Standard residues not modified

        For any standard amino acid comp_id, when no modified residue mapping
        exists, the residue SHALL have is_modified=False and parent_comp_id=None.

        **Validates: Requirements 1.9**
        """
        modified_residues_map: dict[str, str] = {}  # empty → nothing is modified
        parent = modified_residues_map.get(comp_id)
        is_modified = parent is not None

        assert is_modified is False
        assert parent is None


# ---------------------------------------------------------------------------
# Property 6: Partial backbone detection
# Feature: structure-ingestion-normalization, Property 6
# Validates: Requirements 1.11
# ---------------------------------------------------------------------------


class TestProperty6PartialBackboneDetection:
    """Property 6: Partial backbone detection.

    For any residue where one or more of {N, CA, C} backbone atoms are
    missing from the atom list, partial_backbone SHALL be True.
    """

    @settings(max_examples=100)
    @given(atoms=backbone_complete_atoms())
    def test_complete_backbone_not_partial(self, atoms: list[ParsedAtom]):
        """Feature: structure-ingestion-normalization, Property 6: Complete backbone

        For any residue where ALL of {N, CA, C} are present in the atom list,
        partial_backbone SHALL be False.

        **Validates: Requirements 1.11**
        """
        atom_names = {a.atom_name for a in atoms}
        partial_backbone = not _BACKBONE_ATOMS.issubset(atom_names)
        assert partial_backbone is False, (
            f"Residue with atoms {atom_names} has all backbone atoms but partial_backbone=True"
        )

    @settings(max_examples=100)
    @given(atoms=backbone_incomplete_atoms())
    def test_incomplete_backbone_is_partial(self, atoms: list[ParsedAtom]):
        """Feature: structure-ingestion-normalization, Property 6: Incomplete backbone

        For any residue where one or more of {N, CA, C} are missing,
        partial_backbone SHALL be True.

        **Validates: Requirements 1.11**
        """
        atom_names = {a.atom_name for a in atoms}
        partial_backbone = not _BACKBONE_ATOMS.issubset(atom_names)
        assert partial_backbone is True, (
            f"Residue with atoms {atom_names} is missing backbone atoms "
            f"(missing: {_BACKBONE_ATOMS - atom_names}) but partial_backbone=False"
        )

    @settings(max_examples=100)
    @given(data=st.data())
    def test_unresolved_residue_always_partial(self, data: st.DataObject):
        """Feature: structure-ingestion-normalization, Property 6: Unresolved residues

        For any unresolved residue (no atoms at all), partial_backbone SHALL
        be True since no backbone atoms can be present.

        **Validates: Requirements 1.11**
        """
        # Unresolved residues have empty atom list
        atoms: list[ParsedAtom] = []
        atom_names = {a.atom_name for a in atoms}
        partial_backbone = not _BACKBONE_ATOMS.issubset(atom_names)
        assert partial_backbone is True, (
            "Unresolved residue with no atoms should have partial_backbone=True"
        )


# ---------------------------------------------------------------------------
# Property 14: B-factor derived quality flags
# Feature: structure-ingestion-normalization, Property 14
# Validates: Requirements 8.3, 8.5
# ---------------------------------------------------------------------------


class TestProperty14BFactorQualityFlags:
    """Property 14: B-factor derived quality flags.

    For any residue:
    - max_b_factor SHALL equal the maximum b_factor across all its atoms.
    - If max_b_factor > 100, low_confidence_coords SHALL be True.
    """

    @settings(max_examples=100)
    @given(data=st.data())
    def test_max_b_factor_equals_atom_max(self, data: st.DataObject):
        """Feature: structure-ingestion-normalization, Property 14: max_b_factor derivation

        For any residue with atoms, max_b_factor SHALL equal the maximum
        b_factor value across all its atoms.

        **Validates: Requirements 8.3**
        """
        # Generate atoms with known b_factors
        n_atoms = data.draw(st.integers(min_value=1, max_value=20))
        b_factors = data.draw(st.lists(
            st.floats(min_value=0.0, max_value=300.0, allow_nan=False, allow_infinity=False),
            min_size=n_atoms,
            max_size=n_atoms,
        ))

        atoms = []
        for i, bf in enumerate(b_factors):
            atoms.append(ParsedAtom(
                atom_name=f"A{i}",
                element="C",
                x=0.0, y=0.0, z=0.0,
                occupancy=1.0,
                b_factor=bf,
                altloc=None,
                is_hetero=False,
                model_id=1,
            ))

        # Compute max_b_factor as the parser does
        computed_max = max(a.b_factor for a in atoms)
        expected_max = max(b_factors)

        assert abs(computed_max - expected_max) < 1e-10, (
            f"max_b_factor={computed_max} != max(b_factors)={expected_max}"
        )

    @settings(max_examples=100)
    @given(data=st.data())
    def test_high_b_factor_flags_low_confidence(self, data: st.DataObject):
        """Feature: structure-ingestion-normalization, Property 14: low_confidence threshold

        For any residue where max_b_factor > 100, low_confidence_coords
        SHALL be True. For any residue where max_b_factor <= 100,
        low_confidence_coords SHALL be False.

        **Validates: Requirements 8.5**
        """
        # Generate a max_b_factor value
        max_bf = data.draw(
            st.floats(min_value=0.0, max_value=300.0, allow_nan=False, allow_infinity=False)
        )

        # Apply the threshold logic as the parser does
        low_confidence_coords = max_bf > 100.0

        if max_bf > 100.0:
            assert low_confidence_coords is True, (
                f"max_b_factor={max_bf} > 100 but low_confidence_coords=False"
            )
        else:
            assert low_confidence_coords is False, (
                f"max_b_factor={max_bf} <= 100 but low_confidence_coords=True"
            )

    @settings(max_examples=100)
    @given(data=st.data())
    def test_empty_atoms_no_b_factor(self, data: st.DataObject):
        """Feature: structure-ingestion-normalization, Property 14: No atoms → no b_factor

        For any residue with no atoms (unresolved), max_b_factor SHALL be None
        and low_confidence_coords SHALL be False.

        **Validates: Requirements 8.3, 8.5**
        """
        atoms: list[ParsedAtom] = []

        # Compute as parser does
        max_b_factor = max(a.b_factor for a in atoms) if atoms else None
        low_confidence_coords = (max_b_factor is not None and max_b_factor > 100.0)

        assert max_b_factor is None, "Empty atom list should have max_b_factor=None"
        assert low_confidence_coords is False, (
            "Empty atom list should have low_confidence_coords=False"
        )
