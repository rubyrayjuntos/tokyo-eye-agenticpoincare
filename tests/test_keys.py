"""Tests for canonical key generation (Priority 1: residue_id strategy)."""

from __future__ import annotations

import pytest

from science.dtie.common.keys import (
    coerce_residue_id,
    coerce_residue_ids,
    make_atom_id,
    make_chain_id,
    make_residue_id,
    make_site_id,
    make_structure_id,
    validate_residue_id,
)


class TestMakeStructureId:
    def test_rcsb_source(self):
        assert make_structure_id(pdb_id="4OBE", source="rcsb") == "4obe"

    def test_rcsb_strips_whitespace(self):
        assert make_structure_id(pdb_id=" 4OBE ", source="rcsb") == "4obe"

    def test_alphafold_source(self):
        assert make_structure_id(uniprot_id="P01116", source="alphafold") == "af2_P01116"

    def test_user_source(self):
        result = make_structure_id(
            name="kras_mutant", content_hash="a3f8c1d2e5b7", source="user"
        )
        assert result == "user_a3f8c1d2_kras_mutant"

    def test_rcsb_missing_pdb_id_raises(self):
        with pytest.raises(ValueError, match="pdb_id required"):
            make_structure_id(source="rcsb")

    def test_alphafold_missing_uniprot_raises(self):
        with pytest.raises(ValueError, match="uniprot_id required"):
            make_structure_id(source="alphafold")

    def test_user_missing_fields_raises(self):
        with pytest.raises(ValueError):
            make_structure_id(source="user", name="test")

    def test_derived_missing_fields_raises(self):
        with pytest.raises(ValueError):
            make_structure_id(source="derived")

    def test_unknown_source_raises(self):
        with pytest.raises(ValueError, match="Unknown source"):
            make_structure_id(source="magic")


class TestMakeChainId:
    def test_basic(self):
        assert make_chain_id("4obe", "A") == "4obe:A"

    def test_invalid_chain_label_raises(self):
        with pytest.raises(ValueError, match="Invalid chain_label"):
            make_chain_id("4obe", "A!")


class TestMakeResidueId:
    def test_basic(self):
        assert make_residue_id("4obe", "A", 12) == "4obe:A:12"

    def test_with_insertion_code(self):
        assert make_residue_id("4obe", "B", 145, "A") == "4obe:B:145:A"

    def test_no_insertion_code(self):
        result = make_residue_id("4obe", "A", 61, None)
        assert result == "4obe:A:61"
        assert "None" not in result

    def test_invalid_insertion_code_raises(self):
        with pytest.raises(ValueError, match="Invalid insertion_code"):
            make_residue_id("4obe", "A", 12, "AB")

    def test_numeric_insertion_code_raises(self):
        with pytest.raises(ValueError, match="Invalid insertion_code"):
            make_residue_id("4obe", "A", 12, "1")

    def test_deterministic(self):
        """Same inputs always produce same output (critical for cross-run joins)."""
        id1 = make_residue_id("4obe", "A", 12)
        id2 = make_residue_id("4obe", "A", 12)
        assert id1 == id2

    def test_different_structures_different_ids(self):
        id1 = make_residue_id("4obe", "A", 12)
        id2 = make_residue_id("7xkj", "A", 12)
        assert id1 != id2


class TestMakeAtomId:
    def test_basic(self):
        assert make_atom_id("4obe", "A", 12, "CA") == "4obe:A:12:CA"

    def test_with_alt_loc(self):
        assert make_atom_id("4obe", "A", 12, "CA", alt_loc="B") == "4obe:A:12:CA:B"

    def test_with_insertion_code(self):
        assert make_atom_id("4obe", "A", 12, "CA", insertion_code="A") == "4obe:A:12:A:CA"


class TestMakeSiteId:
    def test_basic(self):
        assert make_site_id("4obe", "source_leak", 1) == "4obe:site:source_leak:1"

    def test_normalizes_type(self):
        assert make_site_id("4obe", "Source Leak", 2) == "4obe:site:source_leak:2"


class TestValidateResidueId:
    def test_valid_basic(self):
        assert validate_residue_id("4obe:A:12") is True

    def test_valid_with_insertion(self):
        assert validate_residue_id("4obe:B:145:A") is True

    def test_valid_alphafold(self):
        assert validate_residue_id("af2_P01116:A:61") is True

    def test_invalid_no_colon(self):
        assert validate_residue_id("4obeA12") is False

    def test_invalid_missing_index(self):
        assert validate_residue_id("4obe:A:") is False

    def test_invalid_uuid(self):
        assert validate_residue_id("550e8400-e29b-41d4-a716-446655440000") is False


class TestCoerceResidueId:
    def test_pdb_style_chain_index(self):
        assert coerce_residue_id("A:379", "11qe") == "11qe:A:379"

    def test_already_canonical(self):
        assert coerce_residue_id("11qe:A:379", "11qe") == "11qe:A:379"

    def test_with_insertion_code(self):
        assert coerce_residue_id("A:145:A", "4obe") == "4obe:A:145:A"

    def test_deduplicates_list(self):
        assert coerce_residue_ids(
            ["A:379", "11qe:A:379", "A:379"],
            "11qe",
        ) == ["11qe:A:379"]
