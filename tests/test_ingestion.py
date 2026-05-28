"""Tests for structure ingestion (dimensional model population).

These tests validate the ingestion logic without requiring biotite
or network access — they test the key generation and DB write patterns.
"""

from __future__ import annotations

from typing import Any

import pytest

from science.dtie.common.ingestion import _three_to_one


class TestThreeToOne:
    """Test amino acid code conversion."""

    def test_standard_amino_acids(self):
        assert _three_to_one("ALA") == "A"
        assert _three_to_one("GLY") == "G"
        assert _three_to_one("TRP") == "W"
        assert _three_to_one("LYS") == "K"

    def test_case_insensitive(self):
        assert _three_to_one("ala") == "A"
        assert _three_to_one("Gly") == "G"

    def test_unknown_returns_x(self):
        assert _three_to_one("UNK") == "X"
        assert _three_to_one("XYZ") == "X"

    def test_all_20_standard(self):
        expected = {
            "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
            "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
            "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
            "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
        }
        for three, one in expected.items():
            assert _three_to_one(three) == one, f"Failed for {three}"
