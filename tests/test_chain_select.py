"""P_CHAIN_SELECT — two-sided chain scorer eligibility and selection gates.

Fixtures mirror production failure modes:
- 1BG1-like: DNA chain mis-tagged polymer wins without Cα guard
- 4OBE-like: duplicate KRAS entities; near-tie auto-selects chain A (no caller hints)
"""

from __future__ import annotations

import pytest

from science.dtie.ingest.chain_eligibility import (
    PROTEIN_CHAIN_MIN_CA_COUNT,
    PROTEIN_CHAIN_MIN_CA_FRACTION,
    passes_ca_eligibility,
)
from science.dtie.ingest.chain_scorer import (
    NoEligibleProteinChainError,
    score_chains,
)
from science.dtie.ingest.metadata import EntityMetadata, StructureMetadata
from science.dtie.ingest.parser import ParsedAtom, ParsedChain, ParsedResidue, ParsedStructure


def _ca_residue(seq_id: int, *, resolved: bool = True) -> ParsedResidue:
    atoms = []
    if resolved:
        atoms = [
            ParsedAtom(
                atom_name="CA",
                element="C",
                x=0.0,
                y=0.0,
                z=0.0,
                occupancy=1.0,
                b_factor=20.0,
                altloc=None,
                is_hetero=False,
                model_id=1,
            )
        ]
    return ParsedResidue(
        auth_seq_id=seq_id,
        label_seq_id=seq_id,
        insertion_code=None,
        residue_name="A",
        residue_name_3="ALA",
        comp_id="ALA",
        parent_comp_id=None,
        sse_code=None,
        is_resolved=resolved,
        is_modified=False,
        partial_backbone=not resolved,
        max_b_factor=20.0 if resolved else None,
        low_confidence_coords=False,
        atoms=atoms,
    )


def _dna_residue(seq_id: int) -> ParsedResidue:
    """Nucleic-acid residue: no Cα, partial backbone."""
    return ParsedResidue(
        auth_seq_id=seq_id,
        label_seq_id=seq_id,
        insertion_code=None,
        residue_name="X",
        residue_name_3=" DA",
        comp_id=" DA",
        parent_comp_id=None,
        sse_code=None,
        is_resolved=True,
        is_modified=False,
        partial_backbone=True,
        max_b_factor=30.0,
        low_confidence_coords=False,
        atoms=[
            ParsedAtom(
                atom_name="P",
                element="P",
                x=0.0,
                y=0.0,
                z=0.0,
                occupancy=1.0,
                b_factor=30.0,
                altloc=None,
                is_hetero=False,
                model_id=1,
            )
        ],
    )


def _protein_chain(
    label: str,
    *,
    entity_id: str,
    n_residues: int,
    entity_type: str = "polymer",
) -> ParsedChain:
    return ParsedChain(
        auth_asym_id=label,
        label_asym_id=label,
        entity_id=entity_id,
        entity_type=entity_type,
        residues=[_ca_residue(i) for i in range(1, n_residues + 1)],
    )


def _dna_chain(label: str, *, entity_id: str, n_residues: int) -> ParsedChain:
    return ParsedChain(
        auth_asym_id=label,
        label_asym_id=label,
        entity_id=entity_id,
        entity_type="polymer",
        residues=[_dna_residue(i) for i in range(1, n_residues + 1)],
    )


def _structure(pdb_id: str, chains: list[ParsedChain]) -> ParsedStructure:
    return ParsedStructure(
        pdb_id=pdb_id,
        method="X-RAY DIFFRACTION",
        resolution=2.0,
        r_factor=None,
        r_free=None,
        title="fixture",
        organism=None,
        release_date=None,
        polymer_composition="protein/na",
        model_count=1,
        assembly_id=None,
        chains=chains,
    )


def _metadata_with_uniprot(entity_id: str, accession: str) -> StructureMetadata:
    return StructureMetadata(
        entities=[
            EntityMetadata(
                entity_id=entity_id,
                uniprot_accessions=[accession],
                entity_type="polymer",
            )
        ]
    )


class TestPChainSelect:
    """P_CHAIN_SELECT — protein primary selection with Cα eligibility."""

    def test_1bg1_like_selects_protein_not_dna(self) -> None:
        """POSITIVE: STAT3 protein chain A wins; DNA chain B rejected by Cα floor."""
        stat3 = _protein_chain("A", entity_id="1", n_residues=559)
        dna = _dna_chain("B", entity_id="2", n_residues=639)
        metadata = _metadata_with_uniprot("2", "P42227")

        scope = score_chains(_structure("1bg1", [stat3, dna]), metadata)

        assert scope.primary_chain_ids == ["A"]
        assert not passes_ca_eligibility(dna)

    def test_single_chain_protein_selects_correctly(self) -> None:
        """POSITIVE: normal single-chain protein selects that chain."""
        chain = _protein_chain("A", entity_id="1", n_residues=76)
        scope = score_chains(_structure("1ubq", [chain]), None)
        assert scope.primary_chain_ids == ["A"]

    def test_only_non_protein_chains_fails_loudly(self) -> None:
        """NEGATIVE: no eligible protein chain → NoEligibleProteinChainError."""
        dna_only = _dna_chain("B", entity_id="1", n_residues=100)
        metadata = _metadata_with_uniprot("1", "P42227")

        with pytest.raises(NoEligibleProteinChainError) as exc:
            score_chains(_structure("xxxx", [dna_only]), metadata)

        assert "No eligible protein chain" in str(exc.value)

    def test_peptide_ligand_rejected_by_min_ca_count(self) -> None:
        """NEGATIVE: short peptide passes fraction but fails min CA count."""
        peptide = _protein_chain("P", entity_id="9", n_residues=8)
        protein = _protein_chain("A", entity_id="1", n_residues=120)

        scope = score_chains(_structure("pept", [peptide, protein]), None)

        assert scope.primary_chain_ids == ["A"]
        assert not passes_ca_eligibility(peptide)
        assert 8 < PROTEIN_CHAIN_MIN_CA_COUNT

    def test_4obe_like_duplicate_near_tie_selects_chain_a(self) -> None:
        """POSITIVE: duplicate KRAS entities; vanilla scorer picks A (near-tie rule)."""
        kras_a = _protein_chain("A", entity_id="1", n_residues=169)
        kras_b = _protein_chain("B", entity_id="1", n_residues=170)
        parsed = _structure("4obe", [kras_a, kras_b])
        metadata = _metadata_with_uniprot("1", "P01116")

        scope = score_chains(parsed, metadata)

        assert scope.primary_chain_ids == ["A"]
        assert scope.scope_source == "auto"

    def test_ubq_satisfies_ca_floor(self) -> None:
        """Calibration: smallest Stage-A protein (76 res) passes eligibility."""
        ubq = _protein_chain("A", entity_id="1", n_residues=76)
        assert passes_ca_eligibility(ubq)
        assert PROTEIN_CHAIN_MIN_CA_FRACTION <= 1.0
