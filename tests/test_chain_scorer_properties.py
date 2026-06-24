"""Property-based tests for the chain scorer.

Feature: structure-ingestion-normalization
Properties: 7, 8
Validates: Requirements 2.4, 4.1, 4.6

These tests validate:
- Duplicate entity detection (Property 7)
- Chain scorer selects highest quality (Property 8)
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from science.dtie.ingest.chain_scorer import (
    ComputationScope,
    score_chains,
    score_chains_interface,
    _detect_duplicate_entities,
    _is_protein_chain,
    _score_single_chain,
)
from science.dtie.ingest.metadata import EntityMetadata, StructureMetadata
from science.dtie.ingest.parser import ParsedChain, ParsedResidue, ParsedStructure


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_CHAIN_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

_PROTEIN_TYPES = ["polypeptide(l)", "polypeptide(d)", "polymer", "protein"]
_NON_PROTEIN_TYPES = ["polyribonucleotide", "polydeoxyribonucleotide", "nucleic_acid", "water"]


@st.composite
def st_residue(
    draw: st.DrawFn,
    auth_seq_id: int | None = None,
    is_resolved: bool = True,
    is_modified: bool = False,
    max_b_factor: float | None = None,
) -> ParsedResidue:
    """Generate a ParsedResidue with controllable parameters."""
    seq_id = auth_seq_id if auth_seq_id is not None else draw(st.integers(1, 999))
    bf = max_b_factor if max_b_factor is not None else draw(
        st.floats(min_value=5.0, max_value=150.0, allow_nan=False, allow_infinity=False)
    )
    return ParsedResidue(
        auth_seq_id=seq_id,
        label_seq_id=seq_id,
        insertion_code=None,
        residue_name="A",
        residue_name_3="ALA",
        comp_id="ALA",
        parent_comp_id="ALA" if is_modified else None,
        sse_code=None,
        is_resolved=is_resolved,
        is_modified=is_modified,
        partial_backbone=not is_resolved,
        max_b_factor=bf if is_resolved else None,
        low_confidence_coords=bf > 100.0 if is_resolved else False,
        atoms=[],
    )


@st.composite
def st_protein_chain(
    draw: st.DrawFn,
    auth_asym_id: str | None = None,
    entity_id: str | None = None,
    min_residues: int = 5,
    max_residues: int = 50,
    b_factor_range: tuple[float, float] = (5.0, 150.0),
) -> ParsedChain:
    """Generate a protein chain with controllable quality."""
    chain_label = auth_asym_id or draw(st.sampled_from(_CHAIN_LABELS))
    eid = entity_id or draw(st.text(alphabet="123456789", min_size=1, max_size=2))
    n_residues = draw(st.integers(min_value=min_residues, max_value=max_residues))

    residues = []
    for i in range(1, n_residues + 1):
        bf = draw(st.floats(
            min_value=b_factor_range[0], max_value=b_factor_range[1],
            allow_nan=False, allow_infinity=False,
        ))
        is_resolved = draw(st.booleans()) if draw(st.integers(0, 9)) > 1 else False
        # Bias toward resolved (80% resolved)
        is_resolved = True if draw(st.integers(0, 9)) < 8 else is_resolved
        residues.append(ParsedResidue(
            auth_seq_id=i,
            label_seq_id=i,
            insertion_code=None,
            residue_name="A",
            residue_name_3="ALA",
            comp_id="ALA",
            parent_comp_id=None,
            sse_code=None,
            is_resolved=is_resolved,
            is_modified=False,
            partial_backbone=not is_resolved,
            max_b_factor=bf if is_resolved else None,
            low_confidence_coords=(bf > 100.0) if is_resolved else False,
            atoms=[],
        ))

    return ParsedChain(
        auth_asym_id=chain_label,
        label_asym_id=chain_label,
        entity_id=eid,
        entity_type=draw(st.sampled_from(_PROTEIN_TYPES)),
        residues=residues,
    )


@st.composite
def st_structure_with_duplicates(draw: st.DrawFn) -> ParsedStructure:
    """Generate a structure with at least one duplicate entity group.

    Ensures at least 2 chains share the same entity_id.
    """
    # Create a duplicate entity group: 2-3 chains with same entity_id
    dup_entity_id = "1"
    n_dups = draw(st.integers(min_value=2, max_value=3))
    dup_labels = _CHAIN_LABELS[:n_dups]

    dup_chains = []
    for label in dup_labels:
        chain = draw(st_protein_chain(
            auth_asym_id=label,
            entity_id=dup_entity_id,
            min_residues=10,
            max_residues=40,
        ))
        dup_chains.append(chain)

    # Optionally add a non-duplicate chain
    extra_chains = []
    if draw(st.booleans()):
        extra_label = _CHAIN_LABELS[n_dups]
        extra_chain = draw(st_protein_chain(
            auth_asym_id=extra_label,
            entity_id="2",
            min_residues=5,
            max_residues=30,
        ))
        extra_chains.append(extra_chain)

    all_chains = dup_chains + extra_chains

    return ParsedStructure(
        pdb_id=draw(st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=4, max_size=4)),
        method="X-RAY DIFFRACTION",
        resolution=draw(st.floats(1.0, 4.0, allow_nan=False, allow_infinity=False)),
        r_factor=None,
        r_free=None,
        title="Test structure",
        organism=None,
        release_date=None,
        polymer_composition="protein",
        model_count=1,
        assembly_id=None,
        chains=all_chains,
    )


@st.composite
def st_multi_chain_structure(draw: st.DrawFn) -> tuple[ParsedStructure, StructureMetadata | None]:
    """Generate a multi-chain structure with at least one protein chain.

    Returns (structure, optional_metadata).
    """
    # At least 2 chains, at least 1 protein
    n_chains = draw(st.integers(min_value=2, max_value=5))
    labels = _CHAIN_LABELS[:n_chains]

    # First chain is always protein
    chains = []
    entities = []
    for i, label in enumerate(labels):
        entity_id = str(i + 1)
        if i == 0:
            entity_type = draw(st.sampled_from(_PROTEIN_TYPES))
        else:
            entity_type = draw(st.sampled_from(_PROTEIN_TYPES + _NON_PROTEIN_TYPES))

        n_residues = draw(st.integers(min_value=5, max_value=40))
        bf_range = draw(st.tuples(
            st.floats(5.0, 50.0, allow_nan=False, allow_infinity=False),
            st.floats(51.0, 150.0, allow_nan=False, allow_infinity=False),
        ))

        residues = []
        for r in range(1, n_residues + 1):
            bf = draw(st.floats(bf_range[0], bf_range[1], allow_nan=False, allow_infinity=False))
            residues.append(ParsedResidue(
                auth_seq_id=r,
                label_seq_id=r,
                insertion_code=None,
                residue_name="A",
                residue_name_3="ALA",
                comp_id="ALA",
                parent_comp_id=None,
                sse_code=None,
                is_resolved=True,
                is_modified=False,
                partial_backbone=False,
                max_b_factor=bf,
                low_confidence_coords=bf > 100.0,
                atoms=[],
            ))

        chains.append(ParsedChain(
            auth_asym_id=label,
            label_asym_id=label,
            entity_id=entity_id,
            entity_type=entity_type,
            residues=residues,
        ))

        # Build entity metadata (50% chance of having UniProt)
        has_uniprot = draw(st.booleans())
        entities.append(EntityMetadata(
            entity_id=entity_id,
            uniprot_accessions=[f"P{draw(st.integers(10000, 99999))}"] if has_uniprot else [],
            entity_type=entity_type,
        ))

    parsed = ParsedStructure(
        pdb_id=draw(st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=4, max_size=4)),
        method="X-RAY DIFFRACTION",
        resolution=draw(st.floats(1.0, 4.0, allow_nan=False, allow_infinity=False)),
        r_factor=None,
        r_free=None,
        title="Multi-chain test",
        organism=None,
        release_date=None,
        polymer_composition="protein",
        model_count=1,
        assembly_id=None,
        chains=chains,
    )

    # 70% chance of having metadata
    has_metadata = draw(st.integers(0, 9)) < 7
    metadata = StructureMetadata(entities=entities) if has_metadata else None

    return parsed, metadata


# ---------------------------------------------------------------------------
# Property 7: Duplicate entity detection
# Feature: structure-ingestion-normalization, Property 7
# Validates: Requirements 2.4, 4.6
# ---------------------------------------------------------------------------


class TestProperty7DuplicateEntityDetection:
    """Property 7: Duplicate entity detection.

    For any structure with multiple chains sharing the same entity_id, all
    chains except the representative SHALL have is_entity_duplicate=true in
    the scope exclusion, and exactly one chain per entity_id SHALL be selected
    as representative.
    """

    @settings(max_examples=100)
    @given(parsed=st_structure_with_duplicates())
    def test_exactly_one_representative_per_entity(self, parsed: ParsedStructure):
        """Feature: structure-ingestion-normalization, Property 7: Duplicate entity detection

        For any structure with multiple chains sharing the same entity_id,
        exactly one chain per entity_id SHALL be selected as representative
        (not excluded), and the rest SHALL be excluded.

        **Validates: Requirements 2.4, 4.6**
        """
        scope = score_chains(parsed, None)

        # Identify duplicate entity groups
        from collections import Counter
        entity_counts = Counter(c.entity_id for c in parsed.chains)
        duplicate_eids = {eid for eid, cnt in entity_counts.items() if cnt > 1}

        # For each duplicate entity group, exactly one chain should NOT be excluded
        for dup_eid in duplicate_eids:
            chains_in_group = [c for c in parsed.chains if c.entity_id == dup_eid]
            group_labels = {c.auth_asym_id for c in chains_in_group}

            # Count how many from this group are NOT excluded
            not_excluded = group_labels - set(scope.exclude_chain_ids)

            # The primary chain might be from this group
            primary_in_group = set(scope.primary_chain_ids) & group_labels

            # At most one representative should be active (not excluded)
            # It's either the primary or just not excluded
            representatives_count = len(not_excluded)
            assert representatives_count == 1, (
                f"Entity {dup_eid} with chains {group_labels}: expected exactly 1 "
                f"representative (not excluded), got {representatives_count}. "
                f"Not excluded: {not_excluded}, Excluded: {set(scope.exclude_chain_ids) & group_labels}"
            )

    @settings(max_examples=100)
    @given(parsed=st_structure_with_duplicates())
    def test_non_representatives_are_excluded(self, parsed: ParsedStructure):
        """Feature: structure-ingestion-normalization, Property 7: Non-reps excluded

        For any duplicate entity group, all chains that are NOT the representative
        SHALL appear in exclude_chain_ids.

        **Validates: Requirements 2.4, 4.6**
        """
        scope = score_chains(parsed, None)

        from collections import Counter
        entity_counts = Counter(c.entity_id for c in parsed.chains)
        duplicate_eids = {eid for eid, cnt in entity_counts.items() if cnt > 1}

        for dup_eid in duplicate_eids:
            chains_in_group = [c for c in parsed.chains if c.entity_id == dup_eid]
            group_labels = {c.auth_asym_id for c in chains_in_group}

            # Find which one is the representative (not excluded)
            not_excluded = group_labels - set(scope.exclude_chain_ids)
            excluded_from_group = group_labels & set(scope.exclude_chain_ids)

            # All non-primary chains in the group should be excluded
            assert len(excluded_from_group) == len(chains_in_group) - 1, (
                f"Entity {dup_eid}: expected {len(chains_in_group) - 1} excluded, "
                f"got {len(excluded_from_group)}"
            )


# ---------------------------------------------------------------------------
# Property 8: Chain scorer selects highest quality
# Feature: structure-ingestion-normalization, Property 8
# Validates: Requirements 4.1
# ---------------------------------------------------------------------------


class TestProperty8ChainScorerSelectsHighestQuality:
    """Property 8: Chain scorer selects highest quality.

    For any structure with multiple protein chains, the chain scorer SHALL
    assign the highest score to the chain with the best combination of quality
    signals, and the selected primary_chain SHALL always be a protein chain.
    """

    @settings(max_examples=100)
    @given(data=st.data())
    def test_primary_is_always_protein(
        self, data: st.DataObject
    ):
        """Feature: structure-ingestion-normalization, Property 8: Primary is protein

        For any structure with at least one protein chain, the selected
        primary_chain SHALL always be a protein chain (never nucleic acid
        or ligand).

        **Validates: Requirements 4.1**
        """
        parsed, metadata = data.draw(st_multi_chain_structure())

        # Ensure at least one protein chain exists (strategy guarantees this)
        protein_chains = [c for c in parsed.chains if _is_protein_chain(c)]
        assert len(protein_chains) >= 1, "Strategy should ensure at least 1 protein chain"

        scope = score_chains(parsed, metadata)

        # Primary chain must be protein
        if scope.primary_chain_ids:
            primary_label = scope.primary_chain_ids[0]
            primary_chain = next(
                c for c in parsed.chains if c.auth_asym_id == primary_label
            )
            assert _is_protein_chain(primary_chain), (
                f"Primary chain {primary_label} has entity_type='{primary_chain.entity_type}' "
                f"which is NOT a protein type. Protein chains available: "
                f"{[c.auth_asym_id for c in protein_chains]}"
            )

    @settings(max_examples=100)
    @given(data=st.data())
    def test_highest_scorer_is_primary(
        self, data: st.DataObject
    ):
        """Feature: structure-ingestion-normalization, Property 8: Highest scorer wins

        For any structure, the primary chain SHALL be the protein chain with
        the highest computed score among representatives.

        **Validates: Requirements 4.1**
        """
        parsed, metadata = data.draw(st_multi_chain_structure())

        scope = score_chains(parsed, metadata)

        if not scope.primary_chain_ids:
            return  # Empty structure, nothing to verify

        # Manually compute scores for all protein chains
        from science.dtie.ingest.chain_scorer import (
            _build_entity_uniprot_map,
            _detect_duplicate_entities,
            _select_representatives,
        )

        entity_uniprot = _build_entity_uniprot_map(metadata)
        duplicate_eids = _detect_duplicate_entities(parsed.chains)

        all_scores = []
        for chain in parsed.chains:
            score = _score_single_chain(chain, entity_uniprot, duplicate_eids)
            all_scores.append(score)

        # Get representatives
        reps = _select_representatives(parsed.chains, all_scores, duplicate_eids)
        rep_labels = {r.auth_asym_id for r in reps}

        # Filter to protein representatives only
        protein_rep_scores = [
            s for s in reps
            if _is_protein_chain(
                next(c for c in parsed.chains if c.auth_asym_id == s.auth_asym_id)
            )
        ]

        if not protein_rep_scores:
            return  # No protein representatives, skip

        # The highest scoring protein representative should be primary
        best_protein_rep = max(protein_rep_scores, key=lambda s: s.score)
        primary_label = scope.primary_chain_ids[0]

        assert primary_label == best_protein_rep.auth_asym_id, (
            f"Primary chain is {primary_label} (score unknown) but highest "
            f"protein representative is {best_protein_rep.auth_asym_id} "
            f"(score={best_protein_rep.score:.2f})"
        )
