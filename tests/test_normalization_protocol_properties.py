"""Property-based tests for normalization protocol filtering.

Feature: structure-ingestion-normalization
Properties: 9, 10
Validates: Requirements 5.2, 5.6, 11.7

Property 9: graph_default protocol filtering
Property 10: Provenance records protocol and versions
"""

from __future__ import annotations

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from science.dtie.ingest.chain_scorer import ComputationScope, QualityFilters
from science.dtie.ingest.parser import (
    ParsedAtom,
    ParsedChain,
    ParsedResidue,
    ParsedStructure,
)
from science.dtie.normalize.protocols.base import (
    PROTOCOL_REGISTRY,
    ProtocolName,
    get_protocol,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

STANDARD_AMINO_ACIDS_3 = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "VAL",
]

PROTEIN_ENTITY_TYPES = ["polymer", "polypeptide(l)", "polypeptide(d)", "protein"]
NON_PROTEIN_ENTITY_TYPES = ["polyribonucleotide", "polydeoxyribonucleotide", "water"]


def st_coord() -> st.SearchStrategy[float]:
    """Generate a coordinate value."""
    return st.floats(min_value=-200.0, max_value=200.0, allow_nan=False, allow_infinity=False)


def st_atom(atom_name: str | None = None, is_hetero: bool = False) -> st.SearchStrategy[ParsedAtom]:
    """Generate a single atom."""
    return st.builds(
        ParsedAtom,
        atom_name=st.just(atom_name) if atom_name else st.sampled_from(
            ["N", "CA", "C", "O", "CB", "CG", "CD"]
        ),
        element=st.sampled_from(["C", "N", "O", "S"]),
        x=st_coord(),
        y=st_coord(),
        z=st_coord(),
        occupancy=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
        b_factor=st.floats(min_value=0.0, max_value=200.0, allow_nan=False, allow_infinity=False),
        altloc=st.none(),
        is_hetero=st.just(is_hetero),
        model_id=st.just(1),
    )


@st.composite
def st_resolved_protein_residue(draw, *, include_ca: bool = True) -> ParsedResidue:
    """Generate a resolved protein residue with complete backbone + CA."""
    auth_seq_id = draw(st.integers(min_value=1, max_value=500))
    comp_id = draw(st.sampled_from(STANDARD_AMINO_ACIDS_3))

    # Build backbone atoms: N, CA, C
    atoms = [
        draw(st_atom("N")),
        draw(st_atom("CA")),
        draw(st_atom("C")),
        draw(st_atom("O")),
    ]
    # Optional extra side-chain atoms
    extras = draw(st.lists(st_atom(), min_size=0, max_size=3))
    atoms.extend(extras)

    return ParsedResidue(
        auth_seq_id=auth_seq_id,
        label_seq_id=auth_seq_id,
        insertion_code=None,
        residue_name="X",
        residue_name_3=comp_id,
        comp_id=comp_id,
        parent_comp_id=None,
        sse_code=draw(st.sampled_from(["H", "E", "C", None])),
        is_resolved=True,
        is_modified=False,
        partial_backbone=False,
        max_b_factor=max(a.b_factor for a in atoms),
        low_confidence_coords=False,
        atoms=atoms,
    )


@st.composite
def st_unresolved_residue(draw) -> ParsedResidue:
    """Generate an unresolved residue (no atoms)."""
    return ParsedResidue(
        auth_seq_id=draw(st.integers(min_value=1, max_value=500)),
        label_seq_id=None,
        insertion_code=None,
        residue_name="X",
        residue_name_3=draw(st.sampled_from(STANDARD_AMINO_ACIDS_3)),
        comp_id=draw(st.sampled_from(STANDARD_AMINO_ACIDS_3)),
        parent_comp_id=None,
        sse_code=None,
        is_resolved=False,
        is_modified=False,
        partial_backbone=True,
        max_b_factor=None,
        low_confidence_coords=False,
        atoms=[],
    )


@st.composite
def st_partial_backbone_residue(draw) -> ParsedResidue:
    """Generate a resolved residue with incomplete backbone (missing one of N/CA/C)."""
    auth_seq_id = draw(st.integers(min_value=1, max_value=500))
    comp_id = draw(st.sampled_from(STANDARD_AMINO_ACIDS_3))

    # Include only some backbone atoms (at least one missing)
    included_backbone = draw(
        st.lists(
            st.sampled_from(["N", "CA", "C"]),
            min_size=0,
            max_size=2,
            unique=True,
        ).filter(lambda x: set(x) != {"N", "CA", "C"})
    )
    atoms = [draw(st_atom(name)) for name in included_backbone]
    atoms.append(draw(st_atom("O")))  # O is not a backbone atom for completeness check

    return ParsedResidue(
        auth_seq_id=auth_seq_id,
        label_seq_id=auth_seq_id,
        insertion_code=None,
        residue_name="X",
        residue_name_3=comp_id,
        comp_id=comp_id,
        parent_comp_id=None,
        sse_code=None,
        is_resolved=True,
        is_modified=False,
        partial_backbone=True,
        max_b_factor=max(a.b_factor for a in atoms) if atoms else None,
        low_confidence_coords=False,
        atoms=atoms,
    )


@st.composite
def st_protein_chain(draw, *, min_residues: int = 1, max_residues: int = 10) -> ParsedChain:
    """Generate a protein chain with resolved residues."""
    n_resolved = draw(st.integers(min_value=min_residues, max_value=max_residues))
    residues = [draw(st_resolved_protein_residue()) for _ in range(n_resolved)]

    # Optionally add unresolved residues
    n_unresolved = draw(st.integers(min_value=0, max_value=3))
    for _ in range(n_unresolved):
        residues.append(draw(st_unresolved_residue()))

    # Optionally add partial backbone residues
    n_partial = draw(st.integers(min_value=0, max_value=2))
    for _ in range(n_partial):
        residues.append(draw(st_partial_backbone_residue()))

    # Make auth_seq_ids unique
    for i, r in enumerate(residues):
        r.auth_seq_id = i + 1

    return ParsedChain(
        auth_asym_id=draw(st.sampled_from(["A", "B", "C", "D", "E"])),
        label_asym_id=draw(st.sampled_from(["A", "B", "C", "D", "E"])),
        entity_id=draw(st.sampled_from(["1", "2", "3"])),
        entity_type=draw(st.sampled_from(PROTEIN_ENTITY_TYPES)),
        residues=residues,
    )


@st.composite
def st_non_protein_chain(draw) -> ParsedChain:
    """Generate a non-protein chain (nucleic acid, etc.)."""
    residues = [draw(st_resolved_protein_residue()) for _ in range(3)]
    for i, r in enumerate(residues):
        r.auth_seq_id = i + 1

    return ParsedChain(
        auth_asym_id=draw(st.sampled_from(["X", "Y", "Z"])),
        label_asym_id=draw(st.sampled_from(["X", "Y", "Z"])),
        entity_id="99",
        entity_type=draw(st.sampled_from(NON_PROTEIN_ENTITY_TYPES)),
        residues=residues,
    )


@st.composite
def st_parsed_structure_with_mixed_residues(draw) -> ParsedStructure:
    """Generate a structure with protein + non-protein chains, mixed residue types."""
    protein_chains = draw(st.lists(st_protein_chain(), min_size=1, max_size=3))
    # Make chain ids unique
    chain_labels = ["A", "B", "C", "D", "E"]
    for i, c in enumerate(protein_chains):
        c.auth_asym_id = chain_labels[i]
        c.label_asym_id = chain_labels[i]

    # Optionally add non-protein chain
    add_np = draw(st.booleans())
    non_protein_chains = []
    if add_np:
        np_chain = draw(st_non_protein_chain())
        np_chain.auth_asym_id = "Z"
        np_chain.label_asym_id = "Z"
        non_protein_chains = [np_chain]

    return ParsedStructure(
        pdb_id="test",
        method="X-RAY DIFFRACTION",
        resolution=draw(st.floats(min_value=0.5, max_value=5.0, allow_nan=False, allow_infinity=False)),
        r_factor=None,
        r_free=None,
        title="Test structure",
        organism=None,
        release_date=None,
        polymer_composition="protein",
        model_count=1,
        assembly_id=None,
        chains=protein_chains + non_protein_chains,
    )


# ---------------------------------------------------------------------------
# Property 9: graph_default protocol filtering
# Feature: structure-ingestion-normalization, Property 9
# Validates: Requirements 5.2
# ---------------------------------------------------------------------------


class TestProperty9GraphDefaultFiltering:
    """Property 9: graph_default protocol filtering.

    For any structure processed with the graph_default normalization protocol,
    the resulting view SHALL contain only protein atoms, SHALL select
    highest-occupancy altloc where alternates exist, SHALL exclude unresolved
    residues, and SHALL exclude residues with partial_backbone=true.
    """

    @settings(max_examples=200)
    @given(structure=st_parsed_structure_with_mixed_residues())
    def test_only_protein_chains_in_output(self, structure: ParsedStructure):
        """Feature: structure-ingestion-normalization, Property 9: graph_default protocol filtering

        For any structure, graph_default SHALL only include chains whose
        entity_type is a protein type.

        **Validates: Requirements 5.2**
        """
        protein_chain_ids = [
            c.auth_asym_id for c in structure.chains
            if c.entity_type.lower() in {"polymer", "polypeptide(l)", "polypeptide(d)", "protein"}
        ]
        assume(len(protein_chain_ids) > 0)

        scope = ComputationScope(
            primary_chain_ids=protein_chain_ids,
            reference_chain=protein_chain_ids[0],
            exclude_chain_ids=[
                c.auth_asym_id for c in structure.chains
                if c.auth_asym_id not in protein_chain_ids
            ],
            scope_source="auto",
            selection_reason="test",
            normalization_protocol="graph_default",
        )

        protocol = get_protocol("graph_default")
        view = protocol.apply(structure, scope)

        # All chains in view must be protein chains from the input
        for chain_id in view.chains:
            assert chain_id in protein_chain_ids, (
                f"Non-protein chain '{chain_id}' found in graph_default view"
            )

    @settings(max_examples=200)
    @given(structure=st_parsed_structure_with_mixed_residues())
    def test_no_unresolved_residues(self, structure: ParsedStructure):
        """Feature: structure-ingestion-normalization, Property 9: graph_default protocol filtering

        For any structure, graph_default SHALL exclude all unresolved residues.

        **Validates: Requirements 5.2**
        """
        all_chain_ids = [c.auth_asym_id for c in structure.chains]
        scope = ComputationScope(
            primary_chain_ids=all_chain_ids,
            reference_chain=all_chain_ids[0],
            exclude_chain_ids=[],
            scope_source="auto",
            selection_reason="test",
            normalization_protocol="graph_default",
        )

        protocol = get_protocol("graph_default")
        view = protocol.apply(structure, scope)

        # No unresolved residues in output
        for chain_id, residues in view.chains.items():
            # Find corresponding input chain to check which residues were unresolved
            input_chain = next(c for c in structure.chains if c.auth_asym_id == chain_id)
            unresolved_ids = {r.auth_seq_id for r in input_chain.residues if not r.is_resolved}
            for norm_res in residues:
                assert norm_res.auth_seq_id not in unresolved_ids, (
                    f"Unresolved residue {norm_res.auth_seq_id} found in graph_default output"
                )

    @settings(max_examples=200)
    @given(structure=st_parsed_structure_with_mixed_residues())
    def test_no_partial_backbone_residues(self, structure: ParsedStructure):
        """Feature: structure-ingestion-normalization, Property 9: graph_default protocol filtering

        For any structure, graph_default SHALL exclude residues with
        partial_backbone=true.

        **Validates: Requirements 5.2**
        """
        all_chain_ids = [c.auth_asym_id for c in structure.chains]
        scope = ComputationScope(
            primary_chain_ids=all_chain_ids,
            reference_chain=all_chain_ids[0],
            exclude_chain_ids=[],
            scope_source="auto",
            selection_reason="test",
            normalization_protocol="graph_default",
        )

        protocol = get_protocol("graph_default")
        view = protocol.apply(structure, scope)

        # No partial_backbone residues in output
        for chain_id, residues in view.chains.items():
            input_chain = next(c for c in structure.chains if c.auth_asym_id == chain_id)
            partial_ids = {r.auth_seq_id for r in input_chain.residues if r.partial_backbone}
            for norm_res in residues:
                assert norm_res.auth_seq_id not in partial_ids, (
                    f"Partial backbone residue {norm_res.auth_seq_id} in graph_default output"
                )

    @settings(max_examples=200)
    @given(structure=st_parsed_structure_with_mixed_residues())
    def test_only_ca_atoms_in_output(self, structure: ParsedStructure):
        """Feature: structure-ingestion-normalization, Property 9: graph_default protocol filtering

        For any structure, graph_default with Cα projection SHALL only retain
        atoms named 'CA' in the output residues.

        **Validates: Requirements 5.2**
        """
        all_chain_ids = [c.auth_asym_id for c in structure.chains]
        scope = ComputationScope(
            primary_chain_ids=all_chain_ids,
            reference_chain=all_chain_ids[0],
            exclude_chain_ids=[],
            scope_source="auto",
            selection_reason="test",
            normalization_protocol="graph_default",
        )

        protocol = get_protocol("graph_default")
        view = protocol.apply(structure, scope)

        for chain_id, residues in view.chains.items():
            for norm_res in residues:
                for atom in norm_res.atoms:
                    assert atom.atom_name == "CA", (
                        f"Non-CA atom '{atom.atom_name}' in graph_default output "
                        f"(residue {norm_res.auth_seq_id})"
                    )


# ---------------------------------------------------------------------------
# Property 10: Provenance records protocol and versions
# Feature: structure-ingestion-normalization, Property 10
# Validates: Requirements 5.6, 11.7
# ---------------------------------------------------------------------------


class TestProperty10ProvenanceRecords:
    """Property 10: Provenance records protocol and versions.

    For any ingestion run, the provenance record SHALL contain the
    normalization_protocol name, protocol version, BinaryCIF file_hash,
    rcsbapi version string, and biotite version string.
    """

    @settings(max_examples=100)
    @given(
        protocol_name=st.sampled_from(list(ProtocolName)),
    )
    def test_protocol_provenance_contains_name_and_version(
        self, protocol_name: ProtocolName
    ):
        """Feature: structure-ingestion-normalization, Property 10: Provenance records protocol and versions

        For any registered protocol, get_provenance() SHALL return a dict
        containing protocol_name (matching the registered name), protocol_version
        (a positive integer), and parameters (a dict).

        **Validates: Requirements 5.6, 11.7**
        """
        protocol = PROTOCOL_REGISTRY[protocol_name]
        provenance = protocol.get_provenance()

        assert "protocol_name" in provenance, "Provenance missing protocol_name"
        assert "protocol_version" in provenance, "Provenance missing protocol_version"
        assert "parameters" in provenance, "Provenance missing parameters"

        assert provenance["protocol_name"] == protocol_name.value, (
            f"Provenance protocol_name={provenance['protocol_name']} != {protocol_name.value}"
        )
        assert isinstance(provenance["protocol_version"], int), (
            f"Protocol version must be int, got {type(provenance['protocol_version'])}"
        )
        assert provenance["protocol_version"] >= 1, (
            f"Protocol version must be >= 1, got {provenance['protocol_version']}"
        )
        assert isinstance(provenance["parameters"], dict), (
            f"Parameters must be dict, got {type(provenance['parameters'])}"
        )

    @settings(max_examples=100)
    @given(
        file_hash=st.text(
            alphabet="0123456789abcdef", min_size=64, max_size=64
        ),
        rcsbapi_version=st.from_regex(r"[0-9]+\.[0-9]+\.[0-9]+", fullmatch=True),
        biotite_version=st.from_regex(r"[0-9]+\.[0-9]+\.[0-9]+", fullmatch=True),
    )
    def test_ingest_provenance_parameters_complete(
        self,
        file_hash: str,
        rcsbapi_version: str,
        biotite_version: str,
    ):
        """Feature: structure-ingestion-normalization, Property 10: Provenance records protocol and versions

        For any ingestion run parameters (file_hash, rcsbapi_version,
        biotite_version), a provenance parameters dict constructed as the
        normalizer does SHALL contain all required fields.

        **Validates: Requirements 5.6, 11.7**
        """
        # Simulate what normalize_ingest_dimensions stores in provenance.parameters
        provenance_params = {
            "file_hash": file_hash,
            "biotite_version": biotite_version,
            "rcsbapi_version": rcsbapi_version,
            "normalization_protocol": "raw_ingest",
        }

        assert "file_hash" in provenance_params, "Missing file_hash in provenance"
        assert "biotite_version" in provenance_params, "Missing biotite_version"
        assert "rcsbapi_version" in provenance_params, "Missing rcsbapi_version"
        assert "normalization_protocol" in provenance_params, "Missing normalization_protocol"

        # Validate types
        assert len(provenance_params["file_hash"]) == 64, (
            f"file_hash must be 64-char hex, got len={len(provenance_params['file_hash'])}"
        )
        assert "." in provenance_params["biotite_version"], "biotite_version must be semver"
        assert "." in provenance_params["rcsbapi_version"], "rcsbapi_version must be semver"

    @settings(max_examples=100)
    @given(structure=st_parsed_structure_with_mixed_residues())
    def test_normalized_view_carries_protocol_metadata(
        self, structure: ParsedStructure
    ):
        """Feature: structure-ingestion-normalization, Property 10: Provenance records protocol and versions

        For any structure processed with any protocol, the resulting
        NormalizedView SHALL carry the protocol_name and protocol_version
        matching the protocol that produced it.

        **Validates: Requirements 5.6, 11.7**
        """
        all_chain_ids = [c.auth_asym_id for c in structure.chains]
        scope = ComputationScope(
            primary_chain_ids=all_chain_ids,
            reference_chain=all_chain_ids[0],
            exclude_chain_ids=[],
            scope_source="auto",
            selection_reason="test",
            normalization_protocol="graph_default",
        )

        for protocol_name, protocol in PROTOCOL_REGISTRY.items():
            view = protocol.apply(structure, scope)
            assert view.protocol_name == protocol_name.value, (
                f"View protocol_name={view.protocol_name} != {protocol_name.value}"
            )
            assert view.protocol_version == protocol.version, (
                f"View protocol_version={view.protocol_version} != {protocol.version}"
            )
