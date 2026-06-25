"""Property-based tests for the Normalizer ingestion path.

Feature: structure-ingestion-normalization
Properties: 2, 4, 15, 16
Validates: Requirements 1.8, 3.5, 9.4, 11.1, 11.2, 11.3, 11.5, 11.6

These tests validate:
- Idempotent re-ingest (Property 2)
- Key format rejection (Property 4)
- Source type classification (Property 15)
- Partial failure persistence (Property 16)
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from data.normalizer.core import Normalizer, NormalizerError
from science.dtie.common.ingest_payloads import (
    AlignmentPayload,
    AtomDimension,
    ChainDimension,
    CovalentBondFact,
    IngestDimensionPayload,
    ResidueAlignmentRecord,
    ResidueDimension,
    StructuralAlignmentRecord,
    StructureDimension,
)
from science.dtie.common.keys import (
    make_chain_id,
    make_residue_id,
    make_structure_id,
)
from science.dtie.common.normalizer_payloads import (
    NormalizerResult,
    ProvenanceContext,
    RunType,
    SourceType,
)


# ---------------------------------------------------------------------------
# Mock database for testing
# ---------------------------------------------------------------------------


class IngestMockDatabase:
    """In-memory mock that tracks all writes for assertion.

    Supports partial-failure simulation via fail_on_table.
    """

    def __init__(self, fail_on_table: str | None = None):
        self.tables: dict[str, list[dict[str, Any]]] = {}
        self._in_transaction = False
        self._fail_on_table = fail_on_table

    async def execute(self, query: str, params: dict[str, Any]) -> None:
        table = self._extract_table(query)
        if table and self._fail_on_table and table == self._fail_on_table:
            raise RuntimeError(f"Simulated failure on {table}")
        if table:
            # Simulate ON CONFLICT DO NOTHING for provenance_run
            if table == "provenance_run" and "DO NOTHING" in query:
                for row in self.tables.get("provenance_run", []):
                    if row.get("run_id") == params.get("run_id"):
                        return
            # Simulate ON CONFLICT DO UPDATE (upsert) for dim tables
            if "ON CONFLICT" in query and "DO UPDATE" in query:
                pk_field = self._get_pk_field(table)
                if pk_field and pk_field in params:
                    existing = self.tables.get(table, [])
                    for i, row in enumerate(existing):
                        if row.get(pk_field) == params[pk_field]:
                            existing[i] = params
                            return
            self.tables.setdefault(table, []).append(params)

    async def execute_many(self, query: str, params_list: list[dict[str, Any]]) -> None:
        table = self._extract_table(query)
        if table and self._fail_on_table and table == self._fail_on_table:
            raise RuntimeError(f"Simulated failure on {table}")
        if table:
            for params in params_list:
                # Simulate upsert: check for existing PK
                if "ON CONFLICT" in query and "DO UPDATE" in query:
                    pk_field = self._get_pk_field(table)
                    if pk_field and pk_field in params:
                        existing = self.tables.get(table, [])
                        replaced = False
                        for i, row in enumerate(existing):
                            if row.get(pk_field) == params[pk_field]:
                                existing[i] = params
                                replaced = True
                                break
                        if replaced:
                            continue
                self.tables.setdefault(table, []).append(params)

    async def fetch_one(self, query: str, params: dict[str, Any]) -> dict[str, Any] | None:
        if "provenance_run" in query and "run_id" in params:
            for row in self.tables.get("provenance_run", []):
                if row.get("run_id") == params["run_id"]:
                    return row
        return None

    async def begin(self) -> None:
        self._in_transaction = True

    async def commit(self) -> None:
        self._in_transaction = False

    async def rollback(self) -> None:
        self._in_transaction = False

    def _extract_table(self, query: str) -> str | None:
        q = query.strip().upper()
        if "INSERT INTO" in q:
            parts = q.split("INSERT INTO")[1].strip().split()
            if parts:
                return parts[0].lower()
        return None

    def _get_pk_field(self, table: str) -> str | None:
        pk_map = {
            "dim_structure": "structure_id",
            "dim_chain": "chain_id",
            "dim_residue": "residue_id",
            "dim_atom": "atom_id",
            "fact_covalent_bond": None,  # composite PK
            "fact_residue_alignment": None,
            "fact_structural_alignment": None,
        }
        return pk_map.get(table)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_PDB_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789"


@st.composite
def st_pdb_id(draw: st.DrawFn) -> str:
    """Generate a valid 4-character PDB ID."""
    first = draw(st.sampled_from(list("0123456789")))
    rest = draw(st.text(alphabet=_PDB_CHARS, min_size=3, max_size=3))
    return first + rest


@st.composite
def st_ingest_payload(draw: st.DrawFn) -> IngestDimensionPayload:
    """Generate a valid IngestDimensionPayload with realistic structure."""
    pdb_id = draw(st_pdb_id())
    structure_id = make_structure_id(pdb_id=pdb_id, source="rcsb")

    # Generate 1-3 chains
    n_chains = draw(st.integers(min_value=1, max_value=3))
    chain_labels = ["A", "B", "C"][:n_chains]

    chains = []
    residues = []
    atoms = []

    for chain_label in chain_labels:
        chain_id = make_chain_id(structure_id, chain_label)
        chains.append(ChainDimension(
            chain_id=chain_id,
            structure_id=structure_id,
            auth_asym_id=chain_label,
            label_asym_id=chain_label,
            entity_id="1",
            entity_type="protein",
            sequence_length=draw(st.integers(min_value=10, max_value=50)),
        ))

        # Generate 2-5 residues per chain
        n_residues = draw(st.integers(min_value=2, max_value=5))
        for res_idx in range(1, n_residues + 1):
            residue_id = make_residue_id(structure_id, chain_label, res_idx)
            b_factor = draw(st.floats(
                min_value=5.0, max_value=80.0,
                allow_nan=False, allow_infinity=False,
            ))
            residues.append(ResidueDimension(
                residue_id=residue_id,
                chain_id=chain_id,
                residue_index=res_idx,
                residue_name="A",
                residue_name_3="ALA",
                comp_id="ALA",
                is_resolved=True,
                max_b_factor=b_factor,
                low_confidence_coords=b_factor > 100.0,
            ))

            # Generate 1-3 atoms per residue
            n_atoms = draw(st.integers(min_value=1, max_value=3))
            atom_names = ["N", "CA", "C"][:n_atoms]
            for atom_name in atom_names:
                atom_id = f"{residue_id}:{atom_name}"
                atoms.append(AtomDimension(
                    atom_id=atom_id,
                    residue_id=residue_id,
                    atom_name=atom_name,
                    element="C" if atom_name != "N" else "N",
                    x=draw(st.floats(-50, 50, allow_nan=False, allow_infinity=False)),
                    y=draw(st.floats(-50, 50, allow_nan=False, allow_infinity=False)),
                    z=draw(st.floats(-50, 50, allow_nan=False, allow_infinity=False)),
                    occupancy=1.0,
                    b_factor=b_factor,
                ))

    run_id = f"run_ingest_{pdb_id}_{draw(st.integers(1, 9999))}"

    return IngestDimensionPayload(
        provenance=ProvenanceContext(
            run_id=run_id,
            structure_id=structure_id,
            model_version="ingest_v1",
            pipeline_name="structure_ingest",
            run_type=RunType.ANALYSIS,
            source_type=SourceType.EMPIRICAL,
        ),
        structure=StructureDimension(
            structure_id=structure_id,
            pdb_id=pdb_id,
            source="rcsb",
            method="X-RAY DIFFRACTION",
            resolution=draw(st.floats(1.0, 4.0, allow_nan=False, allow_infinity=False)),
            title=f"Structure of {pdb_id}",
            polymer_composition="protein",
        ),
        chains=chains,
        residues=residues,
        atoms=atoms,
        covalent_bonds=[],
        file_hash="a" * 64,
        biotite_version="1.0.0",
        rcsbapi_version="0.1.0",
    )


@st.composite
def st_alignment_payload(draw: st.DrawFn) -> AlignmentPayload:
    """Generate a valid AlignmentPayload."""
    pdb_id = draw(st_pdb_id())
    structure_id = make_structure_id(pdb_id=pdb_id, source="rcsb")
    chain_label = "A"

    # Generate 2-5 residue alignments
    n_residues = draw(st.integers(min_value=2, max_value=5))
    residue_alignments = []
    for res_idx in range(1, n_residues + 1):
        residue_id = make_residue_id(structure_id, chain_label, res_idx)
        has_mapping = draw(st.booleans())
        residue_alignments.append(ResidueAlignmentRecord(
            residue_id=residue_id,
            uniprot_accession="P12345",
            uniprot_position=res_idx if has_mapping else None,
            mapping_source="sifts",
            mapping_confidence=draw(st.floats(0.5, 1.0, allow_nan=False, allow_infinity=False)),
            reason_code=None if has_mapping else "tag",
        ))

    run_id = f"run_align_{pdb_id}_{draw(st.integers(1, 9999))}"

    return AlignmentPayload(
        provenance=ProvenanceContext(
            run_id=run_id,
            structure_id=structure_id,
            model_version="alignment_v1",
            pipeline_name="alignment_sidecar",
            run_type=RunType.ANALYSIS,
            source_type=SourceType.DETERMINISTIC,
        ),
        structure_id=structure_id,
        residue_alignments=residue_alignments,
    )


# ---------------------------------------------------------------------------
# Property 2: Idempotent re-ingest
# Feature: structure-ingestion-normalization, Property 2
# Validates: Requirements 1.8, 9.4
# ---------------------------------------------------------------------------


class TestProperty2IdempotentReingest:
    """Property 2: Idempotent re-ingest.

    For any structure_id, calling normalize_ingest_dimensions twice with the
    same payload SHALL produce identical dimension row counts and identical
    canonical keys. The second call SHALL NOT create duplicate rows.
    """

    @settings(max_examples=100)
    @given(payload=st_ingest_payload())
    @pytest.mark.asyncio
    async def test_double_ingest_no_duplicates(self, payload: IngestDimensionPayload):
        """Feature: structure-ingestion-normalization, Property 2: Idempotent re-ingest

        For any valid IngestDimensionPayload, ingesting the same payload twice
        SHALL produce the same row counts (no duplicates).

        **Validates: Requirements 1.8, 9.4**
        """
        db = IngestMockDatabase()
        normalizer = Normalizer(db=db)  # type: ignore[arg-type]

        # First ingest
        result1 = await normalizer.normalize_ingest_dimensions(payload)
        assert result1.success is True

        # Capture row counts after first ingest
        structure_count_1 = len(db.tables.get("dim_structure", []))
        chain_count_1 = len(db.tables.get("dim_chain", []))
        residue_count_1 = len(db.tables.get("dim_residue", []))
        atom_count_1 = len(db.tables.get("dim_atom", []))

        # Second ingest (same payload)
        result2 = await normalizer.normalize_ingest_dimensions(payload)
        assert result2.success is True

        # Row counts must be identical (upsert semantics, no duplicates)
        structure_count_2 = len(db.tables.get("dim_structure", []))
        chain_count_2 = len(db.tables.get("dim_chain", []))
        residue_count_2 = len(db.tables.get("dim_residue", []))
        atom_count_2 = len(db.tables.get("dim_atom", []))

        assert structure_count_2 == structure_count_1, (
            f"dim_structure grew from {structure_count_1} to {structure_count_2} on re-ingest"
        )
        assert chain_count_2 == chain_count_1, (
            f"dim_chain grew from {chain_count_1} to {chain_count_2} on re-ingest"
        )
        assert residue_count_2 == residue_count_1, (
            f"dim_residue grew from {residue_count_1} to {residue_count_2} on re-ingest"
        )
        assert atom_count_2 == atom_count_1, (
            f"dim_atom grew from {atom_count_1} to {atom_count_2} on re-ingest"
        )


# ---------------------------------------------------------------------------
# Property 4: Key format rejection
# Feature: structure-ingestion-normalization, Property 4
# Validates: Requirements 3.5, 11.6
# ---------------------------------------------------------------------------


class TestProperty4KeyFormatRejection:
    """Property 4: Key format rejection.

    For any residue_id that does not match the canonical format regex,
    the Normalizer SHALL reject the payload and raise NormalizerError.
    """

    @settings(max_examples=100)
    @given(data=st.data())
    @pytest.mark.asyncio
    async def test_invalid_residue_id_rejected(self, data: st.DataObject):
        """Feature: structure-ingestion-normalization, Property 4: Key format rejection

        For any residue_id that does not match the canonical format
        (structure:chain:index[:insertion]), the system SHALL reject the
        payload — either at Pydantic validation (payload construction) or
        at the Normalizer level (NormalizerError).

        **Validates: Requirements 3.5, 11.6**
        """
        # Generate invalid residue_id patterns
        invalid_id = data.draw(st.sampled_from([
            "no_colons_here",
            "too:many:colons:here:extra:stuff",
            ":leading_colon:A:1",
            "struct:A:",  # missing index
            "struct:A:abc",  # non-numeric index
            "struct::1",  # empty chain
            "",  # empty string
            "has spaces:A:1",
            "struct:A:1:AB",  # insertion code too long
        ]))

        pdb_id = data.draw(st_pdb_id())
        structure_id = make_structure_id(pdb_id=pdb_id, source="rcsb")
        chain_id = make_chain_id(structure_id, "A")

        db = IngestMockDatabase()
        normalizer = Normalizer(db=db)  # type: ignore[arg-type]

        # The system should reject invalid residue_ids at some layer:
        # Pydantic validation raises ValueError (ValidationError inherits ValueError),
        # or the Normalizer raises NormalizerError if bypassed
        with pytest.raises((NormalizerError, ValueError)):
            payload = IngestDimensionPayload(
                provenance=ProvenanceContext(
                    run_id=f"run_bad_{pdb_id}",
                    structure_id=structure_id,
                    model_version="test",
                    pipeline_name="test",
                    run_type=RunType.ANALYSIS,
                    source_type=SourceType.EMPIRICAL,
                ),
                structure=StructureDimension(
                    structure_id=structure_id,
                    pdb_id=pdb_id,
                    source="rcsb",
                    method="X-RAY DIFFRACTION",
                    title="test",
                    polymer_composition="protein",
                ),
                chains=[ChainDimension(
                    chain_id=chain_id,
                    structure_id=structure_id,
                    auth_asym_id="A",
                    label_asym_id="A",
                    entity_id="1",
                    entity_type="protein",
                    sequence_length=10,
                )],
                residues=[ResidueDimension(
                    residue_id=invalid_id,
                    chain_id=chain_id,
                    residue_index=1,
                    residue_name="A",
                    residue_name_3="ALA",
                    comp_id="ALA",
                )],
                atoms=[],
                file_hash="b" * 64,
                biotite_version="1.0.0",
                rcsbapi_version="0.1.0",
            )
            await normalizer.normalize_ingest_dimensions(payload)


# ---------------------------------------------------------------------------
# Property 15: Source type classification
# Feature: structure-ingestion-normalization, Property 15
# Validates: Requirements 11.1, 11.2, 11.3
# ---------------------------------------------------------------------------


class TestProperty15SourceTypeClassification:
    """Property 15: Source type classification.

    For any dimension write, provenance source_type SHALL be 'empirical'.
    For any alignment write, provenance source_type SHALL be 'deterministic'.
    """

    @settings(max_examples=100)
    @given(payload=st_ingest_payload())
    @pytest.mark.asyncio
    async def test_ingest_dimensions_use_empirical(self, payload: IngestDimensionPayload):
        """Feature: structure-ingestion-normalization, Property 15: Source type classification

        For any dimension write (dim_structure, dim_chain, dim_residue, dim_atom),
        provenance source_type SHALL be 'empirical'.

        **Validates: Requirements 11.1, 11.3**
        """
        db = IngestMockDatabase()
        normalizer = Normalizer(db=db)  # type: ignore[arg-type]

        result = await normalizer.normalize_ingest_dimensions(payload)
        assert result.success is True

        # Check provenance_run was created with source_type='empirical'
        prov_rows = db.tables.get("provenance_run", [])
        assert len(prov_rows) >= 1
        prov_row = prov_rows[0]
        assert prov_row["source_type"] == "empirical", (
            f"Expected source_type='empirical', got '{prov_row['source_type']}'"
        )

    @settings(max_examples=100)
    @given(payload=st_alignment_payload())
    @pytest.mark.asyncio
    async def test_alignment_uses_deterministic(self, payload: AlignmentPayload):
        """Feature: structure-ingestion-normalization, Property 15: Source type classification

        For any alignment write (fact_residue_alignment, fact_structural_alignment),
        provenance source_type SHALL be 'deterministic'.

        **Validates: Requirements 11.2**
        """
        db = IngestMockDatabase()
        normalizer = Normalizer(db=db)  # type: ignore[arg-type]

        result = await normalizer.normalize_alignment(payload)
        assert result.success is True

        # Check provenance_run was created with source_type='deterministic'
        prov_rows = db.tables.get("provenance_run", [])
        assert len(prov_rows) >= 1
        prov_row = prov_rows[0]
        assert prov_row["source_type"] == "deterministic", (
            f"Expected source_type='deterministic', got '{prov_row['source_type']}'"
        )


# ---------------------------------------------------------------------------
# Property 16: Partial failure persistence
# Feature: structure-ingestion-normalization, Property 16
# Validates: Requirements 11.5
# ---------------------------------------------------------------------------


class TestProperty16PartialFailurePersistence:
    """Property 16: Partial failure persistence.

    For any ingestion where a later stage (e.g., dim_atom) fails, all data
    from earlier successful stages (dim_structure, dim_chain, dim_residue)
    SHALL remain persisted in the database.
    """

    @settings(max_examples=100)
    @given(payload=st_ingest_payload())
    @pytest.mark.asyncio
    async def test_atom_failure_preserves_earlier_stages(
        self, payload: IngestDimensionPayload
    ):
        """Feature: structure-ingestion-normalization, Property 16: Partial failure persistence

        For any ingestion where dim_atom write fails, dim_structure, dim_chain,
        and dim_residue data SHALL remain persisted.

        **Validates: Requirements 11.5**
        """
        # Configure mock to fail on dim_atom
        db = IngestMockDatabase(fail_on_table="dim_atom")
        normalizer = Normalizer(db=db)  # type: ignore[arg-type]

        result = await normalizer.normalize_ingest_dimensions(payload)

        # Should still succeed (partial failure semantics)
        assert result.success is True

        # Earlier stages should have data persisted
        assert len(db.tables.get("dim_structure", [])) == 1, (
            "dim_structure should be persisted despite dim_atom failure"
        )
        assert len(db.tables.get("dim_chain", [])) == len(payload.chains), (
            "dim_chain should be persisted despite dim_atom failure"
        )
        assert len(db.tables.get("dim_residue", [])) == len(payload.residues), (
            "dim_residue should be persisted despite dim_atom failure"
        )

        # dim_atom should have no data (it failed)
        assert len(db.tables.get("dim_atom", [])) == 0, (
            "dim_atom should be empty after failure"
        )

        # Warnings should mention the failure
        assert any("dim_atom" in w for w in result.warnings), (
            f"Expected warning about dim_atom failure, got: {result.warnings}"
        )

    @settings(max_examples=100)
    @given(payload=st_ingest_payload())
    @pytest.mark.asyncio
    async def test_bond_failure_preserves_dimensions(
        self, payload: IngestDimensionPayload
    ):
        """Feature: structure-ingestion-normalization, Property 16: Bond failure preserves dims

        For any ingestion where fact_covalent_bond write fails, all dimension
        data SHALL remain persisted.

        **Validates: Requirements 11.5**
        """
        # Add a covalent bond to the payload so the failure path is exercised
        if len(payload.residues) >= 2:
            payload = payload.model_copy(update={
                "covalent_bonds": [CovalentBondFact(
                    residue_id_1=payload.residues[0].residue_id,
                    residue_id_2=payload.residues[1].residue_id,
                    atom_name_1="SG",
                    atom_name_2="SG",
                    bond_type="disulf",
                )]
            })

        # Configure mock to fail on fact_covalent_bond
        db = IngestMockDatabase(fail_on_table="fact_covalent_bond")
        normalizer = Normalizer(db=db)  # type: ignore[arg-type]

        result = await normalizer.normalize_ingest_dimensions(payload)

        # Should still succeed (partial failure semantics)
        assert result.success is True

        # All dimensions should be persisted
        assert len(db.tables.get("dim_structure", [])) == 1
        assert len(db.tables.get("dim_chain", [])) == len(payload.chains)
        assert len(db.tables.get("dim_residue", [])) == len(payload.residues)
        assert len(db.tables.get("dim_atom", [])) == len(payload.atoms)
