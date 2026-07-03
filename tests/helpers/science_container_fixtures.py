"""Shared fixtures for science-container integration gates (real DBAdapter)."""

from __future__ import annotations

import uuid
from typing import Any

from science.dtie.common.ingest_payloads import (
    ComputationScopePayload,
    ComputationScopeRecord,
    IngestDimensionPayload,
    StructureDimension,
    ChainDimension,
    ResidueDimension,
)
from science.dtie.common.keys import make_chain_id, make_residue_id, make_structure_id
from science.dtie.common.normalizer_payloads import (
    ProvenanceContext,
    RunType,
    SourceType,
)


def make_test_structure_id(pdb_id: str = "gate1") -> str:
    return make_structure_id(pdb_id=pdb_id.upper(), source="rcsb")


def _ingest_provenance(structure_id: str) -> ProvenanceContext:
    return ProvenanceContext(
        run_id=f"ingest_gate_{uuid.uuid4().hex[:12]}",
        structure_id=structure_id,
        model_version="integration-gate",
        pipeline_name="science_container_gate",
        run_type=RunType.ANALYSIS,
        source_type=SourceType.EMPIRICAL,
        code_version="integration-test",
    )


async def seed_ingest_foundation(
    db: Any,
    *,
    pdb_id: str = "gate1",
    residue_count: int = 5,
) -> str:
    """Persist dim_structure + residues + computation scope via Normalizer."""
    from data.normalizer.core import Normalizer

    structure_id = make_test_structure_id(pdb_id)
    chain_id = make_chain_id(structure_id, "A")
    prov = _ingest_provenance(structure_id)
    normalizer = Normalizer(db=db, caller_identity="science_container_gate")

    residues = [
        ResidueDimension(
            residue_id=make_residue_id(structure_id, "A", index),
            chain_id=chain_id,
            residue_index=index,
            residue_name="ALA",
            residue_name_3="ALA",
            comp_id="ALA",
        )
        for index in range(1, residue_count + 1)
    ]

    ingest_payload = IngestDimensionPayload(
        provenance=prov,
        structure=StructureDimension(
            structure_id=structure_id,
            pdb_id=pdb_id.upper(),
            method="X-RAY",
            resolution=2.0,
            source="rcsb",
            title=f"Gate test {pdb_id.upper()}",
            polymer_composition="protein",
        ),
        chains=[
            ChainDimension(
                chain_id=chain_id,
                structure_id=structure_id,
                auth_asym_id="A",
                label_asym_id="A",
                entity_id="1",
                entity_type="protein",
                sequence_length=residue_count,
            )
        ],
        residues=residues,
        atoms=[],
        file_hash="0" * 64,
        biotite_version="test",
        rcsbapi_version="test",
    )
    await normalizer.normalize_ingest_dimensions(ingest_payload)

    scope_payload = ComputationScopePayload(
        provenance=prov,
        scope=ComputationScopeRecord(
            structure_id=structure_id,
            primary_chain_ids=["A"],
            reference_chain="A",
            scope_source="integration_gate",
            selection_reason="science_container_integration",
        ),
    )
    await normalizer.normalize_computation_scope(scope_payload)
    return structure_id
