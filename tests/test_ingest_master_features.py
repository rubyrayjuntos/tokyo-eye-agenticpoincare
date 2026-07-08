"""Unit tests for MASTER feature ingest persistence helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("freesasa")

from science.dtie.common import ingest_master_features as imf
from science.dtie.common import residue_features as rf

FIXTURE_PDB = Path("science/dtie/assets/benchmark_pdbs/1CRN.pdb")


def test_attach_canonical_residue_ids() -> None:
    feats = rf.build_from_pdb_chain(FIXTURE_PDB, "A", mode=rf.FeatureMode.MASTER)
    sid = imf.structure_id_for_pdb("1CRN")
    attached = imf.attach_canonical_residue_ids(feats, sid)
    assert attached[0].residue_id == f"{sid}:A:{attached[0].residue_index}"


def test_compute_master_features_from_pdb() -> None:
    sid = imf.structure_id_for_pdb("1CRN")
    feats = imf.compute_master_features_from_pdb(FIXTURE_PDB, "A", sid)
    assert len(feats) >= 10
    assert all(f.residue_id for f in feats)
    assert all(f.sse_code in ("H", "E", "C") for f in feats)
    assert max(f.sasa for f in feats) > 1.0
