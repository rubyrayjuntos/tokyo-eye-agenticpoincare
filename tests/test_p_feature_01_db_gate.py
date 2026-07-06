"""P_FEATURE_01 DB round-trip gate — unit + integration tests.

Unit tests exercise evaluate_round_trip without DB (two-sided at DB layer).
Integration test runs full gate against live post-ingest corpus (skipped without DB).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("freesasa")

from science.dtie.common import residue_features as rf
from science.training import p_feature_01_gate as gate

FIXTURE_PDB = Path("science/dtie/assets/benchmark_pdbs/1CRN.pdb")
TIM_PDB = Path("science/dtie/assets/benchmark_pdbs/1TIM.pdb")


def _fresh_1crn() -> list[rf.ResidueNodeFeatures]:
    return rf.build_from_pdb_chain(FIXTURE_PDB, "A", mode=rf.FeatureMode.MASTER)


def test_round_trip_passes_when_all_three_match() -> None:
    fresh = _fresh_1crn()
    result = gate.evaluate_round_trip(fresh, fresh, fresh, pdb_id="1CRN")
    assert result.fresh_matches_written
    assert result.written_matches_train
    assert result.passed


def test_db_gate_fails_when_written_skewed_from_fresh() -> None:
    """NEGATIVE: wrong persisted rows must fail fresh≈written (all-coil class)."""
    fresh = _fresh_1crn()
    all_coil = [
        replace(f, ss_type=1.0, sse_code="C", sasa=f.sasa * 0.5 + 0.1)
        for f in fresh
    ]
    result = gate.evaluate_round_trip(fresh, all_coil, all_coil, pdb_id="1CRN")
    assert not result.fresh_matches_written
    assert not result.passed
    assert any("ss_type" in m or "sasa" in m for m in result.fresh_written_mismatches)


def test_db_gate_fails_when_train_read_skewed() -> None:
    """NEGATIVE: corrupt train read path must fail written≈train even if ingest OK."""
    fresh = _fresh_1crn()
    written = list(fresh)
    train_skewed = [
        replace(f, rho=f.rho + 3.0) if i == 0 else f for i, f in enumerate(written)
    ]
    result = gate.evaluate_round_trip(fresh, written, train_skewed, pdb_id="1CRN")
    assert result.fresh_matches_written
    assert not result.written_matches_train
    assert not result.passed
    assert result.written_train_mismatches


def test_ss_anchor_catches_all_coil_on_tim() -> None:
    if not TIM_PDB.is_file():
        pytest.skip("1TIM fixture missing")
    fresh = rf.build_from_pdb_chain(TIM_PDB, "A", mode=rf.FeatureMode.MASTER)
    all_coil = [replace(f, ss_type=1.0, sse_code="C") for f in fresh]
    errors = gate.check_ss_anchors("1TIM", all_coil)
    assert errors, "all-coil TIM must fail SS anchor"


def test_ss_anchor_tim_barrel_real_features() -> None:
    if not TIM_PDB.is_file():
        pytest.skip("1TIM fixture missing")
    fresh = rf.build_from_pdb_chain(TIM_PDB, "A", mode=rf.FeatureMode.MASTER)
    errors = gate.check_ss_anchors("1TIM", fresh)
    assert not errors, errors
    dist = gate.sse_distribution(fresh)
    assert dist["H"] > 0 and dist["E"] > 0


def test_stamp_validation_rejects_stale_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "corpus.json"
    manifest.write_text('{"proteins": []}\n')
    stamp = {
        "p_feature_01_passed": True,
        "manifest_sha256": "deadbeef",
        "feature_module_sha256": gate.feature_module_sha256(),
    }
    with pytest.raises(RuntimeError, match="manifest hash mismatch"):
        gate.validate_gate_stamp(stamp, manifest)


def test_require_gate_refuses_missing_stamp(tmp_path: Path) -> None:
    manifest = tmp_path / "corpus.json"
    manifest.write_text('{"proteins": []}\n')
    with pytest.raises(FileNotFoundError):
        gate.require_p_feature_01_for_training(
            manifest,
            stamp_path=tmp_path / "missing.json",
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_p_feature_01_live_corpus_round_trip(integration_db) -> None:
    """Full round-trip on enabled 12-corpus structures (requires post-ingest DB)."""
    from data.db import DBAdapter
    from experiments.training.v6.corpus import load_corpus_manifest

    manifest = Path("manifests/v6_corpus_stage_a_small_v1.json")
    if not manifest.is_file():
        pytest.skip("12-corpus manifest missing")

    db = DBAdapter(integration_db)
    report = await gate.run_gate_async(
        manifest,
        Path("/tmp/dtie_pdb_cache"),
        db,
    )
    if report.structures_checked == 0:
        pytest.skip("no enabled structures in manifest")

  # If DB not populated, fail with actionable message (not skip silently)
    failures = [s for s in report.structures if not s.passed]
    if failures and all(s.n_residues == 0 for s in failures):
        pytest.skip("corpus structures not ingested in test DB")

    assert report.passed, (
        f"P_FEATURE_01 live gate failed on {[f.pdb_id for f in failures]} — "
        "run POST /api/ingest on corpus then make gate-p-feature-01"
    )
