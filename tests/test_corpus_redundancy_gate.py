"""P_CORPUS_01 — locked Stage A manifest vs frozen TM-align redundancy report."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from science.training.corpus_governance import (
    FROZEN_REPORT,
    LOCKED_MANIFEST,
    STAGE_A_MAX_RESIDUES,
    validate_p_corpus_01,
)

_REPO = Path(__file__).resolve().parents[1]


def _load_locked_artifacts() -> tuple[dict, dict]:
    if not LOCKED_MANIFEST.is_file() or not FROZEN_REPORT.is_file():
        pytest.skip("locked manifest or frozen report not committed")
    manifest = json.loads(LOCKED_MANIFEST.read_text())
    report = json.loads(FROZEN_REPORT.read_text())
    return report, manifest


def test_p_corpus_01_locked_stage_a_manifest_passes() -> None:
    """Frozen report assertion — no TM-align recompute in CI."""
    report, manifest = _load_locked_artifacts()
    errors = validate_p_corpus_01(report, manifest)
    assert errors == [], "P_CORPUS_01 violations:\n" + "\n".join(errors)


def test_p_corpus_01_fails_on_proxy_metric_regression() -> None:
    """Proxy report must not back the locked manifest."""
    report, manifest = _load_locked_artifacts()
    bad = copy.deepcopy(report)
    bad["structural_metric"] = "biotite_ca_proxy"
    bad["status"] = "provisional"
    bad["locking_artifact"] = False
    errors = validate_p_corpus_01(
        bad,
        manifest,
        expected_report_sha256=None,
        expected_manifest_sha256=None,
    )
    assert any("proxy" in e.lower() or "tm-align" in e.lower() for e in errors)


def test_p_corpus_01_fails_on_train_set_drift() -> None:
    """Re-introducing a demoted filler train slot must trip the gate."""
    report, manifest = _load_locked_artifacts()
    drifted = copy.deepcopy(manifest)
    drifted["proteins"].append(
        {
            "pdb_id": "2ABD",
            "chain": "A",
            "gene": "ABD",
            "fold_id": "1.20.80.10",
            "fold_id_tier": "topology",
            "fold_id_source": "pdbe_cache",
            "role": "train",
            "disposition_rule": "drift injection",
            "enabled": True,
            "stage0": False,
        }
    )
    errors = validate_p_corpus_01(
        report,
        drifted,
        expected_report_sha256=None,
        expected_manifest_sha256=None,
    )
    assert any("train set mismatch" in e for e in errors)


def test_p_corpus_01_fails_on_pin_desync_loudly() -> None:
    """JSON committed without updating pin constant must fail with regeneration message."""
    report, manifest = _load_locked_artifacts()
    errors = validate_p_corpus_01(
        report,
        manifest,
        expected_report_sha256="0" * 64,
        expected_manifest_sha256=None,
    )
    pin_errors = [e for e in errors if "pinned SHA256 does not match" in e]
    assert len(pin_errors) == 1
    msg = pin_errors[0]
    assert "pinned SHA256 does not match" in msg
    assert "regeneration incomplete" in msg
    assert "corpus_governance.py" in msg


def test_p_corpus_01_third_gtpase_not_identity_exempt() -> None:
    """Swapping 5VQ2 for 3CON must not get blanket GTPase-fold identity exemption."""
    report, manifest = _load_locked_artifacts()
    bad = copy.deepcopy(report)
    bad["review_dispositions"]["holdout_to_train"]["5VQ2:A"] = (
        "GTPase cap-2 biological-centrality override: fake third slot"
    )
    del bad["review_dispositions"]["holdout_to_train"]["3CON:A"]
    train = [r for r in bad["stage_a_locked"]["train"] if r["structure_key"] != "3CON:A"]
    vq = next(r for r in bad["stage_a_locked"]["eval_holdout"] if r["structure_key"] == "5VQ2:A")
    bad["stage_a_locked"]["eval_holdout"] = [
        r for r in bad["stage_a_locked"]["eval_holdout"] if r["structure_key"] != "5VQ2:A"
    ]
    vq = dict(vq)
    vq["role"] = "train"
    vq["disposition_rule"] = bad["review_dispositions"]["holdout_to_train"]["5VQ2:A"]
    train.append(vq)
    bad["stage_a_locked"]["train"] = train

    errors = validate_p_corpus_01(
        bad,
        manifest,
        expected_report_sha256=None,
        expected_manifest_sha256=None,
    )
    assert any("WITHIN_FOLD_HIGH_IDENTITY among train: 4OBE:A vs 5VQ2:A" in e for e in errors)


def test_p_corpus_01_fails_on_cross_fold_train_leak() -> None:
    """Train pair with CROSS_FOLD_HIGH_TM in frozen report must fail validation."""
    report, manifest = _load_locked_artifacts()
    bad = copy.deepcopy(report)
    # Promote 1SUP alongside 4OBE — known cross-fold high-TM pair in frozen pairs
    train = bad["stage_a_locked"]["train"]
    holdout = bad["stage_a_locked"]["eval_holdout"]
    sup = next(r for r in holdout if r["structure_key"] == "1SUP:A")
    holdout.remove(sup)
    sup = dict(sup)
    sup["role"] = "train"
    train.append(sup)
    bad["stage_a_locked"]["train"] = train
    bad["stage_a_locked"]["eval_holdout"] = holdout
    bad["stage_a_locked"]["train_count"] = len(train)

    drifted = copy.deepcopy(manifest)
    for p in drifted["proteins"]:
        if p["pdb_id"] == "1SUP":
            p["role"] = "train"
            p["enabled"] = True

    errors = validate_p_corpus_01(
        bad,
        drifted,
        expected_report_sha256=None,
        expected_manifest_sha256=None,
    )
    assert any("CROSS_FOLD_HIGH_TM" in e for e in errors)


def test_p_corpus_01_fails_on_wrong_manifest_chain() -> None:
    """Manifest chain that disagrees with author_chain_from_cache must fail loadability."""
    report, manifest = _load_locked_artifacts()
    bad = copy.deepcopy(manifest)
    for entry in bad["proteins"]:
        if entry["pdb_id"] == "1CEW":
            entry["chain"] = "A"
    errors = validate_p_corpus_01(
        report,
        bad,
        expected_report_sha256=None,
        expected_manifest_sha256=None,
    )
    assert any("1CEW:A" in e and "author chain" in e for e in errors)


def test_p_corpus_01_fails_when_locked_structure_exceeds_residue_cap() -> None:
    """4GQB at 625 residues must fail when STAGE_A_MAX_RESIDUES is below that."""
    report, manifest = _load_locked_artifacts()
    errors = validate_p_corpus_01(
        report,
        manifest,
        expected_report_sha256=None,
        expected_manifest_sha256=None,
    )
    # Current floor includes 4GQB — re-run loadability at an impossible cap.
    from science.training.corpus_governance import validate_corpus_loadability

    cap_errors = validate_corpus_loadability(manifest, report, max_residues=600)
    assert any("4GQB:A" in e and "625" in e for e in cap_errors)
    assert errors == [], "locked corpus must pass at STAGE_A_MAX_RESIDUES"
