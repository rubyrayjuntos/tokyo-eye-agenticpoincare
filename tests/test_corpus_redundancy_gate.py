"""P_CORPUS_01 — locked Stage A manifest vs frozen TM-align redundancy report."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from science.training.corpus_governance import (
    FROZEN_REPORT,
    LOCKED_MANIFEST,
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
