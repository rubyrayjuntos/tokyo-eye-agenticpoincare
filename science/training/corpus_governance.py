"""Frozen-report validation for locked Stage A corpus (P_CORPUS_01)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
LOCKED_MANIFEST = _REPO / "manifests" / "v6_corpus_stage_a.json"
FROZEN_REPORT = _REPO / "manifests" / "corpus_redundancy_report.json"

# Pinned at lock commit — any report/manifest edit must update these deliberately.
FROZEN_REPORT_SHA256 = "17ca829315964393ec9d4654254523eff1104c46edc86b7ebd2ac36c2741e1ea"
LOCKED_MANIFEST_SHA256 = "e35bac9955d774fd7824ededb8e0d1dab9e21be4b365b66fbfc97cafcd976e92"

TM_ALIGN_METRICS = frozenset({"tm_align", "tm_align_binary", "tm_align_tmtools"})
FORBIDDEN_STRUCTURAL_METRICS = frozenset({"biotite_ca_proxy"})

# Structure-specific identity exemption (GTPase cap-2) — not fold-wide.
GTPASE_CAP2_PROMOTED = "3CON:A"
GTPASE_CAP2_ANCHOR = "4OBE:A"
GTPASE_CAP2_FOLD = "3.40.50.300"
GTPASE_CAP2_REASON_PREFIX = "GTPase cap-2 biological-centrality override"


def _pin_mismatch_message(
    *,
    artifact_label: str,
    path: Path,
    expected: str,
    actual: str,
) -> str:
    return (
        f"P_CORPUS_01: pinned SHA256 does not match committed {artifact_label} — "
        f"regeneration incomplete. "
        f"File {path.relative_to(_REPO)} digest is {actual}, but "
        f"science/training/corpus_governance.py expects {expected}. "
        f"Update FROZEN_REPORT_SHA256 / LOCKED_MANIFEST_SHA256 in the same commit as the JSON artifacts."
    )


def _is_documented_identity_exemption(a: str, b: str, report: dict[str, Any]) -> bool:
    """Only the locked 3CON+4OBE cap-2 pair — not any GTPase-fold promotion."""
    pair = frozenset({a, b})
    if pair != frozenset({GTPASE_CAP2_PROMOTED, GTPASE_CAP2_ANCHOR}):
        return False
    review = report.get("review_dispositions") or {}
    promotions = review.get("holdout_to_train") or {}
    reason = promotions.get(GTPASE_CAP2_PROMOTED, "")
    if not str(reason).startswith(GTPASE_CAP2_REASON_PREFIX):
        return False
    override = (review.get("fold_cap_overrides") or {}).get(GTPASE_CAP2_FOLD)
    if not isinstance(override, dict):
        return False
    if override.get("rule") != "biological_centrality_override":
        return False
    if int(override.get("max_train", 0)) != 2:
        return False
    locked = report.get("stage_a_locked") or {}
    train_keys = {
        structure_key(p["pdb_id"], p["chain"]) for p in locked.get("train", [])
    }
    if GTPASE_CAP2_PROMOTED not in train_keys or GTPASE_CAP2_ANCHOR not in train_keys:
        return False
    return True


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def structure_key(pdb_id: str, chain: str) -> str:
    return f"{str(pdb_id).upper()}:{str(chain).upper()}"


def validate_p_corpus_01(
    report: dict[str, Any],
    manifest: dict[str, Any],
    *,
    expected_report_sha256: str | None = FROZEN_REPORT_SHA256,
    expected_manifest_sha256: str | None = LOCKED_MANIFEST_SHA256,
    report_path: Path = FROZEN_REPORT,
    manifest_path: Path = LOCKED_MANIFEST,
) -> list[str]:
    """Return violation messages (empty = pass). No TM-align recompute."""
    errors: list[str] = []

    if expected_report_sha256 and report_path.is_file():
        digest = _sha256(report_path)
        if digest != expected_report_sha256:
            errors.append(
                _pin_mismatch_message(
                    artifact_label="corpus redundancy report",
                    path=report_path,
                    expected=expected_report_sha256,
                    actual=digest,
                )
            )

    if expected_manifest_sha256 and manifest_path.is_file():
        digest = _sha256(manifest_path)
        if digest != expected_manifest_sha256:
            errors.append(
                _pin_mismatch_message(
                    artifact_label="Stage A locked manifest",
                    path=manifest_path,
                    expected=expected_manifest_sha256,
                    actual=digest,
                )
            )

    metric = str(report.get("structural_metric", ""))
    if metric in FORBIDDEN_STRUCTURAL_METRICS:
        errors.append(
            f"P_CORPUS_01: structural_metric must not be proxy ({metric}) — "
            "a biotite_ca_proxy report cannot back the locked manifest; "
            "regenerate with TM-align (tmtools) and re-lock."
        )
    elif metric not in TM_ALIGN_METRICS:
        errors.append(f"structural_metric not TM-align-backed: {metric}")

    if report.get("status") != "locked":
        errors.append(f"report status must be locked, got {report.get('status')!r}")
    if not report.get("locking_artifact"):
        errors.append("report locking_artifact must be true")
    if report.get("superseded_by"):
        errors.append(f"report must not be superseded_by {report.get('superseded_by')!r}")

    if not report.get("pass"):
        errors.append("report pass must be true")

    locked = report.get("stage_a_locked")
    if not isinstance(locked, dict):
        errors.append("report missing stage_a_locked")
        return errors

    if locked.get("violations"):
        errors.append(f"stage_a_locked violations: {locked['violations']}")

    manifest_metric = manifest.get("structural_metric")
    if manifest_metric and manifest_metric != metric:
        errors.append(
            f"manifest structural_metric {manifest_metric!r} != report {metric!r}"
        )

    if manifest.get("source_report") != "corpus_redundancy_report.json":
        errors.append("manifest source_report must be corpus_redundancy_report.json")

    manifest_train = {
        structure_key(p["pdb_id"], p["chain"])
        for p in manifest.get("proteins", [])
        if p.get("enabled") and p.get("role") == "train"
    }
    report_train = {
        structure_key(p["pdb_id"], p["chain"]) for p in locked.get("train", [])
    }
    if manifest_train != report_train:
        errors.append(
            f"train set mismatch manifest={sorted(manifest_train)} report={sorted(report_train)}"
        )

    identity_threshold = float(
        manifest.get("max_sequence_identity_pct", report.get("sequence_identity_threshold_pct", 30))
    )
    report_identity = float(report.get("sequence_identity_threshold_pct", identity_threshold))
    if identity_threshold != report_identity:
        errors.append(
            f"identity threshold mismatch manifest={identity_threshold} report={report_identity}"
        )

    train_by_key = {structure_key(p["pdb_id"], p["chain"]): p for p in locked.get("train", [])}
    for key, row in train_by_key.items():
        if not row.get("fold_id"):
            errors.append(f"train {key} missing fold_id")
        tier = row.get("fold_id_tier")
        if tier not in ("topology", "architecture"):
            errors.append(f"train {key} invalid fold_id_tier: {tier!r}")

    max_per_fold = int(report.get("max_train_per_fold_id", 2))
    fold_counts: dict[str, int] = {}
    for row in train_by_key.values():
        fid = str(row.get("fold_id", "UNVERIFIED"))
        fold_counts[fid] = fold_counts.get(fid, 0) + 1
    for fid, count in fold_counts.items():
        if count > max_per_fold:
            errors.append(f"FOLD_TRAIN_CAP_EXCEEDED {fid}: {count} > {max_per_fold}")

    train_keys = set(train_by_key)
    for pair in report.get("pairs", []):
        if pair.get("decision") != "CROSS_FOLD_HIGH_TM":
            continue
        a, b = pair.get("a"), pair.get("b")
        if a in train_keys and b in train_keys:
            errors.append(
                f"CROSS_FOLD_HIGH_TM among train: {a} vs {b} tm={pair.get('tm_score')}"
            )

    for pair in report.get("pairs", []):
        if pair.get("decision") != "WITHIN_FOLD_HIGH_IDENTITY":
            continue
        a, b = pair.get("a"), pair.get("b")
        if a not in train_keys or b not in train_keys:
            continue
        if _is_documented_identity_exemption(a, b, report):
            continue
        errors.append(
            f"WITHIN_FOLD_HIGH_IDENTITY among train: {a} vs {b} "
            f"identity={pair.get('sequence_identity_pct')}%"
        )

    return errors
