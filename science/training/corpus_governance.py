"""Frozen-report validation for locked Stage A corpus (P_CORPUS_01)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
LOCKED_MANIFEST = _REPO / "manifests" / "v6_corpus_stage_a.json"
FROZEN_REPORT = _REPO / "manifests" / "corpus_redundancy_report.json"

# Pinned at lock commit — any report/manifest edit must update these deliberately.
FROZEN_REPORT_SHA256 = "0ffc74d4a954ded3dbc022b7cc4865e10c153ac45437425abef43eabf573932d"
LOCKED_MANIFEST_SHA256 = "d5d7e128f69dabb7be5f940e80d845badcfe60f291eaf3a6c7895cdc2586b96e"

TM_ALIGN_METRICS = frozenset({"tm_align", "tm_align_binary", "tm_align_tmtools"})
FORBIDDEN_STRUCTURAL_METRICS = frozenset({"biotite_ca_proxy"})

# Structure-specific identity exemption (GTPase cap-2) — not fold-wide.
GTPASE_CAP2_PROMOTED = "3CON:A"
GTPASE_CAP2_ANCHOR = "4OBE:A"
GTPASE_CAP2_FOLD = "3.40.50.300"
GTPASE_CAP2_REASON_PREFIX = "GTPase cap-2 biological-centrality override"

# Stage A train ceiling — smoke, curriculum, and P_CORPUS_01 loadability share this value.
# Locked Stage A max today is 4GQB (625); 650 was historical floor + margin.
# Raised to 1200 so corpus expand can include full-length single chains (kinases,
# multi-domain, larger receptors) without a separate MAX_RESIDUES override on every run.
# Training is per-protein (OOM skips one graph); raise further via MAX_RESIDUES= if needed.
STAGE_A_MAX_RESIDUES = 1200
STAGE_A_TRAIN_STRUCTURE_COUNT = 25


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


def is_locked_stage_a_manifest_path(path: Path) -> bool:
    try:
        return path.resolve() == LOCKED_MANIFEST.resolve()
    except OSError:
        return path.name == LOCKED_MANIFEST.name


def stage_a_train_fold_count(manifest: dict[str, Any]) -> int:
    """Distinct CATH fold_ids among enabled train structures (cap-2 pair shares one fold)."""
    folds: set[str] = set()
    for entry in manifest.get("proteins", []):
        if not entry.get("enabled", True) or entry.get("role") != "train":
            continue
        fold_id = str(entry.get("fold_id", "")).strip()
        if fold_id:
            folds.add(fold_id)
    return len(folds)


def validate_corpus_loadability(
    manifest: dict[str, Any],
    report: dict[str, Any],
    *,
    max_residues: int = STAGE_A_MAX_RESIDUES,
    pdb_dir: Path | None = None,
) -> list[str]:
    """Assert locked train structures are loader-resolvable and within STAGE_A_MAX_RESIDUES."""
    from experiments.training.v6.cath_coverage_probe import author_chain_from_cache

    errors: list[str] = []
    locked = report.get("stage_a_locked") or {}
    train_by_key = {
        structure_key(row["pdb_id"], row["chain"]): row for row in locked.get("train", [])
    }
    train_entries = [
        entry
        for entry in manifest.get("proteins", [])
        if entry.get("enabled", True) and entry.get("role") == "train"
    ]

    for entry in train_entries:
        pdb_id = str(entry["pdb_id"]).upper()
        chain = str(entry.get("chain", "A"))
        key = structure_key(pdb_id, chain)
        author = author_chain_from_cache(pdb_id, chain)
        if author is not None and author.upper() != chain.upper():
            errors.append(
                f"P_CORPUS_01 loadability: {key} manifest chain {chain!r} != author chain "
                f"{author!r} from PDBe cache — locked manifest must store the loader-resolvable "
                f"author chain_id, not struct_asym_id alone."
            )
            continue

        report_row = train_by_key.get(key)
        if not report_row:
            errors.append(f"P_CORPUS_01 loadability: {key} missing from stage_a_locked.train")
            continue
        seq_len = int(report_row.get("sequence_length") or 0)
        if seq_len <= 0:
            errors.append(f"P_CORPUS_01 loadability: {key} missing sequence_length in frozen report")
        elif seq_len > max_residues:
            errors.append(
                f"P_CORPUS_01 loadability: {key} sequence_length={seq_len} > "
                f"STAGE_A_MAX_RESIDUES={max_residues} — locked structure would be excluded "
                f"by residue cap; raise STAGE_A_MAX_RESIDUES or demote the structure."
            )

        if pdb_dir is None or not Path(pdb_dir).is_dir():
            continue
        from experiments.training.v6._data import load_protein_graph

        prot = load_protein_graph(pdb_id, chain, Path(pdb_dir))
        n_res = int(prot.get("n_residues", 0)) if prot else 0
        if n_res <= 0:
            errors.append(
                f"P_CORPUS_01 loadability: {key} loads to 0 residues under pdb_dir={pdb_dir}"
            )
        elif n_res > max_residues:
            errors.append(
                f"P_CORPUS_01 loadability: {key} loads {n_res} residues > "
                f"STAGE_A_MAX_RESIDUES={max_residues}"
            )

    return errors


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

    pdb_dir = _REPO / "pdb_cache"
    errors.extend(
        validate_corpus_loadability(
            manifest,
            report,
            max_residues=STAGE_A_MAX_RESIDUES,
            pdb_dir=pdb_dir if pdb_dir.is_dir() else None,
        )
    )

    return errors
