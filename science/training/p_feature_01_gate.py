"""P_FEATURE_01 — DB round-trip gate for MASTER feature training.

Guards the production training path:
  fresh (SSOT recompute on governed dim_atom — same path as ingest persist)
    ≈ written (fact_ingestion_features persisted at ingest)
    == train_reads (load_graph_from_db → load_protein_graph)

Fresh is NOT read from fact_ingestion_features (that would be tautological).
Fresh is NOT PDB-file-local unless dim_atom is empty — ingest persists from dim_atom.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from science.dtie.common import residue_features as rf
from science.dtie.common.ingest_master_features import (
    DEFAULT_CONDITION,
    structure_id_for_pdb,
)
from science.dtie.common.load_graph_from_db import (
    build_protein_graph_dict,
    fetch_training_rows,
)

logger = logging.getLogger(__name__)

GATE_NAME = "P_FEATURE_01"
DEFAULT_STAMP_PATH = Path("data/gates/p_feature_01_passed.json")
FEATURE_MODULE_PATH = Path("science/dtie/common/residue_features.py")

# Biochemical SS anchors — catches populated-but-wrong (e.g. all-coil) rows.
SS_FOLD_ANCHORS: dict[str, dict[str, float]] = {
    "1TIM": {"min_h_frac": 0.20, "min_e_frac": 0.10},
    "1MBN": {"min_h_frac": 0.65, "max_e_frac": 0.05},
    "1F88": {"min_h_frac": 0.45, "max_e_frac": 0.20},
}

MIN_RESIDUES_FOR_COIL_CHECK = 50


@dataclass(frozen=True)
class StructureGateResult:
    pdb_id: str
    chain: str
    structure_id: str
    n_residues: int
    fresh_matches_written: bool
    written_matches_train: bool
    ss_anchor_ok: bool
    fresh_written_mismatches: list[str] = field(default_factory=list)
    written_train_mismatches: list[str] = field(default_factory=list)
    ss_anchor_errors: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return (
            self.fresh_matches_written
            and self.written_matches_train
            and self.ss_anchor_ok
        )


@dataclass
class GateReport:
    gate: str = GATE_NAME
    passed: bool = False
    manifest_path: str = ""
    manifest_sha256: str = ""
    feature_module_sha256: str = ""
    structures_checked: int = 0
    structures: list[StructureGateResult] = field(default_factory=list)
    checked_at: str = ""
    p_feature_01_passed: bool = False
    # MLflow feature-definition provenance (emitted only when gate passes)
    rho_def: str = "dehydron_wrapping_6.5A"
    tau_def: str = "rho_lt_13.0"
    ss_def: str = "biotite_psea_dssp_class"
    sasa_def: str = "freesasa_heavy_atom_A2"

    def to_stamp(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["structures"] = [
            {
                "pdb_id": s.pdb_id,
                "chain": s.chain,
                "n_residues": s.n_residues,
                "passed": s.passed,
            }
            for s in self.structures
        ]
        return payload


def feature_module_sha256() -> str:
    return hashlib.sha256(FEATURE_MODULE_PATH.read_bytes()).hexdigest()


def manifest_sha256(manifest_path: Path) -> str:
    return hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def sse_distribution(features: Sequence[rf.ResidueNodeFeatures]) -> dict[str, int]:
    counts = {"H": 0, "E": 0, "C": 0}
    for feat in features:
        code = feat.sse_code if feat.sse_code else rf.sse_code_from_ss_type(feat.ss_type)
        counts[code] = counts.get(code, 0) + 1
    return counts


def check_ss_anchors(pdb_id: str, fresh: Sequence[rf.ResidueNodeFeatures]) -> list[str]:
    """Fold-anchored and global non-degenerate SS checks on fresh recompute."""
    errors: list[str] = []
    dist = sse_distribution(fresh)
    n = sum(dist.values())
    if n == 0:
        return [f"{pdb_id}: no residues for SS check"]

    h_frac = dist["H"] / n
    e_frac = dist["E"] / n

    if n >= MIN_RESIDUES_FOR_COIL_CHECK and dist["H"] == 0 and dist["E"] == 0:
        errors.append(
            f"{pdb_id}: all-coil regression (H=0 E=0 C={dist['C']} on {n} residues)"
        )

    anchor = SS_FOLD_ANCHORS.get(pdb_id.upper())
    if anchor:
        if h_frac < anchor.get("min_h_frac", 0.0):
            errors.append(
                f"{pdb_id}: helix fraction {h_frac:.3f} < anchor {anchor['min_h_frac']}"
            )
        if "min_e_frac" in anchor and e_frac < anchor["min_e_frac"]:
            errors.append(
                f"{pdb_id}: sheet fraction {e_frac:.3f} < anchor {anchor['min_e_frac']}"
            )
        if "max_e_frac" in anchor and e_frac > anchor["max_e_frac"]:
            errors.append(
                f"{pdb_id}: sheet fraction {e_frac:.3f} > anchor {anchor['max_e_frac']}"
            )
    return errors


def features_from_train_graph(graph: dict[str, Any], chain: str) -> list[rf.ResidueNodeFeatures]:
    x = graph["data"].x.detach().cpu().numpy()
    feats: list[rf.ResidueNodeFeatures] = []
    for i, rid in enumerate(graph["residue_ids"]):
        idx = int(str(rid).split(":")[1])
        feats.append(
            rf.ResidueNodeFeatures(
                chain_label=chain,
                residue_index=idx,
                rho=float(x[i, 0]),
                tau_flag=float(x[i, 1]),
                ss_type=float(x[i, 2]),
                sasa=float(x[i, 3]),
            )
        )
    return feats


def evaluate_round_trip(
    fresh: Sequence[rf.ResidueNodeFeatures],
    written: Sequence[rf.ResidueNodeFeatures],
    train_reads: Sequence[rf.ResidueNodeFeatures],
    *,
    pdb_id: str = "",
) -> StructureGateResult:
    """Pure round-trip evaluation (unit-testable without DB)."""
    fresh_written = rf.compare_feature_parity(list(fresh), list(written))
    written_train = rf.compare_feature_parity(list(written), list(train_reads))
    ss_errors = check_ss_anchors(pdb_id, fresh) if pdb_id else []

    return StructureGateResult(
        pdb_id=pdb_id,
        chain=fresh[0].chain_label if fresh else "",
        structure_id="",
        n_residues=len(fresh),
        fresh_matches_written=len(fresh_written) == 0,
        written_matches_train=len(written_train) == 0,
        ss_anchor_ok=len(ss_errors) == 0,
        fresh_written_mismatches=[str(m) for m in fresh_written[:10]],
        written_train_mismatches=[str(m) for m in written_train[:10]],
        ss_anchor_errors=ss_errors,
    )


async def fetch_written_features(
    db: Any,
    structure_id: str,
    chain_label: str,
) -> list[rf.ResidueNodeFeatures]:
    """Read persisted ingest facts — ss_type from fact row, not re-derived."""
    rows = await db.fetch_all(
        """
        SELECT r.residue_id, r.residue_index, c.chain_label,
               f.rho, f.tau_flag, f.ss_type, f.sasa, r.sse_code
        FROM dim_residue r
        JOIN dim_chain c ON c.chain_id = r.chain_id
        JOIN fact_ingestion_features f
          ON f.residue_id = r.residue_id
         AND f.structure_id = c.structure_id
         AND f.is_current = TRUE
         AND f.condition = :condition
        WHERE c.structure_id = :structure_id
          AND c.chain_label = :chain_label
        ORDER BY r.residue_index
        """,
        {
            "structure_id": structure_id,
            "chain_label": chain_label,
            "condition": DEFAULT_CONDITION,
        },
    )
    return [
        rf.ResidueNodeFeatures(
            chain_label=chain_label,
            residue_index=int(row["residue_index"]),
            residue_id=str(row["residue_id"]),
            rho=float(row["rho"]),
            tau_flag=float(row["tau_flag"]),
            ss_type=float(row["ss_type"]),
            sasa=float(row["sasa"]),
            sse_code=str(row.get("sse_code") or ""),
        )
        for row in rows
    ]


def resolve_pdb_path(pdb_id: str, pdb_dir: Path) -> Path | None:
    for candidate in (
        pdb_dir / f"{pdb_id.upper()}.pdb",
        pdb_dir / f"{pdb_id.lower()}.pdb",
        Path("science/dtie/assets/benchmark_pdbs") / f"{pdb_id.upper()}.pdb",
    ):
        if candidate.is_file():
            return candidate
    return None


async def check_structure_round_trip(
    db: Any,
    pdb_id: str,
    chain: str,
    pdb_dir: Path,
) -> StructureGateResult:
    """Round-trip: SSOT on governed atoms == persisted facts == training read."""
    from science.dtie.common.structure_readiness import compute_master_features_from_db

    structure_id = structure_id_for_pdb(pdb_id.lower())

    try:
        fresh = await compute_master_features_from_db(db, structure_id, chain)
    except ValueError as exc:
        return StructureGateResult(
            pdb_id=pdb_id,
            chain=chain,
            structure_id=structure_id,
            n_residues=0,
            fresh_matches_written=False,
            written_matches_train=False,
            ss_anchor_ok=False,
            fresh_written_mismatches=[str(exc)],
        )

    written = await fetch_written_features(db, structure_id, chain)
    rows = await fetch_training_rows(db, structure_id, chain)
    graph = build_protein_graph_dict(pdb_id.upper(), chain, rows) if rows else None
    train_feats = features_from_train_graph(graph, chain) if graph else []

    result = evaluate_round_trip(fresh, written, train_feats, pdb_id=pdb_id.upper())
    return StructureGateResult(
        pdb_id=result.pdb_id,
        chain=chain,
        structure_id=structure_id,
        n_residues=result.n_residues,
        fresh_matches_written=result.fresh_matches_written,
        written_matches_train=result.written_matches_train,
        ss_anchor_ok=result.ss_anchor_ok,
        fresh_written_mismatches=result.fresh_written_mismatches,
        written_train_mismatches=result.written_train_mismatches,
        ss_anchor_errors=result.ss_anchor_errors,
    )


async def run_gate_async(
    manifest_path: Path,
    pdb_dir: Path,
    db: Any,
) -> GateReport:
    from experiments.training.v6.corpus import load_corpus_manifest

    manifest_path = manifest_path.resolve()
    data = load_corpus_manifest(manifest_path)
    enabled = [e for e in data.get("proteins", []) if e.get("enabled", True)]

    report = GateReport(
        manifest_path=str(manifest_path),
        manifest_sha256=manifest_sha256(manifest_path),
        feature_module_sha256=feature_module_sha256(),
        checked_at=datetime.now(timezone.utc).isoformat(),
    )

    for entry in enabled:
        pdb_id = str(entry["pdb_id"]).upper()
        chain = str(entry.get("chain", "A"))
        result = await check_structure_round_trip(db, pdb_id, chain, pdb_dir)
        report.structures.append(result)
        if not result.passed:
            logger.error(
                "P_FEATURE_01 FAIL %s:%s fresh≈written=%s written≈train=%s ss=%s",
                pdb_id,
                chain,
                result.fresh_matches_written,
                result.written_matches_train,
                result.ss_anchor_ok,
            )
            for line in result.fresh_written_mismatches[:3]:
                logger.error("  fresh≠written: %s", line)
            for line in result.written_train_mismatches[:3]:
                logger.error("  written≠train: %s", line)
            for line in result.ss_anchor_errors:
                logger.error("  ss: %s", line)

    report.structures_checked = len(report.structures)
    report.passed = all(s.passed for s in report.structures) and report.structures_checked > 0
    report.p_feature_01_passed = report.passed
    return report


def write_gate_stamp(report: GateReport, stamp_path: Path) -> None:
    stamp_path.parent.mkdir(parents=True, exist_ok=True)
    stamp_path.write_text(json.dumps(report.to_stamp(), indent=2, sort_keys=True) + "\n")


def load_gate_stamp(stamp_path: Path) -> dict[str, Any]:
    if not stamp_path.is_file():
        raise FileNotFoundError(f"P_FEATURE_01 gate stamp missing: {stamp_path}")
    return json.loads(stamp_path.read_text())


def validate_gate_stamp(
    stamp: dict[str, Any],
    manifest_path: Path,
    *,
    stamp_path: Path | None = None,
) -> None:
    """Raise if stamp does not authorize training on this manifest."""
    if not stamp.get("p_feature_01_passed"):
        raise RuntimeError("P_FEATURE_01 gate stamp has p_feature_01_passed=false")
    expected_manifest = manifest_sha256(manifest_path.resolve())
    if stamp.get("manifest_sha256") != expected_manifest:
        raise RuntimeError(
            "P_FEATURE_01 stamp manifest hash mismatch — re-run make gate-p-feature-01 "
            f"(stamp={stamp.get('manifest_sha256', '')[:16]}… "
            f"current={expected_manifest[:16]}…)"
        )
    expected_module = feature_module_sha256()
    if stamp.get("feature_module_sha256") != expected_module:
        raise RuntimeError(
            "P_FEATURE_01 stamp feature module hash mismatch — residue_features.py changed; "
            "re-run make gate-p-feature-01"
        )
    if stamp_path and stamp.get("stamp_path") and stamp["stamp_path"] != str(stamp_path):
        pass  # optional path metadata only


def require_p_feature_01_for_training(
    manifest_path: Path,
    *,
    stamp_path: Path = DEFAULT_STAMP_PATH,
) -> dict[str, Any]:
    """Refuse training unless a valid gate stamp exists (production DB path only)."""
    if os.environ.get("TRAINING_LOAD_FROM_PDB", "").lower() in ("1", "true", "yes"):
        logger.warning("TRAINING_LOAD_FROM_PDB=1 — skipping P_FEATURE_01 DB gate (legacy path)")
        return {}
    if os.environ.get("SKIP_P_FEATURE_01_GATE", "").lower() in ("1", "true", "yes"):
        logger.warning("SKIP_P_FEATURE_01_GATE set — P_FEATURE_01 DB gate bypassed (dev only)")
        return {}

    stamp = load_gate_stamp(stamp_path)
    validate_gate_stamp(stamp, manifest_path, stamp_path=stamp_path)
    logger.info(
        "P_FEATURE_01 gate stamp valid (%d structures, checked %s)",
        stamp.get("structures_checked"),
        stamp.get("checked_at"),
    )
    return stamp


def governance_params_from_stamp(stamp: dict[str, Any]) -> dict[str, str]:
    """MLflow params derived from gate stamp — not hand-typed."""
    if not stamp:
        return {"p_feature_01_passed": "false"}
    return {
        "p_feature_01_passed": "true" if stamp.get("p_feature_01_passed") else "false",
        "rho_def": str(stamp.get("rho_def", "")),
        "tau_def": str(stamp.get("tau_def", "")),
        "ss_def": str(stamp.get("ss_def", "")),
        "sasa_def": str(stamp.get("sasa_def", "")),
        "feature_module_sha256": str(stamp.get("feature_module_sha256", "")),
        "p_feature_01_checked_at": str(stamp.get("checked_at", "")),
    }


async def _run_cli_async(args: argparse.Namespace) -> int:
    from data.db import DBAdapter, get_connection

    manifest = Path(args.corpus)
    pdb_dir = Path(args.pdb_dir)
    stamp_path = Path(args.stamp_path)

    async with get_connection() as conn:
        db = DBAdapter(conn)
        report = await run_gate_async(manifest, pdb_dir, db)

    if report.passed:
        payload = report.to_stamp()
        payload["stamp_path"] = str(stamp_path.resolve())
        stamp_path.parent.mkdir(parents=True, exist_ok=True)
        stamp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        logger.info("P_FEATURE_01 PASSED — stamp written to %s", stamp_path)
        return 0

    logger.error("P_FEATURE_01 FAILED — %d/%d structures passed", 
        sum(1 for s in report.structures if s.passed), len(report.structures))
    return 1


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="P_FEATURE_01 DB round-trip gate")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("manifests/v6_corpus_stage_a_small_v1.json"),
    )
    parser.add_argument("--pdb-dir", type=Path, default=Path("/tmp/dtie_pdb_cache"))
    parser.add_argument("--stamp-path", type=Path, default=DEFAULT_STAMP_PATH)
    args = parser.parse_args(argv)
    return asyncio.run(_run_cli_async(args))


if __name__ == "__main__":
    sys.exit(main())
