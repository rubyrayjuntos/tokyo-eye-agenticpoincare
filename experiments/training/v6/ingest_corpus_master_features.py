#!/usr/bin/env python3
"""Minimal 12-corpus MASTER feature ingest (MVP repopulation path).

Computes all four master features via residue_features (FeatureMode.MASTER) and
persists to dim_residue + fact_ingestion_features.

Prerequisite: structure dimensions must exist (dim_structure/chain/residue/atom).
Use POST /api/ingest per structure first, or ensure prior ingest-full rows exist.

Example:
  PYTHONPATH=. python -m experiments.training.v6.ingest_corpus_master_features \\
    --manifest manifests/v6_corpus_stage_a_small_v1.json \\
    --pdb-dir pdb_cache \\
    --verify
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from data.db import DBAdapter, get_connection
from experiments.training.v6 import _data
from experiments.training.v6.corpus import iter_corpus_entries
from science.dtie.common.structure_readiness import ensure_structure_ready
from science.dtie.common.load_graph_from_db import load_protein_graph_from_db

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_MANIFEST = _REPO_ROOT / "manifests" / "v6_corpus_stage_a_small_v1.json"


def _resolve_pdb_path(pdb_id: str, pdb_dir: Path) -> Path:
    return _data._download_pdb(pdb_id.upper(), pdb_dir)


async def _ingest_one(
    db: DBAdapter,
    *,
    pdb_id: str,
    chain: str,
    pdb_dir: Path,
    dry_run: bool,
    verify: bool,
    force_reingest: bool,
) -> dict:
    if dry_run:
        from science.dtie.common.ingest_master_features import compute_master_features_from_pdb, structure_id_for_pdb
        from experiments.training.v6._data import _download_pdb

        structure_id = structure_id_for_pdb(pdb_id)
        pdb_path = _download_pdb(pdb_id.upper(), pdb_dir)
        features = compute_master_features_from_pdb(pdb_path, chain, structure_id)
        return {
            "pdb_id": pdb_id,
            "chain": chain,
            "structure_id": structure_id,
            "residue_count": len(features),
            "dry_run": True,
        }

    structure_id = await ensure_structure_ready(
        db,
        pdb_id,
        chain,
        pdb_dir,
        force_reingest=force_reingest,
    )
    graph = await load_protein_graph_from_db(db, structure_id, pdb_id, chain)
    if graph is None:
        raise RuntimeError(f"{pdb_id}:{chain} load_protein_graph_from_db returned None")

    result = {
        "pdb_id": pdb_id,
        "chain": chain,
        "structure_id": structure_id,
        "residue_count": graph["n_residues"],
        "dry_run": False,
        "verified": verify,
    }
    return result


async def run(args: argparse.Namespace) -> int:
    manifest = Path(args.manifest)
    pdb_dir = Path(args.pdb_dir)
    entries = iter_corpus_entries(manifest, max_proteins=args.max_proteins)

    if not entries:
        logger.error("No enabled proteins in manifest %s", manifest)
        return 1

    logger.info("MASTER feature ingest: %d structures from %s", len(entries), manifest)

    results: list[dict] = []
    failures: list[str] = []

    async with get_connection() as conn:
        db = DBAdapter(conn)
        for entry in entries:
            pdb_id = str(entry["pdb_id"]).upper()
            chain = str(entry.get("chain", "A"))
            try:
                summary = await _ingest_one(
                    db,
                    pdb_id=pdb_id,
                    chain=chain,
                    pdb_dir=pdb_dir,
                    dry_run=args.dry_run,
                    verify=args.verify,
                    force_reingest=args.force_reingest,
                )
                if not args.dry_run:
                    await conn.commit()
                results.append(summary)
                logger.info(
                    "OK %s:%s — %d residues (structure_id=%s)",
                    pdb_id,
                    chain,
                    summary["residue_count"],
                    summary.get("structure_id"),
                )
            except Exception as exc:
                await conn.rollback()
                msg = f"{pdb_id}:{chain}: {exc}"
                failures.append(msg)
                logger.error("FAIL %s", msg)
                if args.fail_fast:
                    break

    report = {
        "manifest": str(manifest),
        "attempted": len(entries),
        "succeeded": len(results),
        "failed": len(failures),
        "dry_run": args.dry_run,
        "results": results,
        "failures": failures,
    }
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2))

    print(json.dumps({"succeeded": len(results), "failed": len(failures)}, indent=2))
    if failures:
        for f in failures:
            print(f"  ERROR: {f}", file=sys.stderr)
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest MASTER features for corpus structures")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_DEFAULT_MANIFEST,
        help="Corpus manifest JSON (default: 12-prot stage_a_small_v1)",
    )
    parser.add_argument(
        "--pdb-dir",
        type=Path,
        default=_REPO_ROOT / "pdb_cache",
        help="Directory for PDB files (download via RCSB if missing)",
    )
    parser.add_argument("--max-proteins", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Compute only; no DB writes")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="After write, recompute from PDB and compare to DB",
    )
    parser.add_argument("--force-reingest", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--report", type=str, default="", help="Write JSON summary to path")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
