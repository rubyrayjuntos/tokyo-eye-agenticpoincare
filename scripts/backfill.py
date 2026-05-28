#!/usr/bin/env python
"""Backfill CLI — import historical data into the governed layer.

Usage:
    python scripts/backfill.py structures 4OBE 7XKJ    # Ingest structures from RCSB
    python scripts/backfill.py structures --file list.txt  # From file
    python scripts/backfill.py dry-run 4OBE             # Show what would be written
    python scripts/backfill.py status                    # Show backfill progress

All backfills:
- Are idempotent (safe to re-run)
- Create synthetic provenance records (run_type='historical_backfill')
- Report what was written vs skipped
- Can be run in dry-run mode

See: data/DATA_POPULATION_AND_BACKFILL_STRATEGY.md
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path


async def backfill_structures(pdb_ids: list[str], dry_run: bool = False) -> None:
    """Ingest structures from RCSB into the governed dimensional model."""
    from data.db import DBAdapter, get_connection
    from science.dtie.common.ingestion import StructureIngestor

    print(f"{'[DRY RUN] ' if dry_run else ''}Backfilling {len(pdb_ids)} structures...")

    async with get_connection() as conn:
        db = DBAdapter(conn)
        ingestor = StructureIngestor(db=db)

        for pdb_id in pdb_ids:
            if dry_run:
                print(f"  Would ingest: {pdb_id}")
                continue

            try:
                result = await ingestor.ingest_pdb(pdb_id, include_atoms=True)
                if result.warnings and "Already ingested" in result.warnings:
                    print(f"  ⊘ {pdb_id} — already ingested")
                else:
                    print(
                        f"  ✓ {pdb_id} → {result.structure_id} "
                        f"({result.chains_created} chains, {result.residues_created} residues)"
                    )
            except Exception as e:
                print(f"  ✗ {pdb_id} — failed: {e}")

    print("Done.")


async def show_status() -> None:
    """Show current backfill status from the governed layer."""
    from data.db import DBAdapter, get_connection

    async with get_connection() as conn:
        db = DBAdapter(conn)

        structures = await db.fetch_one(
            "SELECT COUNT(*) as cnt FROM dim_structure", {}
        )
        residues = await db.fetch_one(
            "SELECT COUNT(*) as cnt FROM dim_residue", {}
        )
        embeddings = await db.fetch_one(
            "SELECT COUNT(*) as cnt FROM fact_gnn_node_embedding", {}
        )
        runs = await db.fetch_one(
            "SELECT COUNT(*) as cnt FROM provenance_run", {}
        )
        assets = await db.fetch_one(
            "SELECT COUNT(*) as cnt FROM governed_asset", {}
        )

    print("=== Governed Data Layer Status ===")
    print(f"  Structures:  {structures['cnt'] if structures else 0}")
    print(f"  Residues:    {residues['cnt'] if residues else 0}")
    print(f"  Embeddings:  {embeddings['cnt'] if embeddings else 0}")
    print(f"  Runs:        {runs['cnt'] if runs else 0}")
    print(f"  Assets:      {assets['cnt'] if assets else 0}")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    if command == "structures":
        if "--file" in sys.argv:
            idx = sys.argv.index("--file")
            file_path = Path(sys.argv[idx + 1])
            pdb_ids = [line.strip() for line in file_path.read_text().splitlines() if line.strip()]
        else:
            pdb_ids = sys.argv[2:]

        if not pdb_ids:
            print("No PDB IDs provided.")
            sys.exit(1)

        asyncio.run(backfill_structures(pdb_ids))

    elif command == "dry-run":
        pdb_ids = sys.argv[2:]
        asyncio.run(backfill_structures(pdb_ids, dry_run=True))

    elif command == "status":
        asyncio.run(show_status())

    else:
        print(f"Unknown command: {command}")
        print("Commands: structures, dry-run, status")
        sys.exit(1)


if __name__ == "__main__":
    main()
