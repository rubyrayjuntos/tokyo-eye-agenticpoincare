"""Fetch PDB structures required by the calibration benchmark dataset.

Downloads all structures referenced in data/calibration/benchmark_cryptic_sites.json
from the RCSB PDB archive via BinaryCIF → PDB conversion, storing them in
data/structures/.

Usage:
    PYTHONPATH=. python scripts/fetch_benchmark_structures.py

This is a prerequisite for `make calibrate-cryptic-real`.
"""

from __future__ import annotations

import json
import logging
import sys
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

BENCHMARK_PATH = Path("data/calibration/benchmark_cryptic_sites.json")
STRUCTURES_DIR = Path("data/structures")
RCSB_PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"


def fetch_pdb(pdb_id: str, output_dir: Path) -> Path:
    """Download a PDB file from RCSB if not already cached."""
    pdb_id_upper = pdb_id.upper()
    output_path = output_dir / f"{pdb_id_upper}.pdb"

    if output_path.exists():
        logger.info("  %s — already cached", pdb_id_upper)
        return output_path

    url = RCSB_PDB_URL.format(pdb_id=pdb_id_upper)
    logger.info("  %s — downloading from %s", pdb_id_upper, url)

    try:
        urllib.request.urlretrieve(url, str(output_path))
    except Exception as e:
        logger.error("  FAILED to download %s: %s", pdb_id_upper, e)
        raise

    return output_path


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not BENCHMARK_PATH.exists():
        logger.error("Benchmark file not found: %s", BENCHMARK_PATH)
        sys.exit(1)

    with BENCHMARK_PATH.open() as f:
        sites = json.load(f)

    # Collect unique PDB IDs
    pdb_ids = sorted({site["pdb_id"].upper() for site in sites})
    logger.info("Fetching %d unique PDB structures for calibration benchmark", len(pdb_ids))

    STRUCTURES_DIR.mkdir(parents=True, exist_ok=True)

    failed = []
    for pdb_id in pdb_ids:
        try:
            fetch_pdb(pdb_id, STRUCTURES_DIR)
        except Exception:
            failed.append(pdb_id)

    if failed:
        logger.error("%d structures failed to download: %s", len(failed), failed)
        sys.exit(1)

    logger.info("All %d structures downloaded to %s", len(pdb_ids), STRUCTURES_DIR)

    # Update benchmark JSON pdb_path fields to canonical locations
    updated = False
    for site in sites:
        expected_path = f"data/structures/{site['pdb_id'].upper()}.pdb"
        if site.get("pdb_path") != expected_path:
            site["pdb_path"] = expected_path
            updated = True

    if updated:
        with BENCHMARK_PATH.open("w") as f:
            json.dump(sites, f, indent=2)
        logger.info("Updated pdb_path entries in benchmark JSON")


if __name__ == "__main__":
    main()
