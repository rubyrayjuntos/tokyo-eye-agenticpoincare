"""Surface pocket detector adapter for the binding site scan phase.

Dispatches fpocket (geometry-based pocket detection) on an ingested structure
and parses results into GeometryPocket dataclasses. Handles fpocket
unavailability gracefully by returning an empty list with a warning.

Requirements: 2.1, 2.4
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GeometryPocket:
    """A pocket detected by fpocket or equivalent geometry method."""

    pocket_index: int
    centroid_xyz: tuple[float, float, float]
    volume_angstrom3: float
    druggability_score: float  # fpocket druggability [0, 1]
    residue_ids: list[str]


async def _fetch_pdb_id(structure_id: str, db: Any) -> str | None:
    """Resolve pdb_id from dim_structure."""
    row = await db.fetch_one(
        "SELECT pdb_id FROM dim_structure WHERE structure_id = :structure_id",
        {"structure_id": structure_id},
    )
    if row:
        return row["pdb_id"]
    return None


async def _write_pdb_from_dim_atom(structure_id: str, db: Any, output_path: Path) -> bool:
    """Reconstruct a minimal PDB file from dim_atom coordinates.

    Writes ATOM records sufficient for fpocket to detect pockets.
    Returns True if at least one atom was written.
    """
    rows = await db.fetch_all(
        """
        SELECT a.atom_name, a.element, a.x, a.y, a.z, a.occupancy, a.b_factor,
               r.residue_index, r.residue_name, c.chain_label
        FROM dim_atom a
        JOIN dim_residue r ON r.residue_id = a.residue_id
        JOIN dim_chain c ON c.chain_id = r.chain_id
        WHERE c.structure_id = :structure_id
        ORDER BY c.chain_label, r.residue_index, a.atom_name
        """,
        {"structure_id": structure_id},
    )

    if not rows:
        return False

    with open(output_path, "w") as fh:
        for i, row in enumerate(rows, start=1):
            atom_name = row["atom_name"]
            # PDB format: atom name is left-justified in cols 13-16 if len < 4
            if len(atom_name) < 4:
                atom_name_fmt = f" {atom_name:<3s}"
            else:
                atom_name_fmt = f"{atom_name:<4s}"

            residue_name = row["residue_name"] or "UNK"
            chain = row["chain_label"] or "A"
            res_index = row["residue_index"] or 1

            fh.write(
                f"ATOM  {i:5d} {atom_name_fmt}{residue_name:>3s} "
                f"{chain[0]:1s}{res_index:4d}    "
                f"{row['x']:8.3f}{row['y']:8.3f}{row['z']:8.3f}"
                f"{row.get('occupancy', 1.0):6.2f}{row.get('b_factor', 0.0):6.2f}"
                f"          {(row['element'] or 'C'):>2s}\n"
            )
        fh.write("END\n")

    return True


def _parse_fpocket_info(info_file: Path) -> list[dict[str, Any]]:
    """Parse fpocket _info.txt for druggability score, volume, and pocket id."""
    pockets: list[dict[str, Any]] = []
    current: dict[str, Any] = {}

    import re

    with open(info_file) as fh:
        for line in fh:
            line = line.strip()
            m = re.match(r"Pocket\s+(\d+)\s*:", line)
            if m:
                if current:
                    pockets.append(current)
                current = {"id": int(m.group(1))}
                continue
            if ":" in line and current:
                key, _, val = line.partition(":")
                key = key.strip().lower()
                try:
                    val_f = float(val.strip())
                except ValueError:
                    continue
                if "druggability" in key:
                    current["druggability"] = val_f
                elif key == "score":
                    current["score"] = val_f
                elif "volume" in key and "score" not in key:
                    current["volume"] = val_f

    if current:
        pockets.append(current)
    return pockets


def _parse_fpocket_centroids(pockets_file: Path) -> dict[int, list[float]]:
    """Compute per-pocket centroids from fpocket alpha-sphere output.

    Handles two formats:
    - *_pockets.pdb: MODEL N / ENDMDL blocks
    - *_pockets.pqr: flat ATOM/HETATM STP records grouped by residue seq
    """
    centroids: dict[int, list[float]] = {}
    use_model_blocks = pockets_file.suffix.lower() == ".pdb"

    if use_model_blocks:
        pocket_id = 0
        coords: list[list[float]] = []
        with open(pockets_file) as fh:
            for line in fh:
                if line.startswith("MODEL"):
                    pocket_id += 1
                    coords = []
                elif line.startswith("ENDMDL"):
                    if coords:
                        centroids[pocket_id] = np.array(coords).mean(axis=0).tolist()
                elif line.startswith(("HETATM", "ATOM")):
                    try:
                        coords.append(
                            [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                        )
                    except ValueError:
                        pass
    else:
        # PQR format: group STP atoms by residue sequence number
        groups: dict[int, list[list[float]]] = {}
        with open(pockets_file) as fh:
            for line in fh:
                if not line.startswith(("ATOM", "HETATM")):
                    continue
                try:
                    pid = int(line[22:26])
                    groups.setdefault(pid, []).append(
                        [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                    )
                except ValueError:
                    pass
        centroids = {
            pid: np.array(c).mean(axis=0).tolist() for pid, c in groups.items()
        }

    return centroids


def _parse_fpocket_residues(pockets_dir: Path, structure_id: str) -> dict[int, list[str]]:
    """Extract residue IDs per pocket from individual pocket PDB files.

    fpocket writes pocket{N}_atm.pdb files containing the protein atoms
    lining each pocket. We extract unique residue identifiers from these.
    """
    from science.dtie.common.keys import make_residue_id

    residue_map: dict[int, list[str]] = {}

    import re

    for pocket_file in sorted(pockets_dir.glob("pocket*_atm.pdb")):
        m = re.match(r"pocket(\d+)_atm\.pdb", pocket_file.name)
        if not m:
            continue
        pocket_id = int(m.group(1))
        residue_ids: set[str] = set()

        with open(pocket_file) as fh:
            for line in fh:
                if line.startswith(("ATOM", "HETATM")):
                    try:
                        chain = line[21:22].strip() or "A"
                        res_seq = int(line[22:26])
                        residue_ids.add(
                            make_residue_id(structure_id, chain, res_seq)
                        )
                    except (ValueError, IndexError):
                        pass

        residue_map[pocket_id] = sorted(residue_ids)

    return residue_map


def _run_fpocket_subprocess(
    pdb_path: Path,
    structure_id: str,
    min_alpha: float = 3.0,
    max_alpha: float = 6.0,
) -> dict[str, Any]:
    """Run fpocket binary and return parsed pocket data.

    Returns:
        {"pockets": list[dict], "status": "success"|"fpocket_unavailable"|"error"}
    """
    stem = pdb_path.stem
    out_dir = pdb_path.parent / f"{stem}_out"

    try:
        cmd = [
            "fpocket",
            "-f", str(pdb_path),
            "-m", str(min_alpha),
            "-M", str(max_alpha),
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode != 0:
            logger.warning(
                "fpocket exited %d: %s", result.returncode, result.stderr.decode()[:200]
            )

        info_file = out_dir / f"{stem}_info.txt"
        # fpocket ≥4 writes .pqr; older builds write .pdb
        pockets_file = out_dir / f"{stem}_pockets.pdb"
        if not pockets_file.exists():
            pockets_file = out_dir / f"{stem}_pockets.pqr"

        if not info_file.exists():
            logger.warning("fpocket output not found at %s", out_dir)
            return {"pockets": [], "status": "error"}

        pockets = _parse_fpocket_info(info_file)
        centroids = (
            _parse_fpocket_centroids(pockets_file) if pockets_file.exists() else {}
        )

        # Pocket-lining residues from individual pocket files
        pockets_subdir = out_dir / "pockets"
        if not pockets_subdir.exists():
            pockets_subdir = out_dir
        residue_map = _parse_fpocket_residues(pockets_subdir, structure_id)

        for p in pockets:
            pid = p["id"]
            if pid in centroids:
                p["centroid_xyz"] = centroids[pid]
            if pid in residue_map:
                p["residue_ids"] = residue_map[pid]

        return {"pockets": pockets, "status": "success"}

    except FileNotFoundError:
        logger.warning(
            "fpocket binary not found; geometry-based pocket detection unavailable"
        )
        return {"pockets": [], "status": "fpocket_unavailable"}
    except subprocess.TimeoutExpired:
        logger.warning("fpocket timed out after 120s")
        return {"pockets": [], "status": "timeout"}
    except Exception as e:
        logger.warning("fpocket error: %s", e)
        return {"pockets": [], "status": "error"}


async def detect_surface_pockets(
    structure_id: str,
    db: Any,
) -> list[GeometryPocket]:
    """Detect surface pockets for a structure using fpocket.

    Dispatches fpocket on the structure by reconstructing a PDB file from
    dim_atom coordinates and running the fpocket binary. Parses results into
    GeometryPocket dataclasses.

    Handles fpocket unavailability gracefully by returning an empty list
    with a logged warning.

    Requirements: 2.1, 2.4
    """
    # Create a temporary PDB file from the stored atomic coordinates
    tmp_dir = tempfile.mkdtemp(prefix="fpocket_scan_")
    pdb_path = Path(tmp_dir) / f"{structure_id}.pdb"

    try:
        wrote = await _write_pdb_from_dim_atom(structure_id, db, pdb_path)
        if not wrote:
            logger.warning(
                "No atom data found for structure %s; skipping pocket detection",
                structure_id,
            )
            return []

        # Run fpocket
        fpocket_result = _run_fpocket_subprocess(pdb_path, structure_id)

        if fpocket_result["status"] == "fpocket_unavailable":
            logger.warning(
                "fpocket unavailable; returning empty pocket list for %s", structure_id
            )
            return []

        if fpocket_result["status"] in ("error", "timeout"):
            logger.warning(
                "fpocket %s for structure %s; returning empty pocket list",
                fpocket_result["status"],
                structure_id,
            )
            return []

        # Convert raw pocket dicts to GeometryPocket dataclasses
        geometry_pockets: list[GeometryPocket] = []
        for pocket in fpocket_result["pockets"]:
            centroid = pocket.get("centroid_xyz")
            if centroid is None:
                # Skip pockets without a computable centroid
                continue

            geometry_pockets.append(
                GeometryPocket(
                    pocket_index=pocket["id"],
                    centroid_xyz=(
                        float(centroid[0]),
                        float(centroid[1]),
                        float(centroid[2]),
                    ),
                    volume_angstrom3=float(pocket.get("volume", 0.0)),
                    druggability_score=float(pocket.get("druggability", 0.0)),
                    residue_ids=pocket.get("residue_ids", []),
                )
            )

        logger.info(
            "Detected %d surface pockets for structure %s",
            len(geometry_pockets),
            structure_id,
        )
        return geometry_pockets

    finally:
        # Clean up temporary files
        import shutil

        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass
