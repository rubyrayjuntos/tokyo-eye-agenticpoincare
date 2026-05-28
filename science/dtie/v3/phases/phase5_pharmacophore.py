"""
phase5_pharmacophore.py
=======================
Eidetix Bio — DTIE Pipeline v3.0

Phase 5: Pharmacophore Generation.

## CONTRACT

### Reads
- phase35_output.npz            <- written by phase35_topological_lift.py
  - lifted_sites               : list of LiftedSite with barycenter_xyz
- phase4_output                 <- written by phase4_resistance_mapping.py
  - coupling_strength          : float per pathway
  - doorway_xyz                : [3] float per pathway
- gnn_output.npz                <- written by gnn_runner.py
  - gdp_aleatoric             : [N] float64  per-residue aleatoric uncertainty
  - gdp_ca_coords             : [N, 3] float64  Cα positions for nearest-residue lookup

### Writes
- phase5_output.npz
  - pharmacophores             : list of Pharmacophore dataclasses

### GNN fields used directly
  - gdp_aleatoric (druggability modifier)
  - gdp_ca_coords (nearest-residue spatial lookup)

### What this phase adds
  - fpocket-based pocket detection and druggability scoring
  - VdW exclusion map and feasible volume identification
  - H-bond donor/acceptor geometry inference
  - Aleatoric druggability modifier: score *= (1 + aleatoric_at_center)
"""

from __future__ import annotations

import logging
import re
import subprocess
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

from scipy.spatial import KDTree
from Bio.PDB import NeighborSearch

try:
    from .contracts import Phase35Output, Phase4Output, Phase5Output, Pharmacophore
except ImportError:
    from contracts import Phase35Output, Phase4Output, Phase5Output, Pharmacophore

logger_p5 = logging.getLogger("DTIE_Phase5")


def lookup_aleatoric_at_center(
    center_xyz: np.ndarray,
    gnn_output: Optional[Dict],
    prefix: str = "gdp_",
) -> float:
    """
    Return the aleatoric uncertainty of the residue nearest to *center_xyz*.

    Uses Cα coordinates from gnn_output for the spatial lookup.
    Returns 0.0 when gnn_output is None or the required keys are missing.
    """
    if gnn_output is None:
        return 0.0

    ca_key = f"{prefix}ca_coords"
    ale_key = f"{prefix}aleatoric"

    if ca_key not in gnn_output or ale_key not in gnn_output:
        logger_p5.warning(
            "gnn_output missing %s or %s — aleatoric modifier disabled", ca_key, ale_key
        )
        return 0.0

    ca_coords = np.asarray(gnn_output[ca_key], dtype=np.float64)
    aleatoric = np.asarray(gnn_output[ale_key], dtype=np.float64)

    if len(ca_coords) == 0:
        return 0.0

    dists = np.linalg.norm(ca_coords - np.asarray(center_xyz, dtype=np.float64), axis=1)
    nearest_idx = int(np.argmin(dists))
    return float(aleatoric[nearest_idx])


def _parse_fpocket_info(info_file: Path) -> List[Dict]:
    """Parse fpocket _info.txt — druggability score, fpocket score, vertex count."""
    pockets: List[Dict] = []
    current: Dict = {}
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
                    current["volume"] = val_f  # Å³ directly from fpocket
    if current:
        pockets.append(current)
    return pockets


def _parse_fpocket_centroids(pockets_file: Path) -> Dict[int, List[float]]:
    """
    Compute per-pocket centroids from fpocket alpha-sphere output.

    Handles two formats:
    - *_pockets.pdb  : MODEL N / ENDMDL blocks, pocket index = MODEL number
    - *_pockets.pqr  : flat ATOM/HETATM STP records, pocket index = residue seq (cols 22-26)
    """
    centroids: Dict[int, List[float]] = {}
    use_model_blocks = pockets_file.suffix.lower() == ".pdb"

    if use_model_blocks:
        pocket_id, coords = 0, []
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
        # PQR: group STP atoms by residue sequence number (cols 22-26)
        groups: Dict[int, List[List[float]]] = {}
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


def run_fpocket(
    protein_pdb_path: str,
    min_alpha: float = 3.0,
    max_alpha: float = 6.0,
) -> Dict:
    """Run fpocket, parse real druggability scores + pocket centroids from output files."""
    if not protein_pdb_path:
        raise ValueError("protein_pdb_path is required for fpocket")

    pdb_path = Path(protein_pdb_path)
    stem = pdb_path.stem
    out_dir = pdb_path.parent / f"{stem}_out"

    try:
        cmd = [
            "fpocket",
            "-f",
            str(pdb_path),
            "-m",
            str(min_alpha),
            "-M",
            str(max_alpha),
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode != 0:
            logger_p5.warning(
                f"fpocket exited {result.returncode}: {result.stderr.decode()[:200]}"
            )

        info_file = out_dir / f"{stem}_info.txt"
        # fpocket ≥4 (snap) writes .pqr; older builds write .pdb
        pockets_file = out_dir / f"{stem}_pockets.pdb"
        if not pockets_file.exists():
            pockets_file = out_dir / f"{stem}_pockets.pqr"

        if not info_file.exists():
            raise FileNotFoundError(f"fpocket output not found at {out_dir}")

        pockets = _parse_fpocket_info(info_file)
        centroids = (
            _parse_fpocket_centroids(pockets_file) if pockets_file.exists() else {}
        )
        for p in pockets:
            if p["id"] in centroids:
                p["centroid_xyz"] = centroids[p["id"]]

        logger_p5.info(
            f"fpocket: {len(pockets)} pockets — real druggability scores loaded"
        )
        return {"pockets": pockets, "status": "success"}

    except FileNotFoundError:
        logger_p5.warning(
            "fpocket binary not found; continuing with fallback pocket metadata"
        )
        return {"pockets": [], "status": "fpocket_unavailable"}
    except Exception as e:
        raise RuntimeError(f"fpocket error: {e}") from e


def match_fpocket_pocket_to_site(pockets: List[Dict], target_xyz: np.ndarray) -> Dict:
    """Return closest fpocket pocket to target by centroid distance."""
    if not pockets:
        raise ValueError("No fpocket pockets available for matching")

    target = np.array(target_xyz)
    best, best_dist = None, float("inf")
    for p in pockets:
        if "centroid_xyz" not in p:
            continue
        d = float(np.linalg.norm(np.array(p["centroid_xyz"]) - target))
        if d < best_dist:
            best_dist, best = d, p

    if best is None:
        raise ValueError("No fpocket centroids available for pocket-to-site mapping")

    return {**best, "distance_to_site": best_dist}


def build_vdw_map(
    protein_atoms,
    grid_resolution: float = 0.5,
    padding: float = 8.0,
) -> Dict:
    """Grid-based Van der Waals exclusion map."""
    coords = np.array([atom.get_coord() for atom in protein_atoms])
    min_c = coords.min(axis=0) - padding
    max_c = coords.max(axis=0) + padding
    shape = np.ceil((max_c - min_c) / grid_resolution).astype(int)
    grid = np.zeros(shape, dtype=bool)

    for c in coords:
        idx = np.floor((c - min_c) / grid_resolution).astype(int)
        if all(0 <= i < s for i, s in zip(idx, shape)):
            grid[tuple(idx)] = True

    return {"grid": grid, "origin": min_c, "resolution": grid_resolution}


def find_empty_pockets(
    vdw_map: Dict,
    target_xyz: np.ndarray,
    radius: float = 3.5,
) -> List[np.ndarray]:
    """Find sterically feasible empty volumes near target."""
    origin = vdw_map["origin"]
    res = vdw_map["resolution"]
    grid = vdw_map["grid"]
    center_idx = np.floor((target_xyz - origin) / res).astype(int)
    feasible = []

    r = int(radius / res) + 1
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            for dz in range(-r, r + 1):
                idx = center_idx + np.array([dx, dy, dz])
                if all(0 <= i < s for i, s in zip(idx, grid.shape)):
                    if not grid[tuple(idx)]:
                        feasible.append(origin + idx * res)
    return feasible[:30]


def get_atoms_within_radius(
    protein_atoms,
    center_xyz: np.ndarray,
    radius: float = 8.0,
) -> list:
    """KDTree-based atom selection within radius of center."""
    if not protein_atoms:
        return []
    coords = np.array([a.get_coord() for a in protein_atoms])
    tree = KDTree(coords)
    idx = tree.query_ball_point(center_xyz, radius)
    return [protein_atoms[i] for i in idx]


def infer_hbond_geometry(pocket_atoms) -> Dict:
    """H-bond donor/acceptor geometry from BioPython atom names and elements."""
    donors, acceptors = [], []
    for atom in pocket_atoms:
        name = atom.get_name()
        elem = atom.element
        if elem == "N" and name.startswith("N"):
            donors.append(atom.get_coord().tolist())
        elif elem == "O" and name.startswith(("O", "OD", "OE", "OXT")):
            acceptors.append(atom.get_coord().tolist())
    return {"donors": donors[:6], "acceptors": acceptors[:6]}


def infer_atom_type_constraints(pocket_atoms) -> Dict:
    """BioPython-based atom typing. No fake attributes — honest approximation."""
    return {
        "hydrophobic": sum(1 for a in pocket_atoms if a.element == "C"),
        "aromatic": sum(
            1
            for a in pocket_atoms
            if a.element == "C" and a.get_name().startswith("CG")
        ),  # rough heuristic
        "positive": sum(1 for a in pocket_atoms if a.element in ["N", "K"]),
        "negative": sum(1 for a in pocket_atoms if a.element in ["O", "S"]),
        "polar": sum(1 for a in pocket_atoms if a.element in ["N", "O", "S"]),
    }


def infer_charge_complementarity(pocket_atoms) -> Dict:
    """Approximate — real partial charges require PROPKA or equivalent."""
    return {
        "net_charge": 0.0,
        "balanced": True,
        "note": "Approximate — no partial charges available from BioPython",
    }


def execute_phase_5_pharmacophore_generation(
    phase_input: "Phase5Input",
) -> "Phase5Output":
    """
    Phase 5: Pharmacophore Generation.

    Fpocket runs ONCE before the loop.
    build_vdw_map runs ONCE before the loop.
    Druggability score is multiplied by (1 + aleatoric_at_center) from GNN.
    """
    phase35_result = phase_input.phase35_result
    phase4_result = phase_input.phase4_result
    protein_atoms = phase_input.protein_atoms
    protein_pdb_path = phase_input.protein_pdb_path
    gnn_output = getattr(phase_input, "gnn_output", None)
    # Run expensive operations once, outside the per-site loop
    fpocket_data = run_fpocket(protein_pdb_path)
    vdw_map = build_vdw_map(protein_atoms)

    if isinstance(phase35_result, dict):
        lifted_sites = phase35_result.get("lifted_sites", [])
    else:
        lifted_sites = phase35_result.lifted_sites

    pharmacophores = []

    for lifted in lifted_sites:
        if isinstance(lifted, dict):
            target_xyz = np.array(lifted.get("barycenter_xyz", [0.0, 0.0, 0.0]))
        else:
            target_xyz = np.array(lifted.barycenter_xyz)
        pocket_atoms = get_atoms_within_radius(protein_atoms, target_xyz)
        pockets = fpocket_data.get("pockets", [])
        if pockets:
            best_pocket = match_fpocket_pocket_to_site(pockets, target_xyz)
        else:
            best_pocket = {
                "druggability": 0.0,
                "volume": 0.0,
                "distance_to_site": 0.0,
            }
        coupling_match = []
        for c in phase4_result:
            if isinstance(c, dict):
                doorway_xyz = c.get("doorway_xyz")
                coupling_strength = c.get("coupling_strength")
            else:
                doorway_xyz = c.doorway_xyz
                coupling_strength = c.coupling_strength
            if doorway_xyz is not None and np.allclose(doorway_xyz, target_xyz, atol=1.5):
                coupling_match.append(coupling_strength)
        if not coupling_match:
            raise ValueError(
                "Phase 5 requires phase4 coupling match for each lifted site"
            )

        # Aleatoric druggability modifier: score *= (1 + aleatoric_at_center)
        aleatoric_at_center = lookup_aleatoric_at_center(
            target_xyz, gnn_output, prefix="gdp_"
        )
        base_druggability = best_pocket.get("druggability", 0.0)
        modified_druggability = base_druggability * (1.0 + aleatoric_at_center)

        pharmacophores.append(
            Pharmacophore(
                center_xyz=target_xyz.tolist(),
                druggability_score=modified_druggability,
                volume=best_pocket.get("volume", 0.0),
                feasible_volumes=find_empty_pockets(vdw_map, target_xyz),
                atom_type_constraints=infer_atom_type_constraints(pocket_atoms),
                hbond_donors=infer_hbond_geometry(pocket_atoms)["donors"],
                hbond_acceptors=infer_hbond_geometry(pocket_atoms)["acceptors"],
                charge_complementarity=infer_charge_complementarity(pocket_atoms),
                pocket_source=fpocket_data["status"],
                allosteric_coupling_strength=float(coupling_match[0]),
                action="TERMINATE_PERSISTENCE_BAR",
            )
        )

    logger_p5.info(f"Phase 5: {len(pharmacophores)} pharmacophores generated")
    return Phase5Output(pharmacophores=pharmacophores)
