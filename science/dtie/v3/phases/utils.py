"""
utils.py
========
Eidetix Bio — Dynamic Topology Inference Engine v3.0
Utility functions for the DTIE pipeline.
"""

import datetime
import hashlib
import json
import logging
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import shutil
import tempfile
from typing import Tuple


import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from Bio.PDB import PDBParser, PDBIO, Superimposer
from meeko import MoleculePreparation, PDBQTWriterLegacy


def setup_logging(output_dir: Path, verbose: bool = False) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "dtie_run.log"

    fmt = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers = [
        logging.FileHandler(log_path),
        logging.StreamHandler(sys.stdout),
    ]

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
    )
    return logging.getLogger("DTIE_Orchestrator")


def save_results(results: Dict, output_dir: Path, logger: logging.Logger):
    """
    Serialise all pipeline results to JSON. numpy arrays and non-serialisable
    objects are converted to lists/strings so the output is always readable.
    """

    def _serialise(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, Path):
            return str(obj)
        return str(obj)

    out_path = output_dir / "dtie_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=_serialise)
    logger.info(f"Full results → {out_path}")


def _timed(label: str, fn, logger: logging.Logger):
    """Run fn(), log elapsed time, re-raise on failure with phase label."""
    logger.info(f"▶  Starting {label}")
    t0 = time.time()
    try:
        result = fn()
        elapsed = time.time() - t0
        logger.info(f"✓  {label} completed in {elapsed:.1f}s")
        return result
    except Exception as e:
        elapsed = time.time() - t0
        logger.error(f"✗  {label} FAILED after {elapsed:.1f}s: {e}")
        raise


def _summarize_for_log(value: Any, max_items: int = 5, depth: int = 0) -> Any:
    """Return a compact, JSON-safe summary of a payload for phase I/O logging."""
    if value is None:
        return None

    if isinstance(value, np.ndarray):
        return {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        summary: Dict[str, Any] = {}
        keys = list(value.keys())
        for k in keys[:max_items]:
            summary[str(k)] = _summarize_for_log(
                value[k], max_items=max_items, depth=depth + 1
            )
        if len(keys) > max_items:
            summary["..."] = f"{len(keys) - max_items} more keys"
        return {
            "type": "dict",
            "keys": keys,
            "summary": summary,
        }

    if isinstance(value, (list, tuple)):
        sample = [
            _summarize_for_log(v, max_items=max_items, depth=depth + 1)
            for v in value[:max_items]
        ]
        return {
            "type": type(value).__name__,
            "len": len(value),
            "sample": sample,
        }

    # Fallback: avoid noisy reprs for large objects.
    return {"type": type(value).__name__}


def _log_phase_io(
    phase_label: str, direction: str, payload: Any, logger: logging.Logger
):
    """Log a compact phase input/output summary to help trace handoff consistency."""
    summary = _summarize_for_log(payload)
    logger.info(
        f"[PHASE_IO] {phase_label} {direction} :: "
        f"{json.dumps(summary, default=str, ensure_ascii=True)}"
    )


def _assert_required_keys(
    phase_label: str,
    payload: Dict[str, Any],
    required_keys: List[str],
    logger: logging.Logger,
):
    """Fail fast if a phase payload is missing required contract keys."""
    missing = [k for k in required_keys if k not in payload]
    if missing:
        msg = f"{phase_label} missing required keys: {missing}"
        logger.error(msg)
        raise ValueError(msg)


def _count_structure_atoms(ensemble: Any) -> Optional[int]:
    """Best-effort atom count for logging without coupling to wrapper internals."""
    structure = getattr(ensemble, "_structure", None)
    if structure is None:
        return None
    get_atoms = getattr(structure, "get_atoms", None)
    if get_atoms is None:
        return None
    return sum(1 for _ in get_atoms())


def _md5(path: str) -> str:
    """MD5 hex digest of a file — used to anchor results to exact input structures."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_provenance(
    gdp_pdb: str,
    gtp_pdb: str,
    protein_pdb: str,
    library_path: Optional[str],
    checkpoint_path: Optional[str],
    n_landmarks: int,
    curvature_c: float,
    top_n_screen: int,
    top_n_dock: int,
    selectivity_threshold: float,
    effector_sites: Optional[List[int]],
) -> Dict:
    """
    Build a self-describing provenance block for the results JSON.

    The intent is that dtie_results.json is interpretable in isolation —
    without its parent directory — because it carries the complete record of
    what was run, with which inputs, and with which key parameters.
    """

    def _file_entry(path: Optional[str]) -> Optional[Dict]:
        if path is None:
            return None
        p = Path(path)
        entry: Dict = {
            "path": str(p.resolve()),
            "filename": p.name,
        }
        if p.exists():
            entry["md5"] = _md5(str(p))
            entry["size_bytes"] = p.stat().st_size
        return entry

    return {
        "dtie_version": "3.0",
        "run_timestamp_utc": datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "inputs": {
            "gdp_structure": _file_entry(gdp_pdb),
            "gtp_structure": _file_entry(gtp_pdb),
            "protein_structure": _file_entry(protein_pdb),
            "compound_library": _file_entry(library_path),
            "gnn_checkpoint": _file_entry(checkpoint_path),
        },
        "parameters": {
            "n_landmarks": n_landmarks,
            "curvature_c": curvature_c,
            "top_n_screen": top_n_screen,
            "top_n_dock": top_n_dock,
            "selectivity_threshold": selectivity_threshold,
            "effector_sites": effector_sites,
            "tau": 13.0,
            "wrapping_radius_ang": 6.5,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "hostname": platform.node(),
        },
    }


def _align_pdb_to_reference(mobile_pdb: str, reference_pdb: str, out_pdb: str) -> bool:
    """
    Superimpose mobile_pdb onto reference_pdb using shared Cα atoms.
    Writes the aligned structure to out_pdb. Returns True on success.

    This is required when GDP and GTP structures come from different crystal
    datasets (different unit cells / origins) so that the pharmacophore center
    computed in the GDP frame maps correctly onto the GTP receptor during
    Phase 6b/6d docking.
    """
    try:
        from Bio.PDB import PDBParser, PDBIO, Superimposer

        parser = PDBParser(QUIET=True)
        ref = parser.get_structure("ref", reference_pdb)[0]
        mob = parser.get_structure("mob", mobile_pdb)[0]

        # Collect matching Cα pairs by residue number
        ref_ca = {
            r.get_id()[1]: r["CA"]
            for r in ref.get_residues()
            if r.get_id()[0] == " " and "CA" in r
        }
        mob_ca = {
            r.get_id()[1]: r["CA"]
            for r in mob.get_residues()
            if r.get_id()[0] == " " and "CA" in r
        }
        common = sorted(set(ref_ca) & set(mob_ca))
        if len(common) < 10:
            return False

        sup = Superimposer()
        sup.set_atoms([ref_ca[i] for i in common], [mob_ca[i] for i in common])
        sup.apply(mob.get_atoms())

        io = PDBIO()
        io.set_structure(mob)
        io.save(out_pdb)
        return True
    except Exception:
        return False


def _which(name: str) -> Optional[str]:
    """Resolve executable path from system PATH only."""
    return shutil.which(name)


def _write_ligand_pdbqt_meeko(mol: Chem.Mol, out_path: str) -> bool:
    """Prepare ligand PDBQT with Meeko. Returns True on success."""
    try:
        preparator = MoleculePreparation()
        mol_setups = preparator.prepare(mol)
        pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(mol_setups[0])
        if not is_ok:
            pass
        with open(out_path, "w") as f:
            f.write(pdbqt_string)
        return True
    except Exception as e:
        raise RuntimeError(f"Meeko ligand preparation failed: {e}") from e


def _prepare_receptor_pdbqt(protein_pdb_path: str, out_path: str) -> bool:
    """Convert receptor PDB → PDBQT.

    Requires prepare_receptor (MGLTools) for receptor preparation.
    """
    prep = _which("prepare_receptor")
    if prep is None:
        raise RuntimeError("prepare_receptor binary not found")
    result = subprocess.run(
        [prep, "-r", protein_pdb_path, "-o", out_path, "-A", "hydrogens"],
        capture_output=True,
        timeout=120,
    )
    if result.returncode != 0 or not Path(out_path).exists():
        raise RuntimeError("prepare_receptor failed to generate receptor PDBQT")
    return True


def _parse_vina_affinity(output: str) -> Optional[float]:
    """
    Extract best binding affinity (kcal/mol) from Vina stdout.
    """
    lines = output.splitlines()
    sep_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("-----"):
            sep_idx = i
            break
    if sep_idx is None:
        return None
    for line in lines[sep_idx + 1 :]:
        parts = line.split()
        if len(parts) >= 2:
            try:
                return float(parts[1])
            except ValueError:
                continue
    return None
