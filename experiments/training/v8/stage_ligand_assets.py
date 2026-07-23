#!/usr/bin/env python3
"""Stage PDBBind ligand MOL2/SDF assets into data/pdbbind/ligands/ (Sprint 10.1.1).

Supports PDBBind refined-set layout ``{pdb}/{pdb}_ligand.mol2`` or flat ``{pdb}.mol2``.
Missing IDs are reported; staging does not hard-fail the whole run.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage ligand mol2/sdf for Sprint 10.1.1")
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/v8_pdbbind_refined_cluster30_v1.json"),
    )
    p.add_argument(
        "--src",
        type=Path,
        default=None,
        help="PDBBind refined-set root (contains per-pdb folders)",
    )
    p.add_argument(
        "--src-ligands",
        type=Path,
        default=None,
        help="Flat directory of {pdb}.mol2 / {pdb}.sdf",
    )
    p.add_argument("--dst", type=Path, default=Path("data/pdbbind/ligands"))
    p.add_argument("--mode", choices=("symlink", "copy"), default="symlink")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _ids_from_manifest(path: Path) -> list[str]:
    data = json.loads(path.read_text())
    ids: list[str] = []
    seen: set[str] = set()
    for key in ("train", "val", "core_test"):
        for row in data.get(key, []):
            pid = str(row["pdb_id"]).lower()
            if pid not in seen:
                seen.add(pid)
                ids.append(pid)
    return ids


def _find_src(pid: str, src: Path | None, src_ligands: Path | None) -> Path | None:
    cands: list[Path] = []
    if src_ligands is not None:
        for ext in (".mol2", ".sdf"):
            cands.append(src_ligands / f"{pid}{ext}")
    if src is not None:
        folder = src / pid
        cands.extend(
            [
                folder / f"{pid}_ligand.mol2",
                folder / f"{pid}_ligand.sdf",
                folder / f"{pid}.mol2",
                folder / f"{pid}.sdf",
                src / f"{pid}.mol2",
                src / f"{pid}.sdf",
            ]
        )
    for c in cands:
        if c.is_file() and c.stat().st_size > 0:
            return c
    return None


def main() -> None:
    args = parse_args()
    ids = _ids_from_manifest(args.manifest)
    args.dst.mkdir(parents=True, exist_ok=True)
    n_mol2 = n_sdf = n_miss = 0
    missing: list[str] = []
    for pid in ids:
        src_path = _find_src(pid, args.src, args.src_ligands)
        if src_path is None:
            n_miss += 1
            missing.append(pid)
            continue
        ext = src_path.suffix.lower()
        dst = args.dst / f"{pid}{ext}"
        if ext == ".mol2":
            n_mol2 += 1
        else:
            n_sdf += 1
        if args.dry_run:
            continue
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if args.mode == "symlink":
            dst.symlink_to(src_path.resolve())
        else:
            shutil.copy2(src_path, dst)
    report = {
        "n_ids": len(ids),
        "n_mol2": n_mol2,
        "n_sdf": n_sdf,
        "n_missing": n_miss,
        "coverage": (n_mol2 + n_sdf) / max(len(ids), 1),
        "dst": str(args.dst),
        "dry_run": bool(args.dry_run),
        "missing_examples": missing[:20],
    }
    print(json.dumps(report, indent=2))
    (args.dst / "_stage_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
